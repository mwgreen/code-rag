"""
File watcher for automatic incremental reindexing.

Uses macOS FSEvents (via watchdog) to detect file changes and reindex affected
files in batches. Designed to handle burst scenarios like git checkout without
degrading search performance.

Pipeline: FSEvents -> watchdog thread -> IndexRules filter -> asyncio.Queue -> debounce -> batch

FSEvents is not a reliable source of truth: it drops events under heavy bursts
and cannot report what happened while the server was down. So the watcher also
runs a reconcile job (disk vs Milvus vs FTS) at start, after .git/HEAD changes,
after .ragignore/.ragconfig changes, and on a timer.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import jobs
import rag_milvus
from indexing_rules import IndexRules

logger = logging.getLogger("code-rag.watcher")

# --- Configuration ---

_watcher_config = {
    'enabled': os.getenv('CODE_RAG_WATCH', 'true').lower() in ('true', '1', 'yes'),
    'debounce_seconds': float(os.getenv('CODE_RAG_WATCH_DEBOUNCE', '2.0')),
    'max_batch_size': int(os.getenv('CODE_RAG_WATCH_MAX_BATCH', '100')),
    'git_settle_seconds': float(os.getenv('CODE_RAG_WATCH_GIT_SETTLE', '3.0')),
    # Full disk/index/FTS reconcile on this interval (seconds). 0 disables the timer.
    'reconcile_interval': float(os.getenv('CODE_RAG_RECONCILE_INTERVAL', str(6 * 3600))),
}


# --- FileChangeHandler ---

class FileChangeHandler(FileSystemEventHandler):
    """Watchdog event handler that filters through IndexRules and forwards to asyncio."""

    def __init__(self, rules: IndexRules, change_queue: asyncio.Queue,
                 loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.rules = rules
        self.change_queue = change_queue
        self.loop = loop

    def _classify(self, path: str, deleted: bool = False) -> Optional[str]:
        """'config' | 'git' | 'file' | None. Runs in the watchdog thread; must be cheap."""
        if self.rules.is_config_file(path):
            return 'config'
        if self.rules.is_git_ref_file(path):
            return 'git'
        if deleted:
            # The file is gone, so skip the stat/content checks; anything that
            # would otherwise have been indexable must be removed.
            reason = self.rules.exclusion_reason(path, size=0)
            return 'file' if reason in (None, 'missing') else None
        return 'file' if self.rules.should_index(path) else None

    def _enqueue(self, action: str, path: str):
        try:
            self.loop.call_soon_threadsafe(self.change_queue.put_nowait, (action, path))
        except RuntimeError:
            pass  # loop closed during shutdown

    def _handle(self, path: str, action: str):
        kind = self._classify(path, deleted=(action == 'deleted'))
        if kind == 'file':
            self._enqueue(action, path)
        elif kind is not None:
            self._enqueue(kind, path)

    def on_modified(self, event):
        if not event.is_directory:
            self._handle(event.src_path, 'modified')

    def on_created(self, event):
        if not event.is_directory:
            self._handle(event.src_path, 'created')

    def on_deleted(self, event):
        if not event.is_directory:
            self._handle(event.src_path, 'deleted')

    def on_moved(self, event):
        if not event.is_directory:
            self._handle(event.src_path, 'deleted')
            self._handle(event.dest_path, 'created')


# --- ProjectWatcher ---

class ProjectWatcher:
    """Watches a single project directory and processes file changes in batches."""

    def __init__(self, project_root: str, db_path: str,
                 debounce_seconds: float = 2.0,
                 max_batch_size: int = 100,
                 git_settle_seconds: float = 3.0,
                 reconcile_interval: float = 6 * 3600):
        self.project_root = project_root
        self.db_path = db_path
        self.debounce_seconds = debounce_seconds
        self.max_batch_size = max_batch_size
        self.git_settle_seconds = git_settle_seconds
        self.reconcile_interval = reconcile_interval
        self.rules = IndexRules(project_root)

        self._observer: Optional[Observer] = None
        self._handler: Optional[FileChangeHandler] = None
        self._change_queue: asyncio.Queue = asyncio.Queue()
        self._drain_task: Optional[asyncio.Task] = None
        self._pending_changes: Dict[str, str] = {}  # path -> action
        self._debounce_handle: Optional[asyncio.TimerHandle] = None
        self._periodic_handle: Optional[asyncio.TimerHandle] = None
        self._processing = False
        self._stopped = False
        self._reconcile_job: Optional[jobs.Job] = None
        self.stats = {
            'files_indexed': 0,
            'files_deleted': 0,
            'batches_processed': 0,
            'reconciles': 0,
            'errors': 0,
        }

    @property
    def short(self) -> str:
        return Path(self.project_root).name

    async def start(self):
        loop = asyncio.get_running_loop()
        self._handler = FileChangeHandler(self.rules, self._change_queue, loop)

        self._observer = Observer()
        self._observer.schedule(self._handler, self.project_root, recursive=True)
        self._observer.daemon = True
        self._observer.start()

        self._drain_task = asyncio.create_task(self._drain_queue())

        # Backfill anything that changed while the server was down, and repair
        # any drift between Milvus and FTS. Runs as a background job.
        self.kick_reconcile("watcher start")
        self._schedule_periodic()

        logger.info("Started watching %s/ (debounce=%.1fs, reconcile every %.0fs)",
                    self.short, self.debounce_seconds, self.reconcile_interval)

    def kick_reconcile(self, reason: str) -> Optional[jobs.Job]:
        if self._stopped:
            return None
        job = jobs.start_reconcile_job(self.project_root, self.db_path, reason=reason)
        if job.kind == 'reconcile' and job is not self._reconcile_job:
            self.stats['reconciles'] += 1
            logger.info("%s/: reconcile started (%s, job %s)", self.short, reason, job.id)
        self._reconcile_job = job
        return job

    def _schedule_periodic(self):
        if self.reconcile_interval <= 0 or self._stopped:
            return
        loop = asyncio.get_running_loop()
        self._periodic_handle = loop.call_later(self.reconcile_interval, self._periodic_tick)

    def _periodic_tick(self):
        if self._stopped:
            return
        self.kick_reconcile("periodic")
        self._schedule_periodic()

    async def stop(self):
        self._stopped = True
        for handle in (self._debounce_handle, self._periodic_handle):
            if handle is not None:
                handle.cancel()
        self._debounce_handle = None
        self._periodic_handle = None

        if self._drain_task is not None:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass
            self._drain_task = None

        if self._reconcile_job is not None and self._reconcile_job.active:
            self._reconcile_job.cancel.set()

        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

        logger.info("Stopped watching %s/ (indexed=%d, deleted=%d, batches=%d, reconciles=%d)",
                    self.short, self.stats['files_indexed'], self.stats['files_deleted'],
                    self.stats['batches_processed'], self.stats['reconciles'])

    async def _drain_queue(self):
        """Continuously read events from the queue into the pending set."""
        try:
            while True:
                action, path = await self._change_queue.get()
                existing = self._pending_changes.get(path)
                if action in ('config', 'git'):
                    self._pending_changes[path] = action
                elif action == 'deleted':
                    self._pending_changes[path] = 'deleted'
                elif existing == 'deleted':
                    self._pending_changes[path] = 'modified'  # deleted then recreated
                else:
                    self._pending_changes[path] = action
                self._reset_debounce()
        except asyncio.CancelledError:
            pass

    def _reset_debounce(self):
        if self._debounce_handle is not None:
            self._debounce_handle.cancel()
        loop = asyncio.get_running_loop()
        self._debounce_handle = loop.call_later(
            self.debounce_seconds, lambda: asyncio.ensure_future(self._trigger_processing()))

    async def _trigger_processing(self):
        if self._stopped:
            return
        if self._is_git_active():
            logger.info("%s/: git operation in progress, deferring %.0fs", self.short, self.git_settle_seconds)
            loop = asyncio.get_running_loop()
            self._debounce_handle = loop.call_later(
                self.git_settle_seconds, lambda: asyncio.ensure_future(self._trigger_processing()))
            return
        if self._processing:
            self._reset_debounce()
            return
        await self._process_batch()

    def _is_git_active(self) -> bool:
        git_dir = Path(self.project_root) / '.git'
        return git_dir.is_dir() and (git_dir / 'index.lock').exists()

    async def _process_batch(self):
        if not self._pending_changes:
            return
        self._processing = True

        batch = dict(self._pending_changes)
        self._pending_changes.clear()

        try:
            config_changed = any(a == 'config' for a in batch.values())
            git_changed = any(a == 'git' for a in batch.values())
            batch = {p: a for p, a in batch.items() if a not in ('config', 'git')}

            if config_changed:
                self.rules.reload()
                rag_milvus.invalidate_ragconfig(self.project_root)
                logger.info("%s/: .ragignore/.ragconfig changed, reloading rules", self.short)
                self.kick_reconcile("config change")
            elif git_changed:
                # Checkout/rebase: FSEvents may have dropped events for some of the
                # files it touched, so a reconcile pass follows the per-file work.
                self.kick_reconcile("git ref change")

            if len(batch) > self.max_batch_size:
                items = list(batch.items())
                overflow = dict(items[self.max_batch_size:])
                batch = dict(items[:self.max_batch_size])
                self._pending_changes.update(overflow)
                logger.info("%s/: large batch, processing %d now, %d deferred",
                            self.short, len(batch), len(overflow))

            deletes = {p for p, a in batch.items() if a == 'deleted'}
            upserts = {p for p, a in batch.items() if a != 'deleted'}

            indexed = deleted = skipped = errors = 0

            for path in sorted(deletes):
                try:
                    if await rag_milvus.delete_by_path_async(path, self.db_path) > 0:
                        deleted += 1
                    else:
                        skipped += 1
                except Exception as e:
                    logger.warning("%s/: error deleting %s: %s", self.short, Path(path).name, e)
                    errors += 1

            for path in sorted(upserts):
                if not self.rules.should_index(path):   # vanished, grew too large, etc.
                    skipped += 1
                    continue
                try:
                    if await rag_milvus.add_file_async(path, force=False, db_path=self.db_path) > 0:
                        indexed += 1
                    else:
                        skipped += 1
                except Exception as e:
                    logger.warning("%s/: error indexing %s: %s", self.short, Path(path).name, e)
                    errors += 1

            self.stats['files_indexed'] += indexed
            self.stats['files_deleted'] += deleted
            self.stats['errors'] += errors
            self.stats['batches_processed'] += 1

            if indexed or deleted or errors:
                parts = [f"{n} {label}" for n, label in
                         ((indexed, "indexed"), (deleted, "deleted"), (skipped, "skipped"), (errors, "errors")) if n]
                logger.info("%s/: processed %d changes: %s", self.short, len(batch), ", ".join(parts))

        except Exception as e:
            logger.exception("%s/: batch processing error: %s", self.short, e)
            self.stats['errors'] += 1
        finally:
            self._processing = False
            if self._pending_changes:
                self._reset_debounce()


# --- Watcher Manager ---

_watchers: Dict[str, ProjectWatcher] = {}


async def ensure_watcher(project_root: str, db_path: str) -> Optional[ProjectWatcher]:
    """Ensure a watcher exists for the given project. Returns None if watching is disabled."""
    if not _watcher_config['enabled']:
        return None
    if project_root in _watchers:
        return _watchers[project_root]

    watcher = ProjectWatcher(
        project_root=project_root,
        db_path=db_path,
        debounce_seconds=_watcher_config['debounce_seconds'],
        max_batch_size=_watcher_config['max_batch_size'],
        git_settle_seconds=_watcher_config['git_settle_seconds'],
        reconcile_interval=_watcher_config['reconcile_interval'],
    )
    _watchers[project_root] = watcher
    await watcher.start()
    return watcher


async def stop_all_watchers():
    for watcher in list(_watchers.values()):
        await watcher.stop()
    _watchers.clear()


def get_watcher_status() -> Dict:
    status = {}
    for root, w in _watchers.items():
        job = jobs.active_job(root)
        status[root] = {
            'pending': len(w._pending_changes),
            'processing': w._processing,
            'stats': dict(w.stats),
            'active_job': job.to_dict() if job else None,
        }
    return status
