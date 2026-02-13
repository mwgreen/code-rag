"""
File watcher for automatic incremental reindexing.

Uses macOS FSEvents (via watchdog) to detect file changes and reindex
affected files in batches. Designed to handle burst scenarios like
git checkout without degrading search performance.

Pipeline: FSEvents -> watchdog thread -> filter -> asyncio.Queue -> debounce -> batch process
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import rag_milvus

# --- Configuration ---

_watcher_config = {
    'enabled': os.getenv('CODE_RAG_WATCH', 'true').lower() in ('true', '1', 'yes'),
    'debounce_seconds': float(os.getenv('CODE_RAG_WATCH_DEBOUNCE', '2.0')),
    'max_batch_size': int(os.getenv('CODE_RAG_WATCH_MAX_BATCH', '100')),
    'git_settle_seconds': float(os.getenv('CODE_RAG_WATCH_GIT_SETTLE', '3.0')),
}

# File extensions to watch (matches index_directory defaults)
_WATCH_EXTENSIONS = {
    '.java', '.js', '.ts', '.tsx', '.jsx', '.json',
    '.xml', '.yaml', '.yml', '.md', '.gradle', '.properties'
}


def _log(msg: str):
    print(f"[watcher] {msg}", file=sys.stderr)


# --- FileChangeHandler ---

class FileChangeHandler(FileSystemEventHandler):
    """Watchdog event handler that filters and forwards changes to asyncio."""

    def __init__(self, project_root: str, change_queue: asyncio.Queue,
                 loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.project_root = project_root
        self.change_queue = change_queue
        self.loop = loop
        self._excluded_dirs: Optional[set] = None

    def _should_handle(self, path: str) -> bool:
        """Fast pre-filter. Runs in watchdog thread — must be cheap."""
        p = Path(path)
        # Extension check
        if p.suffix not in _WATCH_EXTENSIONS:
            return False
        # Dotfiles
        if p.name.startswith('.'):
            return False
        # TypeScript declaration files
        if p.name.endswith('.d.ts'):
            return False
        # Lazy-load excluded dirs (once per handler)
        if self._excluded_dirs is None:
            self._excluded_dirs = rag_milvus.get_excluded_dirs(
                project_root=self.project_root)
        # Excluded directories
        if self._excluded_dirs & set(p.parts):
            return False
        # Skip large files (>1MB)
        try:
            if p.exists() and p.stat().st_size > 1024 * 1024:
                return False
        except OSError:
            pass
        return True

    def _enqueue(self, action: str, path: str):
        """Thread-safe enqueue to asyncio loop."""
        try:
            self.loop.call_soon_threadsafe(
                self.change_queue.put_nowait,
                (action, path)
            )
        except RuntimeError:
            pass  # Loop closed during shutdown

    def on_modified(self, event):
        if not event.is_directory and self._should_handle(event.src_path):
            self._enqueue('modified', event.src_path)

    def on_created(self, event):
        if not event.is_directory and self._should_handle(event.src_path):
            self._enqueue('created', event.src_path)

    def on_deleted(self, event):
        if not event.is_directory and self._should_handle(event.src_path):
            self._enqueue('deleted', event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            # Decompose move into delete + create
            if self._should_handle(event.src_path):
                self._enqueue('deleted', event.src_path)
            if self._should_handle(event.dest_path):
                self._enqueue('created', event.dest_path)


# --- ProjectWatcher ---

class ProjectWatcher:
    """Watches a single project directory and processes file changes in batches."""

    def __init__(self, project_root: str, db_path: str,
                 debounce_seconds: float = 2.0,
                 max_batch_size: int = 100,
                 git_settle_seconds: float = 3.0):
        self.project_root = project_root
        self.db_path = db_path
        self.debounce_seconds = debounce_seconds
        self.max_batch_size = max_batch_size
        self.git_settle_seconds = git_settle_seconds

        self._observer: Optional[Observer] = None
        self._change_queue: asyncio.Queue = asyncio.Queue()
        self._drain_task: Optional[asyncio.Task] = None
        self._pending_changes: Dict[str, str] = {}  # path -> action
        self._debounce_handle: Optional[asyncio.TimerHandle] = None
        self._processing = False
        self._stopped = False
        self.stats = {
            'files_indexed': 0,
            'files_deleted': 0,
            'batches_processed': 0,
            'errors': 0,
        }

    async def start(self):
        """Start watching the project directory."""
        loop = asyncio.get_event_loop()
        handler = FileChangeHandler(self.project_root, self._change_queue, loop)

        self._observer = Observer()
        self._observer.schedule(handler, self.project_root, recursive=True)
        self._observer.daemon = True
        self._observer.start()

        self._drain_task = asyncio.create_task(self._drain_queue())

        short = Path(self.project_root).name
        _log(f"Started watching {short}/ (debounce={self.debounce_seconds}s)")

    async def stop(self):
        """Stop the watcher and cancel pending work."""
        self._stopped = True

        if self._debounce_handle is not None:
            self._debounce_handle.cancel()
            self._debounce_handle = None

        if self._drain_task is not None:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass
            self._drain_task = None

        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

        short = Path(self.project_root).name
        _log(f"Stopped watching {short}/ "
             f"(indexed={self.stats['files_indexed']}, "
             f"deleted={self.stats['files_deleted']}, "
             f"batches={self.stats['batches_processed']})")

    async def _drain_queue(self):
        """Continuously read events from the queue into the pending set."""
        try:
            while True:
                action, path = await self._change_queue.get()

                # Merge logic: delete always wins over prior actions
                existing = self._pending_changes.get(path)
                if action == 'deleted':
                    self._pending_changes[path] = 'deleted'
                elif existing == 'deleted':
                    # Was deleted, now recreated -> modified
                    self._pending_changes[path] = 'modified'
                else:
                    self._pending_changes[path] = action

                self._reset_debounce()
        except asyncio.CancelledError:
            pass

    def _reset_debounce(self):
        """Reset the debounce timer. Called each time a new event arrives."""
        if self._debounce_handle is not None:
            self._debounce_handle.cancel()

        loop = asyncio.get_event_loop()
        self._debounce_handle = loop.call_later(
            self.debounce_seconds,
            lambda: asyncio.ensure_future(self._trigger_processing())
        )

    async def _trigger_processing(self):
        """Called when debounce timer fires."""
        if self._stopped:
            return

        # Check if git is mid-operation
        if self._is_git_active():
            short = Path(self.project_root).name
            _log(f"{short}/: Git operation in progress, deferring {self.git_settle_seconds}s...")
            loop = asyncio.get_event_loop()
            self._debounce_handle = loop.call_later(
                self.git_settle_seconds,
                lambda: asyncio.ensure_future(self._trigger_processing())
            )
            return

        # Don't start a new batch while one is processing
        if self._processing:
            self._reset_debounce()
            return

        await self._process_batch()

    def _is_git_active(self) -> bool:
        """Check if git is mid-operation (.git/index.lock exists)."""
        git_dir = Path(self.project_root) / '.git'
        if not git_dir.is_dir():
            return False
        return (git_dir / 'index.lock').exists()

    async def _process_batch(self):
        """Process accumulated changes."""
        if not self._pending_changes:
            return

        self._processing = True
        short = Path(self.project_root).name

        # Snapshot and clear — new events accumulate in a fresh dict
        batch = dict(self._pending_changes)
        self._pending_changes.clear()

        # Cap batch size, re-queue overflow
        if len(batch) > self.max_batch_size:
            items = list(batch.items())
            overflow = dict(items[self.max_batch_size:])
            batch = dict(items[:self.max_batch_size])
            self._pending_changes.update(overflow)
            _log(f"{short}/: Large batch, processing {len(batch)} now, "
                 f"{len(overflow)} deferred")

        deletes = {p for p, a in batch.items() if a == 'deleted'}
        upserts = {p for p, a in batch.items() if a != 'deleted'}

        indexed = 0
        deleted = 0
        skipped = 0
        errors = 0

        try:
            loop = asyncio.get_event_loop()

            # Phase 1: Handle deletes (fast, no embedding)
            for path in deletes:
                try:
                    count = await rag_milvus.delete_by_path_async(path, self.db_path)
                    if count > 0:
                        deleted += 1
                    else:
                        skipped += 1
                except Exception as e:
                    _log(f"Error deleting {Path(path).name}: {e}")
                    errors += 1

            # Phase 2: Handle upserts (embed + write, one at a time)
            for path in upserts:
                # File may have disappeared between event and processing
                if not Path(path).exists():
                    skipped += 1
                    continue
                try:
                    if Path(path).stat().st_size > 1024 * 1024:
                        skipped += 1
                        continue
                except OSError:
                    skipped += 1
                    continue

                try:
                    chunks = await rag_milvus.add_file_async(
                        path, force=False, db_path=self.db_path)
                    if chunks > 0:
                        indexed += 1
                    else:
                        skipped += 1
                except Exception as e:
                    _log(f"Error indexing {Path(path).name}: {e}")
                    errors += 1

                # Yield control so search requests can proceed
                await asyncio.sleep(0)

            # MLX cache cleanup
            if indexed > 0 and indexed % 20 == 0:
                try:
                    import mlx.core as mx
                    mx.clear_cache()
                except Exception:
                    pass

            self.stats['files_indexed'] += indexed
            self.stats['files_deleted'] += deleted
            self.stats['errors'] += errors
            self.stats['batches_processed'] += 1

            total = indexed + deleted
            if total > 0:
                parts = []
                if indexed:
                    parts.append(f"{indexed} indexed")
                if deleted:
                    parts.append(f"{deleted} deleted")
                if skipped:
                    parts.append(f"{skipped} skipped")
                if errors:
                    parts.append(f"{errors} errors")
                _log(f"{short}/: Processed {len(batch)} changes: {', '.join(parts)}")

        except Exception as e:
            _log(f"{short}/: Batch processing error: {e}")
            self.stats['errors'] += 1
        finally:
            self._processing = False
            # If more changes accumulated during processing, trigger again
            if self._pending_changes:
                self._reset_debounce()


# --- Watcher Manager ---

_watchers: Dict[str, ProjectWatcher] = {}


async def ensure_watcher(project_root: str, db_path: str) -> Optional[ProjectWatcher]:
    """Ensure a watcher exists for the given project. Creates one if needed.
    Returns None if watching is disabled."""
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
    )
    await watcher.start()
    _watchers[project_root] = watcher
    return watcher


async def stop_all_watchers():
    """Stop all active watchers. Called during server shutdown."""
    for watcher in list(_watchers.values()):
        await watcher.stop()
    _watchers.clear()


def get_watcher_status() -> Dict:
    """Return status of all active watchers."""
    return {
        root: {
            'pending': len(w._pending_changes),
            'processing': w._processing,
            'stats': dict(w.stats),
        }
        for root, w in _watchers.items()
    }
