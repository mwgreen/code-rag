"""Background job registry for long-running index operations.

index_directory and reconcile used to run synchronously inside MCP tool
handlers, which froze the event loop for minutes, failed the health checks, and
got the server killed by its own watchdog mid-index. They now run here, in a
daemon thread, and callers poll the job. One write job per project at a time.
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("code-rag.jobs")

MAX_KEPT = 50


@dataclass
class Job:
    id: str
    kind: str
    project_root: str
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    progress: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None
    cancel: threading.Event = field(default_factory=threading.Event)

    @property
    def status(self) -> str:
        if self.finished is not None:
            if self.error:
                return "failed"
            if isinstance(self.result, dict) and self.result.get("cancelled"):
                return "cancelled"
            return "done"
        return "running" if self.started is not None else "queued"

    @property
    def active(self) -> bool:
        return self.finished is None

    def to_dict(self) -> Dict[str, Any]:
        now = time.time()
        return {
            "id": self.id,
            "kind": self.kind,
            "project_root": self.project_root,
            "status": self.status,
            "elapsed_s": round((self.finished or now) - (self.started or self.created), 1),
            "progress": dict(self.progress),
            "result": self.result if self.finished is not None else None,
            "error": self.error,
        }


_jobs: Dict[str, Job] = {}
_lock = threading.Lock()


def get_job(job_id: str) -> Optional[Job]:
    return _jobs.get(job_id)


def active_job(project_root: str) -> Optional[Job]:
    with _lock:
        for job in _jobs.values():
            if job.project_root == project_root and job.active:
                return job
    return None


def list_jobs(project_root: Optional[str] = None) -> List[Job]:
    with _lock:
        jobs = [j for j in _jobs.values() if project_root is None or j.project_root == project_root]
    return sorted(jobs, key=lambda j: j.created, reverse=True)


def _prune() -> None:
    finished = sorted((j for j in _jobs.values() if not j.active), key=lambda j: j.finished or 0)
    for job in finished[:-MAX_KEPT] if len(finished) > MAX_KEPT else []:
        _jobs.pop(job.id, None)


def start_job(kind: str, project_root: str, target: Callable[[Job], Any]) -> Job:
    """Start `target(job)` in a daemon thread. Returns the existing active job for
    the project instead of starting a second writer."""
    with _lock:
        for job in _jobs.values():
            if job.project_root == project_root and job.active:
                return job
        job = Job(id=uuid.uuid4().hex[:8], kind=kind, project_root=project_root)
        _jobs[job.id] = job
        _prune()

    def run():
        job.started = time.time()
        try:
            job.result = target(job)
        except Exception as e:  # noqa: BLE001 - report every failure through the job
            job.error = f"{type(e).__name__}: {e}"
            logger.exception("Job %s (%s) failed", job.id, kind)
        finally:
            job.finished = time.time()
            logger.info("Job %s (%s) finished: %s", job.id, kind, job.status)

    threading.Thread(target=run, name=f"job-{kind}-{job.id}", daemon=True).start()
    return job


def start_index_job(project_root: str, db_path: str, path: Optional[str] = None,
                    full: bool = False, extensions: Optional[List[str]] = None,
                    clear: bool = False) -> Job:
    import rag_milvus  # local import: rag_milvus imports MLX, keep this module light

    def target(job: Job):
        if clear:
            rag_milvus.clear_collection(db_path=db_path)

        def progress(current: int, total: int, name: str = ""):
            job.progress.update(files_done=current, files_total=total, current=name)

        return rag_milvus.index_directory(
            path or project_root, extensions, incremental=not full,
            progress_callback=progress, db_path=db_path, cancel_event=job.cancel,
        )

    return start_job("index", project_root, target)


def start_reconcile_job(project_root: str, db_path: str, reason: str = "") -> Job:
    import rag_milvus

    def target(job: Job):
        job.progress["reason"] = reason

        def progress(current: int, total: int, name: str = ""):
            job.progress.update(files_done=current, files_total=total, current=name)

        return rag_milvus.reconcile(project_root, db_path, progress_callback=progress,
                                    cancel_event=job.cancel)

    return start_job("reconcile", project_root, target)


def cancel_job(job_id: str) -> bool:
    job = _jobs.get(job_id)
    if job is None or not job.active:
        return False
    job.cancel.set()
    return True
