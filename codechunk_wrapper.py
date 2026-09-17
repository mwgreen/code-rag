"""
Python wrapper for code-chunk (Node.js library with AST-aware, contextualized chunking).

One long-lived Node.js process serves every request over NDJSON (chunker_batch.mjs).
The previous design spawned `node chunker.mjs` per file, which cost 100-300 ms of
cold start for each of thousands of files, and resolved `node` from PATH at call
time, so a launch from a hook environment without Homebrew on PATH silently
degraded to regex chunking. Node is now resolved once at import and reported
through status() so /health can show which chunker is really active.
"""

import atexit
import json
import logging
import os
import select
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("code-rag.chunker")

_HERE = Path(__file__).parent
CHUNKER_SCRIPT = str(_HERE / "chunker.mjs")
CHUNKER_BATCH_SCRIPT = str(_HERE / "chunker_batch.mjs")
SUPPORTED_LANGUAGES = ['java', 'typescript', 'javascript', 'python', 'rust', 'go']

REQUEST_TIMEOUT_S = float(os.getenv("CODE_RAG_CHUNKER_TIMEOUT", "60"))


def _resolve_node() -> Optional[str]:
    candidates = [os.getenv("CODE_RAG_NODE", "")]
    found = shutil.which("node")
    if found:
        candidates.append(found)
    candidates += ["/opt/homebrew/bin/node", "/usr/local/bin/node", "/usr/bin/node"]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


NODE_BIN: Optional[str] = _resolve_node()
_CODECHUNK_INSTALLED = (_HERE / "node_modules" / "code-chunk").is_dir()

_unavailable_reason: Optional[str] = None
if NODE_BIN is None:
    _unavailable_reason = "node not found (set CODE_RAG_NODE or install Node.js)"
elif not _CODECHUNK_INSTALLED:
    _unavailable_reason = "node_modules/code-chunk missing (run npm install)"


def available() -> bool:
    return _unavailable_reason is None


def _parse_chunks(chunks_data: list) -> List[Dict]:
    """Convert code-chunk output to the internal chunk format."""
    chunks = []
    for c in chunks_data:
        chunk = {
            'content': c['contextualized'],
            'raw_content': c.get('content', ''),
            'start_line': c['start_line'],
            'end_line': c['end_line'],
            'node_type': c.get('node_type', 'chunk'),
        }
        if c.get('scope'):
            chunk['scope'] = c['scope']
        if c.get('signatures'):
            chunk['signatures'] = c['signatures']
        if c.get('imports'):
            chunk['imports'] = c['imports']
        chunks.append(chunk)
    return chunks


class _BatchChunker:
    """Thread-safe client for one long-lived chunker_batch.mjs process."""

    def __init__(self):
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._stderr = None
        self.failures = 0
        self.requests = 0

    def _start(self) -> None:
        self._stderr = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(
            [NODE_BIN, CHUNKER_BATCH_SCRIPT],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
            cwd=str(_HERE),
        )
        logger.info("Started code-chunk process (pid %s, %s)", self._proc.pid, NODE_BIN)

    def _stderr_tail(self) -> str:
        try:
            self._stderr.seek(0)
            return self._stderr.read()[-800:].decode('utf-8', errors='replace').strip()
        except Exception:
            return ""

    def _kill(self, why: str) -> None:
        if self._proc is not None:
            logger.warning("Restarting code-chunk process: %s. stderr: %s", why, self._stderr_tail())
            try:
                self._proc.kill()
                self._proc.wait(timeout=5)
            except Exception:
                pass
        self._proc = None

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def close(self) -> None:
        with self._lock:
            if self._proc is not None:
                try:
                    self._proc.stdin.close()
                    self._proc.wait(timeout=5)
                except Exception:
                    self._proc.kill()
                self._proc = None

    def chunk(self, content: str, filepath: str, max_size: int) -> Optional[List[Dict]]:
        with self._lock:
            if not self.alive():
                self._start()
            self.requests += 1
            request = json.dumps({'filepath': filepath, 'content': content, 'max_size': max_size})
            try:
                self._proc.stdin.write((request + '\n').encode('utf-8'))
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                self.failures += 1
                self._kill(f"write failed: {e}")
                return None

            ready, _, _ = select.select([self._proc.stdout], [], [], REQUEST_TIMEOUT_S)
            if not ready:
                self.failures += 1
                self._kill(f"timeout after {REQUEST_TIMEOUT_S}s on {filepath}")
                return None
            line = self._proc.stdout.readline()
            if not line:
                self.failures += 1
                self._kill(f"process exited (code {self._proc.poll()}) on {filepath}")
                return None

        try:
            result = json.loads(line)
        except json.JSONDecodeError as e:
            logger.warning("code-chunk returned invalid JSON for %s: %s", filepath, e)
            return None
        if 'error' in result:
            if 'Unsupported' not in result['error']:
                logger.info("code-chunk could not chunk %s: %s", filepath, result['error'][:120])
            return None
        chunks = _parse_chunks(result.get('chunks', []))
        return chunks or None


_shared = _BatchChunker()
atexit.register(lambda: _shared.close())


def chunk_with_codechunk(content: str, filepath: str, language: str,
                         max_size: Optional[int] = None) -> Optional[List[Dict]]:
    """Chunk one file through the shared long-lived Node.js process. None on any failure."""
    if not available() or language not in SUPPORTED_LANGUAGES:
        return None
    if max_size is None:
        max_size = int(os.getenv("MAX_CHUNK_SIZE", "3000"))
    try:
        return _shared.chunk(content, filepath, max_size)
    except Exception as e:
        logger.warning("code-chunk failed for %s: %s", filepath, e)
        return None


def shutdown() -> None:
    _shared.close()


def status() -> Dict:
    """For /health: which chunker is really in use."""
    return {
        "available": available(),
        "node": NODE_BIN,
        "reason": _unavailable_reason,
        "process_alive": _shared.alive(),
        "requests": _shared.requests,
        "failures": _shared.failures,
    }


# Backwards-compatible alias: the old per-call context manager API.
class CodeChunkBatch:
    """Kept for compatibility; the shared process is used regardless."""

    def __init__(self, max_size: Optional[int] = None):
        self.max_size = max_size or int(os.getenv("MAX_CHUNK_SIZE", "3000"))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def chunk(self, content: str, filepath: str, language: str) -> Optional[List[Dict]]:
        return chunk_with_codechunk(content, filepath, language, self.max_size)
