"""Central logging configuration for code-rag.

Before this existed no handler was configured, so every logger.info() in the
codebase was silently dropped and the only output was bare print() calls with
no timestamps. Call configure_logging() once at process start (http_server,
mcp_server, index_codebase). Logs go to stderr; the launcher script redirects
stderr to ~/.code-rag/server.log and rotates it.
"""

import logging
import os
import sys

_configured = False

_NOISY = ("pymilvus", "urllib3", "httpx", "httpcore", "grpc", "asyncio",
          "watchdog", "uvicorn.access", "mcp", "sse_starlette", "milvus_lite")


def configure_logging(level: str | None = None, stream=None) -> None:
    global _configured
    if _configured:
        return
    # milvus-lite forks a gRPC server; without this, gRPC prints INFO chatter about
    # inherited file descriptors on every fork.
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    level_name = (level or os.getenv("CODE_RAG_LOG_LEVEL", "INFO")).upper()
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(getattr(logging, level_name, logging.INFO))
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
    # pymilvus warns on every query_iterator against milvus-lite ("failed to get mvccTs");
    # it is a known, harmless limitation of the embedded server.
    logging.getLogger("pymilvus").setLevel(logging.ERROR)
    _configured = True
