#!/usr/bin/env python3
"""
Persistent HTTP server for code-rag MCP system.

Runs as a long-lived process serving MCP via StreamableHTTP.
Projects are identified by the X-Project-Root header in each request.
The DB for each project lives at {project_root}/.code-rag/milvus.db.

Start: ./code-rag-server.sh
Health: curl http://127.0.0.1:7101/health
Index:  curl -X POST http://127.0.0.1:7101/index -d '{"path": "/abs/project"}'
Jobs:   curl http://127.0.0.1:7101/jobs
"""

import contextlib
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

from logging_setup import configure_logging
configure_logging()

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

import chunking
import file_watcher
import jobs
import model_config
import nl_descriptions
import rag_milvus
from tools import register_tools, set_current_project_root, resolve_under_root

logger = logging.getLogger("code-rag.http")

# --- Configuration ---

HOST = os.getenv("CODE_RAG_HOST", "127.0.0.1")
PORT = int(os.getenv("CODE_RAG_PORT", "7101"))

# Server runtime files live in ~/.code-rag/ (not in any project)
_SERVER_DIR = Path.home() / ".code-rag"
PID_FILE = _SERVER_DIR / "server.pid"

_started_at = time.time()
_model_loaded = False
_server_mode_ready = False
_last_error: dict | None = None


# --- Project middleware ---

class ProjectMiddleware:
    """ASGI middleware that extracts X-Project-Root header and sets ContextVar."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        global _last_error
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            project_root = headers.get(b"x-project-root", b"").decode("utf-8").strip()
            set_current_project_root(project_root or None)

        try:
            await self.app(scope, receive, send)
        except Exception as exc:
            _last_error = {"time": time.time(), "error": f"{type(exc).__name__}: {exc}"}
            logger.error("ASGI handler error: %s\n%s", exc, traceback.format_exc())
            if scope["type"] == "http":
                body = json.dumps({"error": "internal_server_error", "detail": str(exc)}).encode()
                try:
                    await send({"type": "http.response.start", "status": 500, "headers": [
                        [b"content-type", b"application/json"],
                        [b"content-length", str(len(body)).encode()],
                    ]})
                    await send({"type": "http.response.body", "body": body})
                except Exception:
                    pass  # response already started


# --- HTTP endpoints ---

def _gpu_memory_mb() -> float | None:
    try:
        import mlx.core as mx
        fn = getattr(mx, "get_active_memory", None)
        return round(fn() / 1e6, 1) if fn else None
    except Exception:
        return None


async def health(request: Request) -> JSONResponse:
    watchers = {}
    for root, s in file_watcher.get_watcher_status().items():
        watchers[Path(root).name] = {
            "pending": s["pending"],
            "processing": s["processing"],
            "files_indexed": s["stats"]["files_indexed"],
            "files_deleted": s["stats"]["files_deleted"],
            "batches": s["stats"]["batches_processed"],
            "reconciles": s["stats"].get("reconciles", 0),
            "errors": s["stats"]["errors"],
            "active_job": s.get("active_job"),
        }
    active_jobs = [j.to_dict() for j in jobs.list_jobs() if j.active]
    return JSONResponse({
        "status": "ok",
        "uptime_s": round(time.time() - _started_at, 1),
        "pid": os.getpid(),
        "model": _model_loaded,
        "embed_model": Path(rag_milvus._MODEL_PATH).name if _model_loaded else None,
        "embed_dim": rag_milvus._EMBED_DIM,
        "embed_max_tokens": rag_milvus.EMBED_MAX_TOKENS,
        "description_model": nl_descriptions.MODEL_ID if nl_descriptions.is_enabled() else None,
        "description_model_loaded": nl_descriptions.is_loaded(),
        "model_profile": model_config.summary()["profile"],
        "chunker": chunking.chunker_status(),
        "milvus": _server_mode_ready,
        "gpu_active_memory_mb": _gpu_memory_mb(),
        "watchers": watchers,
        "active_jobs": active_jobs,
        "last_error": _last_error,
    })


def _project_root_from(request: Request, body: dict) -> str | None:
    root = body.get("project_root") or request.headers.get("x-project-root", "").strip()
    return root or None


async def index_endpoint(request: Request) -> JSONResponse:
    """Start a background index job. Body: {path, project_root?, full?, clear?, extensions?}.
    Used by index.sh when the server is running, so the CLI never has to stop it."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    path = body.get("path")
    if not path:
        return JSONResponse({"error": "path is required"}, status_code=400)
    project_root = _project_root_from(request, body) or path
    try:
        path = resolve_under_root(path, project_root)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    if not os.path.isdir(path):
        return JSONResponse({"error": f"not a directory: {path}"}, status_code=400)
    db_path = str(Path(project_root) / ".code-rag" / "milvus.db")
    job = jobs.start_index_job(project_root, db_path, path=path, full=bool(body.get("full")),
                               extensions=body.get("extensions"), clear=bool(body.get("clear")))
    return JSONResponse(job.to_dict(), status_code=202)


async def jobs_list(request: Request) -> JSONResponse:
    root = request.query_params.get("project_root")
    return JSONResponse([j.to_dict() for j in jobs.list_jobs(root)])


async def job_detail(request: Request) -> JSONResponse:
    job = jobs.get_job(request.path_params["job_id"])
    if job is None:
        return JSONResponse({"error": "no such job"}, status_code=404)
    return JSONResponse(job.to_dict())


async def job_cancel(request: Request) -> JSONResponse:
    ok = jobs.cancel_job(request.path_params["job_id"])
    return JSONResponse({"cancelled": ok}, status_code=200 if ok else 404)


# --- Lifespan ---

@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    """Server lifecycle: PID file, model preload, server mode init."""
    import asyncio
    global _model_loaded, _server_mode_ready

    _SERVER_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))
    logger.info("PID %d written to %s", os.getpid(), PID_FILE)
    logger.info("Model config: %s", json.dumps(model_config.summary()))
    logger.info("Chunker: %s", json.dumps(chunking.chunker_status()))

    try:
        await asyncio.to_thread(rag_milvus.get_mlx_model)
        _model_loaded = True
    except Exception as e:
        logger.error("Could not pre-load embedding model: %s", e)

    try:
        rag_milvus.init_server_mode()
        _server_mode_ready = True
    except Exception as e:
        logger.error("Could not init server mode: %s", e)

    if nl_descriptions.is_enabled():
        nl_descriptions.start_idle_unloader()

    async with session_manager.run():
        logger.info("Server ready on http://%s:%d", HOST, PORT)
        try:
            yield
        finally:
            logger.info("Shutting down...")
            for job in jobs.list_jobs():
                if job.active:
                    job.cancel.set()
            await file_watcher.stop_all_watchers()

    rag_milvus.close_server_mode()
    try:
        import codechunk_wrapper
        codechunk_wrapper.shutdown()
    except Exception:
        pass
    if PID_FILE.exists() and PID_FILE.read_text().strip() == str(os.getpid()):
        PID_FILE.unlink()
    logger.info("Server stopped.")


# --- MCP server setup ---

mcp_server = Server("code-rag")
register_tools(mcp_server)

session_manager = StreamableHTTPSessionManager(
    app=mcp_server,
    stateless=True,
    json_response=True,
)


# --- Starlette app ---

app = Starlette(
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/index", index_endpoint, methods=["POST"]),
        Route("/jobs", jobs_list, methods=["GET"]),
        Route("/jobs/{job_id}", job_detail, methods=["GET"]),
        Route("/jobs/{job_id}/cancel", job_cancel, methods=["POST"]),
        Mount("/mcp", app=ProjectMiddleware(session_manager.handle_request)),
    ],
    lifespan=lifespan,
)


if __name__ == "__main__":
    # log_config=None: uvicorn's loggers propagate to our root handler instead of
    # installing their own untimestamped handlers.
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning", log_config=None,
                timeout_graceful_shutdown=30)
