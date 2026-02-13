#!/usr/bin/env python3
"""
Persistent HTTP server for code-rag MCP system.

Runs as a long-lived process serving MCP via StreamableHTTP.
Projects are identified by the X-Project-Root header in each request.
The DB for each project lives at {project_root}/.code-rag/milvus.db.

Start: ./code-rag-server.sh
Health: curl http://127.0.0.1:7101/health
"""

import contextlib
import os
import signal
import sys
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

import rag_milvus
from tools import register_tools, set_current_project_root

# --- Configuration ---

HOST = os.getenv("CODE_RAG_HOST", "127.0.0.1")
PORT = int(os.getenv("CODE_RAG_PORT", "7101"))

# Server runtime files live in ~/.code-rag/ (not in any project)
_SERVER_DIR = Path.home() / ".code-rag"
PID_FILE = _SERVER_DIR / "server.pid"
LOG_FILE = _SERVER_DIR / "server.log"


# --- Project middleware ---

class ProjectMiddleware:
    """ASGI middleware that extracts X-Project-Root header and sets ContextVar."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            project_root = headers.get(b"x-project-root", b"").decode("utf-8").strip()

            if project_root:
                set_current_project_root(project_root)
            else:
                set_current_project_root(None)

        await self.app(scope, receive, send)


# --- Health endpoint ---

async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


# --- Lifespan ---

@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    """Server lifecycle: PID file, model preload, server mode init."""
    _SERVER_DIR.mkdir(parents=True, exist_ok=True)

    # Write PID file
    PID_FILE.write_text(str(os.getpid()))
    print(f"[HTTP] PID {os.getpid()} written to {PID_FILE}", file=sys.stderr)

    # Pre-load MLX model
    try:
        print("[HTTP] Pre-loading MLX model...", file=sys.stderr)
        rag_milvus.get_mlx_model()
        print("[HTTP] MLX model loaded.", file=sys.stderr)
    except Exception as e:
        print(f"[HTTP] Warning: Could not pre-load model: {e}", file=sys.stderr)

    # Init server mode (lazy persistent clients per project)
    rag_milvus.init_server_mode()

    # Start session manager
    async with session_manager.run():
        print(f"[HTTP] Server ready on http://{HOST}:{PORT}", file=sys.stderr)
        try:
            yield
        finally:
            pass

    # Cleanup
    rag_milvus.close_server_mode()
    if PID_FILE.exists():
        PID_FILE.unlink()
    print("[HTTP] Server stopped.", file=sys.stderr)


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
        Mount("/mcp", app=ProjectMiddleware(session_manager.handle_request)),
    ],
    lifespan=lifespan,
)


# --- Signal handling ---

def _handle_signal(signum, frame):
    """Graceful shutdown on SIGTERM/SIGINT."""
    print(f"[HTTP] Received signal {signum}, shutting down...", file=sys.stderr)
    raise SystemExit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _handle_signal)

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="warning",
    )
