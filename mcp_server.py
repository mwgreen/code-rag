#!/usr/bin/env python3
"""
MCP server for code-rag system (stdio transport).
For persistent HTTP server, see http_server.py.
"""

import asyncio
import os
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server

import rag_milvus
from tools import register_tools, set_current_project_root


# Create MCP server
server = Server("code-rag")

# Register all tools
register_tools(server)


async def main():
    """Run the MCP server via stdio."""
    # stdio mode: project root from env var or current working directory
    project_root = os.getenv("CODE_RAG_PROJECT_ROOT", os.getcwd())
    set_current_project_root(project_root)
    print(f"[MCP] Project root: {project_root}", file=sys.stderr)
    print(f"[MCP] DB path: {project_root}/.code-rag/milvus.db", file=sys.stderr)

    # Pre-load only the MLX model (not the DB client - that would lock the file!)
    try:
        print("[MCP] Pre-loading MLX model...", file=sys.stderr)
        rag_milvus.get_mlx_model()
        print(f"[MCP] Ready! Model loaded. DB will open on first search.", file=sys.stderr)
    except Exception as e:
        print(f"[MCP] Warning: Could not pre-load model: {e}", file=sys.stderr)
        print("[MCP] Server will still start, but first search may be slow.", file=sys.stderr)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
