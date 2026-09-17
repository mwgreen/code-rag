#!/usr/bin/env python3
"""
MCP server for code-rag system (stdio transport).
For the persistent HTTP server, see http_server.py.
"""

import asyncio
import logging
import os
import sys

from logging_setup import configure_logging
configure_logging()

from mcp.server import Server
from mcp.server.stdio import stdio_server

import nl_descriptions
import rag_milvus
from tools import register_tools, set_current_project_root

logger = logging.getLogger("code-rag.stdio")

server = Server("code-rag")
register_tools(server)


async def main():
    project_root = os.getenv("CODE_RAG_PROJECT_ROOT", os.getcwd())
    set_current_project_root(project_root)
    logger.info("Project root: %s", project_root)
    logger.info("DB path: %s/.code-rag/milvus.db", project_root)

    # Pre-load only the MLX model (not the DB client, which would lock the file)
    try:
        await asyncio.to_thread(rag_milvus.get_mlx_model)
        logger.info("Model loaded; DB opens on first search.")
    except Exception as e:
        logger.warning("Could not pre-load model: %s. First search will be slow.", e)

    if nl_descriptions.is_enabled():
        nl_descriptions.start_idle_unloader()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
