"""
Shared tool definitions and project context for code-rag MCP servers.
Both stdio (mcp_server.py) and HTTP (http_server.py) import from here.
"""

import contextvars
from pathlib import Path
from mcp.server import Server
from mcp import types

import rag_milvus
import file_watcher


# --- Project context ---

_current_project_root: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_project_root", default=None
)


class ProjectNotConfiguredError(Exception):
    pass


_PROJECT_ERROR_MSG = (
    "No project configured. Add to your .mcp.json:\n"
    '  "headers": {"X-Project-Root": "/path/to/your/project"}'
)


def set_current_project_root(root: str | None):
    """Set the project root for the current context (called by middleware or stdio init)."""
    _current_project_root.set(root)


def get_current_project_root() -> str:
    """Get the project root for the current context. Raises if not set."""
    root = _current_project_root.get()
    if root is None:
        raise ProjectNotConfiguredError(_PROJECT_ERROR_MSG)
    return root


def get_db_path() -> str:
    """Derive the Milvus DB path from the current project root."""
    root = get_current_project_root()
    return str(Path(root) / ".code-rag" / "milvus.db")


# --- Formatting helpers ---

def format_results(results: list[dict]) -> str:
    """Format search results as markdown."""
    if not results:
        return "No results found."

    output = []
    for r in results:
        output.append(f"### {r['path']}:{r.get('start_line', '?')}-{r.get('end_line', '?')}")

        meta_parts = [f"**Language:** {r['language']}", f"**Type:** {r['type']}"]
        if r.get('class_name'):
            meta_parts.append(f"**Class:** {r['class_name']}")
        if r.get('component'):
            meta_parts.append(f"**Component:** {r['component']}")
        if r.get('distance') is not None:
            meta_parts.append(f"**Relevance:** {1 - r['distance']:.2f}")

        output.append(" | ".join(meta_parts))
        output.append(f"```{r['language']}\n{r['content']}\n```")
        output.append("")

    return "\n".join(output)


def format_results_grouped(results: list[dict]) -> str:
    """Format search results grouped by type."""
    if not results:
        return "No results found."

    grouped = {}
    for r in results:
        t = r['type']
        if t not in grouped:
            grouped[t] = []
        grouped[t].append(r)

    output = []
    for type_name in ['documentation', 'code', 'config']:
        if type_name in grouped:
            output.append(f"## {type_name.title()} Results\n")
            output.append(format_results(grouped[type_name]))

    return "\n".join(output)


def format_file_list(files: dict[str, list[str]]) -> str:
    """Format indexed files list."""
    if not files:
        return "No files indexed."

    output = []
    for file_type, paths in sorted(files.items()):
        output.append(f"## {file_type.title()} ({len(paths)} files)\n")
        for path in sorted(paths)[:50]:
            output.append(f"- {path}")
        if len(paths) > 50:
            output.append(f"\n...and {len(paths) - 50} more")
        output.append("")

    return "\n".join(output)


def format_stats(stats: dict, db_path: str) -> str:
    """Format collection statistics."""
    lines = [
        f"**Total Files:** {stats['total_files']}",
        f"**Total Chunks:** {stats['total_chunks']}",
        "",
        "### By Language"
    ]

    for lang, count in sorted(stats['by_language'].items(), key=lambda x: x[1], reverse=True):
        lines.append(f"- {lang}: {count} chunks")

    lines.append("\n### By Type")
    for t, count in sorted(stats['by_type'].items(), key=lambda x: x[1], reverse=True):
        lines.append(f"- {t}: {count} chunks")

    lines.append(f"\n**Index Location:** {db_path}")

    if 'note' in stats:
        lines.append(f"\n*Note: {stats['note']}*")

    return "\n".join(lines)


# --- Tool registration ---

def register_tools(server: Server):
    """Register all code-rag tools on the given MCP server."""

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="search_code",
                description="Search the indexed codebase for relevant code snippets",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query (e.g., 'user authentication logic')"
                        },
                        "n": {
                            "type": "integer",
                            "description": "Number of results to return (default: 5)",
                            "default": 5
                        },
                        "language": {
                            "type": "string",
                            "description": "Filter by language: java, javascript, typescript, etc.",
                            "enum": ["java", "javascript", "typescript", "yaml", "json", "xml", "markdown", "properties", "gradle"]
                        }
                    },
                    "required": ["query"]
                }
            ),
            types.Tool(
                name="search_docs",
                description="Search the YAML documentation for information about components and architecture",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query about the system (e.g., 'how does UserGrid work?')"
                        },
                        "n": {
                            "type": "integer",
                            "description": "Number of results to return (default: 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            ),
            types.Tool(
                name="search_all",
                description="Search everything - code, documentation, and config files",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query"
                        },
                        "n": {
                            "type": "integer",
                            "description": "Number of results to return (default: 10)",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            ),
            types.Tool(
                name="index_file",
                description="Index or re-index a single file",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to file"
                        }
                    },
                    "required": ["path"]
                }
            ),
            types.Tool(
                name="index_directory",
                description="Index all supported files in a directory",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to directory"
                        }
                    },
                    "required": ["path"]
                }
            ),
            types.Tool(
                name="list_indexed",
                description="List all indexed files grouped by type",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            ),
            types.Tool(
                name="get_stats",
                description="Get index statistics (file count, chunk count by language/type)",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            ),
            types.Tool(
                name="watcher_status",
                description="Get file watcher status (pending changes, indexing stats)",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        try:
            db = get_db_path()
            project_root = get_current_project_root()
        except ProjectNotConfiguredError as e:
            return [types.TextContent(type="text", text=str(e))]

        # Ensure file watcher is running for this project (best-effort)
        try:
            await file_watcher.ensure_watcher(project_root, db)
        except Exception:
            pass

        try:
            if name == "search_code":
                results = rag_milvus.search(
                    arguments["query"], arguments.get("n", 5),
                    type_filter="code", language_filter=arguments.get("language"),
                    db_path=db
                )
                return [types.TextContent(type="text", text=format_results(results))]

            elif name == "search_docs":
                results = rag_milvus.search(
                    arguments["query"], arguments.get("n", 5),
                    type_filter="documentation", db_path=db
                )
                return [types.TextContent(type="text", text=format_results(results))]

            elif name == "search_all":
                results = rag_milvus.search(
                    arguments["query"], arguments.get("n", 10),
                    db_path=db
                )
                return [types.TextContent(type="text", text=format_results_grouped(results))]

            elif name == "index_file":
                count = rag_milvus.add_file(arguments["path"], force=True, db_path=db)
                return [types.TextContent(type="text", text=f"Indexed {arguments['path']}\n\n**Chunks created:** {count}")]

            elif name == "index_directory":
                stats = rag_milvus.index_directory(arguments["path"], db_path=db)
                output = f"Indexed {arguments['path']}\n\n"
                output += f"**Files indexed:** {stats['files_indexed']}\n"
                output += f"**Chunks created:** {stats['chunks_created']}\n\n"
                if stats['by_language']:
                    output += "### By Language\n"
                    for lang, count in sorted(stats['by_language'].items(), key=lambda x: x[1], reverse=True):
                        output += f"- {lang}: {count} chunks\n"
                return [types.TextContent(type="text", text=output)]

            elif name == "list_indexed":
                files = rag_milvus.list_indexed_files(db_path=db)
                return [types.TextContent(type="text", text=format_file_list(files))]

            elif name == "get_stats":
                stats = rag_milvus.get_stats(db_path=db)
                return [types.TextContent(type="text", text=format_stats(stats, db))]

            elif name == "watcher_status":
                status = file_watcher.get_watcher_status()
                project_status = status.get(project_root)
                if project_status is None:
                    return [types.TextContent(type="text", text="No active file watcher for this project.")]
                s = project_status['stats']
                lines = [
                    f"**File Watcher:** active",
                    f"**Pending changes:** {project_status['pending']}",
                    f"**Currently processing:** {project_status['processing']}",
                    "",
                    "### Cumulative Stats",
                    f"- Files indexed: {s['files_indexed']}",
                    f"- Files deleted: {s['files_deleted']}",
                    f"- Batches processed: {s['batches_processed']}",
                    f"- Errors: {s['errors']}",
                ]
                return [types.TextContent(type="text", text="\n".join(lines))]

            else:
                raise ValueError(f"Unknown tool: {name}")

        except Exception as e:
            return [types.TextContent(type="text", text=f"Error executing {name}: {str(e)}")]
