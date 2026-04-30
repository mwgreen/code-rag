"""
Shared tool definitions and project context for code-rag MCP servers.
Both stdio (mcp_server.py) and HTTP (http_server.py) import from here.
"""

import asyncio
import contextvars
import fnmatch
import os
from pathlib import Path
from typing import Optional
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


# --- Relevance filtering ---

def _apply_relevance_floor(results: list[dict], min_relevance: float) -> tuple[list[dict], bool]:
    """Filter results below min_relevance. Returns (filtered_results, all_below_floor)."""
    if min_relevance <= 0 or not results:
        return results, False
    filtered = [r for r in results
                if r.get('distance') is None or (1 - r['distance']) >= min_relevance]
    if not filtered:
        return results, True  # Return all with warning
    return filtered, False


def _format_search_results(results: list[dict], min_relevance: float, grouped: bool = False) -> str:
    """Apply relevance floor, format results, and append read_file hint."""
    results, below_floor = _apply_relevance_floor(results, min_relevance)
    text = format_results_grouped(results) if grouped else format_results(results)
    if below_floor:
        text = ("*Warning: All results are below the configured relevance threshold "
                f"({min_relevance:.2f}). Showing best matches anyway.*\n\n" + text)
    if results:
        text += "\n*To read the full file, use the read_file tool with the path shown above.*"
        # Multi-chunk hint: when a single file shows up in 2+ result chunks, the
        # caller probably wants the whole doc, not just disconnected slices.
        # Cascade / list / "walk me through" questions are the canonical case
        # where partial reads produce confidently-incomplete answers.
        path_counts: dict[str, int] = {}
        for r in results:
            p = r.get('path')
            if p:
                path_counts[p] = path_counts.get(p, 0) + 1
        multi_hit = sorted(
            [(p, n) for p, n in path_counts.items() if n >= 2],
            key=lambda x: -x[1],
        )
        if multi_hit:
            top = "\n".join(f"  - {p} ({n} chunks)" for p, n in multi_hit[:3])
            text += (
                "\n\n*Multiple chunks of the same file matched. For cascade / "
                '"walk me through" / list questions, read the whole file before '
                "answering — partial chunks lead to confidently incomplete "
                f"answers:*\n{top}"
            )
    return text


# --- Formatting helpers ---

def format_results(results: list[dict]) -> str:
    """Format search results as markdown."""
    if not results:
        return "No results found."

    output = []

    # Low-confidence warning when best vector result has poor similarity
    vector_distances = [r['distance'] for r in results
                        if r.get('distance') is not None and r['distance'] > 0.01]
    if vector_distances and min(vector_distances) > 0.5:
        output.append(
            "*Note: Low semantic similarity — results may not match your query well. "
            "Consider using Grep for exact keyword matching.*\n"
        )

    for r in results:
        header = f"### {r['path']}:{r.get('start_line', '?')}-{r.get('end_line', '?')}"
        output.append(header)

        meta_parts = [f"**Language:** {r['language']}", f"**Type:** {r['type']}"]
        if r.get('class_name'):
            meta_parts.append(f"**Class:** {r['class_name']}")
        if r.get('component'):
            meta_parts.append(f"**Component:** {r['component']}")
        if r.get('distance') is not None:
            meta_parts.append(f"**Relevance:** {1 - r['distance']:.2f}")

        output.append(" | ".join(meta_parts))
        if r.get('description'):
            output.append(f"**Summary:** {r['description']}")
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


_FILE_LIST_CAP_PER_SECTION = 1000


def format_file_list(files: dict[str, list[str]], path_glob: Optional[str] = None) -> str:
    """Format indexed files list. Optional path_glob filters paths via fnmatch."""
    import fnmatch as _fn
    if not files:
        return "No files indexed."

    if path_glob:
        files = {
            t: [p for p in paths if _fn.fnmatch(p, path_glob)]
            for t, paths in files.items()
        }
        files = {t: paths for t, paths in files.items() if paths}
        if not files:
            return f"No indexed files match glob: {path_glob}"

    output = []
    for file_type, paths in sorted(files.items()):
        suffix = f" (filtered by {path_glob})" if path_glob else ""
        output.append(f"## {file_type.title()} ({len(paths)} files{suffix})\n")
        for path in sorted(paths)[:_FILE_LIST_CAP_PER_SECTION]:
            output.append(f"- {path}")
        if len(paths) > _FILE_LIST_CAP_PER_SECTION:
            output.append(
                f"\n...and {len(paths) - _FILE_LIST_CAP_PER_SECTION} more "
                f"(narrow with the path_glob argument, e.g. \"docs/**/*.md\")"
            )
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
                description="List indexed files grouped by type. Up to 1000 per type; pass `path_glob` to narrow.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path_glob": {
                            "type": "string",
                            "description": "Optional fnmatch glob to filter paths, e.g. 'docs/**/*.md'"
                        }
                    },
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
            types.Tool(
                name="read_file",
                description="Read the full content of a file in the project. Use after searching to see complete file context.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to the file to read"
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "1-based start line (optional, for reading a range)"
                        },
                        "end_line": {
                            "type": "integer",
                            "description": "1-based end line, inclusive (optional)"
                        }
                    },
                    "required": ["path"]
                }
            ),
            types.Tool(
                name="delete_by_pattern",
                description="Delete indexed entries matching a glob pattern. Use dry_run=true (default) to preview.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern matched against indexed file paths (e.g. '**/*.yaml', '.playwright-mcp/*')"
                        },
                        "dry_run": {
                            "type": "boolean",
                            "description": "If true (default), show what would be deleted without deleting",
                            "default": True
                        }
                    },
                    "required": ["pattern"]
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

        # Lazy stale entry cleanup (once per project per server lifetime)
        try:
            rag_milvus.cleanup_stale_entries(db)
        except Exception:
            pass

        # Load project config
        ragconfig = rag_milvus.load_ragconfig(project_root)
        min_relevance = ragconfig.get('min_relevance', 0.0)

        try:
            if name == "search_code":
                results = await rag_milvus.search_async(
                    arguments["query"], arguments.get("n", 5),
                    type_filter="code", language_filter=arguments.get("language"),
                    db_path=db
                )
                return [types.TextContent(type="text",
                    text=_format_search_results(results, min_relevance))]

            elif name == "search_docs":
                results = await rag_milvus.search_async(
                    arguments["query"], arguments.get("n", 5),
                    type_filter="documentation", db_path=db
                )
                return [types.TextContent(type="text",
                    text=_format_search_results(results, min_relevance))]

            elif name == "search_all":
                results = await rag_milvus.search_async(
                    arguments["query"], arguments.get("n", 10),
                    db_path=db
                )
                return [types.TextContent(type="text",
                    text=_format_search_results(results, min_relevance, grouped=True))]

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
                return [types.TextContent(type="text",
                    text=format_file_list(files, path_glob=arguments.get("path_glob")))]

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

            elif name == "read_file":
                file_path = arguments["path"]
                abs_path = str(Path(file_path).resolve())

                # Security: must be under project root
                real_root = os.path.realpath(project_root)
                if not abs_path.startswith(real_root + os.sep) and abs_path != real_root:
                    return [types.TextContent(type="text",
                        text=f"Error: Path must be within the project root ({project_root})")]

                if not os.path.isfile(abs_path):
                    return [types.TextContent(type="text",
                        text=f"Error: File not found: {abs_path}")]

                max_bytes = ragconfig.get('read_file_max_bytes', 102400)
                start_line = arguments.get("start_line")
                end_line = arguments.get("end_line")

                try:
                    with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
                        all_lines = f.readlines()
                except Exception as e:
                    return [types.TextContent(type="text", text=f"Error reading file: {e}")]

                total_lines = len(all_lines)
                file_size = os.path.getsize(abs_path)

                if start_line or end_line:
                    s = max(1, start_line or 1)
                    e = min(total_lines, end_line or total_lines)
                    selected = all_lines[s-1:e]
                    line_range = f"{s}-{e}"
                    first_line_num = s
                else:
                    selected = all_lines
                    line_range = f"1-{total_lines}"
                    first_line_num = 1

                content = ''.join(selected)
                truncated = False
                if len(content.encode('utf-8')) > max_bytes:
                    content = content[:max_bytes]
                    last_nl = content.rfind('\n')
                    if last_nl > 0:
                        content = content[:last_nl]
                    truncated = True

                # Add line numbers
                numbered = []
                for i, line in enumerate(content.splitlines(), start=first_line_num):
                    numbered.append(f"{i:4d} | {line}")
                display = '\n'.join(numbered)

                meta = f"**{abs_path}** | Lines {line_range} | {total_lines} total lines | {file_size} bytes"
                if truncated:
                    shown = len(numbered)
                    meta += (f"\n*File truncated at {shown} lines ({max_bytes} bytes). "
                             "Use start_line/end_line to read specific sections.*")

                return [types.TextContent(type="text", text=f"{meta}\n```\n{display}\n```")]

            elif name == "delete_by_pattern":
                pattern = arguments["pattern"]
                dry_run = arguments.get("dry_run", True)

                indexed = rag_milvus.list_indexed_files(db_path=db)
                all_paths = []
                for paths in indexed.values():
                    all_paths.extend(paths)

                matches = []
                for path in all_paths:
                    rel_path = os.path.relpath(path, project_root)
                    if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(path, pattern):
                        matches.append(path)

                if not matches:
                    return [types.TextContent(type="text",
                        text=f"No indexed files match pattern: {pattern}")]

                if dry_run:
                    output = f"**Dry run** — {len(matches)} files would be deleted:\n\n"
                    for p in sorted(matches):
                        output += f"- {p}\n"
                    output += f"\nRe-run with dry_run=false to delete."
                    return [types.TextContent(type="text", text=output)]

                deleted = 0
                for path in matches:
                    count = rag_milvus.delete_by_path(path, db_path=db)
                    if count > 0:
                        deleted += 1

                return [types.TextContent(type="text",
                    text=f"Deleted {deleted} files matching pattern: {pattern}")]

            else:
                raise ValueError(f"Unknown tool: {name}")

        except (Exception, asyncio.CancelledError) as e:
            return [types.TextContent(type="text", text=f"Error executing {name}: {str(e)}")]
