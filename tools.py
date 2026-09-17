"""
Shared tool definitions and project context for code-rag MCP servers.
Both stdio (mcp_server.py) and HTTP (http_server.py) import from here.

Every blocking operation runs in a worker thread (asyncio.to_thread) so the
event loop, and with it /health, stays responsive. Long operations (directory
indexing, reconcile) run as background jobs and are polled with index_status.
"""

import asyncio
import contextvars
import fnmatch
import logging
import os
from pathlib import Path
from typing import Optional

from mcp.server import Server
from mcp import types

import file_watcher
import jobs
import rag_milvus

logger = logging.getLogger("code-rag.tools")

# Below this cosine similarity the best semantic hit is probably not what the
# user meant; the note nudges the model toward grep.
LOW_SIMILARITY_NOTE = 0.40


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
    _current_project_root.set(root)


def get_current_project_root() -> str:
    root = _current_project_root.get()
    if root is None:
        raise ProjectNotConfiguredError(_PROJECT_ERROR_MSG)
    return root


def get_db_path() -> str:
    return str(Path(get_current_project_root()) / ".code-rag" / "milvus.db")


def resolve_under_root(path: str, project_root: str) -> str:
    """Absolute real path of `path`, or ValueError if it is outside the project root.

    index_file / index_directory used to accept any absolute path, which is how
    files from a different project ended up in one index."""
    real_root = os.path.realpath(project_root)
    abs_path = os.path.realpath(path if os.path.isabs(path) else os.path.join(real_root, path))
    if abs_path != real_root and not abs_path.startswith(real_root + os.sep):
        raise ValueError(f"Path must be within the project root ({project_root}): {path}")
    return abs_path


# --- Relevance filtering ---

def _apply_relevance_floor(results: list[dict], min_relevance: float) -> tuple[list[dict], bool]:
    """Drop semantic hits whose cosine similarity is below min_relevance.
    Keyword-only hits (similarity None) always pass. Returns (results, all_below_floor)."""
    if min_relevance <= 0 or not results:
        return results, False
    filtered = [r for r in results
                if r.get('similarity') is None or r['similarity'] >= min_relevance]
    if not filtered:
        return results, True
    return filtered, False


def _format_search_results(results: list[dict], min_relevance: float, grouped: bool = False) -> str:
    results, below_floor = _apply_relevance_floor(results, min_relevance)
    text = format_results_grouped(results) if grouped else format_results(results)
    if below_floor:
        text = ("*Warning: All results are below the configured relevance threshold "
                f"({min_relevance:.2f}). Showing best matches anyway.*\n\n" + text)
    if results:
        text += "\n*To read the full file, use the read_file tool with the path shown above.*"
        path_counts: dict[str, int] = {}
        for r in results:
            p = r.get('path')
            if p:
                path_counts[p] = path_counts.get(p, 0) + 1
        multi_hit = sorted([(p, n) for p, n in path_counts.items() if n >= 2], key=lambda x: -x[1])
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
    similarities = [r['similarity'] for r in results if r.get('similarity') is not None]
    if similarities and max(similarities) < LOW_SIMILARITY_NOTE:
        output.append(
            "*Note: Low semantic similarity — results may not match your query well. "
            "Consider using Grep for exact keyword matching.*\n"
        )

    for r in results:
        output.append(f"### {r['path']}:{r.get('start_line', '?')}-{r.get('end_line', '?')}")
        meta_parts = [f"**Language:** {r.get('language', '')}", f"**Type:** {r.get('type', '')}"]
        if r.get('class_name'):
            meta_parts.append(f"**Class:** {r['class_name']}")
        if r.get('component'):
            meta_parts.append(f"**Component:** {r['component']}")
        if r.get('similarity') is not None:
            meta_parts.append(f"**Relevance:** {r['similarity']:.2f}")
        source = r.get('source')
        if source == 'keyword':
            meta_parts.append("**Match:** keyword")
        elif source == 'both':
            meta_parts.append("**Match:** semantic+keyword")
        output.append(" | ".join(meta_parts))
        if r.get('description'):
            output.append(f"**Summary:** {r['description']}")
        output.append(f"```{r.get('language', '')}\n{r.get('content', '')}\n```")
        output.append("")

    return "\n".join(output)


def format_results_grouped(results: list[dict]) -> str:
    if not results:
        return "No results found."
    grouped: dict[str, list[dict]] = {}
    for r in results:
        grouped.setdefault(r.get('type', 'code'), []).append(r)
    output = []
    for type_name in ['documentation', 'code', 'config']:
        if type_name in grouped:
            output.append(f"## {type_name.title()} Results\n")
            output.append(format_results(grouped[type_name]))
    for type_name, items in grouped.items():
        if type_name not in ('documentation', 'code', 'config'):
            output.append(f"## {type_name.title()} Results\n")
            output.append(format_results(items))
    return "\n".join(output)


_FILE_LIST_CAP_PER_SECTION = 1000


def format_file_list(files: dict[str, list[str]], path_glob: Optional[str] = None) -> str:
    if not files:
        return "No files indexed."
    if path_glob:
        files = {t: [p for p in paths if fnmatch.fnmatch(p, path_glob)] for t, paths in files.items()}
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
            output.append(f"\n...and {len(paths) - _FILE_LIST_CAP_PER_SECTION} more "
                          f"(narrow with the path_glob argument, e.g. \"docs/**/*.md\")")
        output.append("")
    return "\n".join(output)


def format_stats(stats: dict, db_path: str) -> str:
    lines = [f"**Total Files:** {stats['total_files']}", f"**Total Chunks:** {stats['total_chunks']}",
             "", "### By Language"]
    for lang, count in sorted(stats['by_language'].items(), key=lambda x: x[1], reverse=True):
        lines.append(f"- {lang}: {count} chunks")
    lines.append("\n### By Type")
    for t, count in sorted(stats['by_type'].items(), key=lambda x: x[1], reverse=True):
        lines.append(f"- {t}: {count} chunks")
    lines.append(f"\n**Index Location:** {db_path}")
    return "\n".join(lines)


def format_job(job: Optional[jobs.Job], heading: str = "") -> str:
    if job is None:
        return "No indexing job found for this project."
    d = job.to_dict()
    lines = []
    if heading:
        lines.append(heading)
    lines += [f"**Job:** {d['id']} ({d['kind']})", f"**Status:** {d['status']}",
              f"**Elapsed:** {d['elapsed_s']}s"]
    p = d.get('progress') or {}
    if p.get('files_total'):
        lines.append(f"**Progress:** {p.get('files_done', 0)}/{p['files_total']} files"
                     + (f" (current: {p['current']})" if p.get('current') else ""))
    if p.get('reason'):
        lines.append(f"**Reason:** {p['reason']}")
    if d['error']:
        lines.append(f"**Error:** {d['error']}")
    result = d.get('result')
    if isinstance(result, dict):
        lines.append("")
        lines.append("### Result")
        for key in ('files_indexed', 'files_skipped', 'files_removed', 'chunks_created', 'removed_missing',
                    'fts_orphans_removed', 'fts_rebuilt', 'errors', 'cancelled'):
            if key in result and result[key]:
                lines.append(f"- {key}: {result[key]}")
        if result.get('by_language'):
            lines.append("- by language: " + ", ".join(
                f"{k} {v}" for k, v in sorted(result['by_language'].items(), key=lambda x: -x[1])))
    if job.active:
        lines.append("\n*Poll with the index_status tool.*")
    return "\n".join(lines)


def format_verify(report: dict) -> str:
    lines = [f"**Consistent:** {'yes' if report['consistent'] else 'no'}",
             f"**Files on disk (indexable):** {report['disk_files']}",
             f"**Files indexed:** {report['indexed_files']}",
             f"**Chunks:** Milvus {report['indexed_chunks']}, FTS {report['fts_chunks']}", ""]
    labels = {
        'missing_on_disk': 'Indexed but deleted/excluded on disk',
        'not_indexed': 'On disk but not indexed',
        'changed': 'Changed since indexed',
        'fts_orphans': 'Keyword-index rows with no vector rows',
        'fts_mismatch': 'Files whose keyword rows differ from vector rows',
    }
    for key, label in labels.items():
        lines.append(f"- {label}: {report[key]}")
        for p in report.get(f'{key}_sample', []):
            lines.append(f"    - {p}")
    if not report['consistent']:
        lines.append("\n*Run verify_index with repair=true to fix (runs as a background job).*")
    return "\n".join(lines)


# --- Blocking helpers (run via asyncio.to_thread) ---

def _read_file(arguments: dict, project_root: str, ragconfig: dict) -> str:
    abs_path = resolve_under_root(arguments["path"], project_root)
    if not os.path.isfile(abs_path):
        return f"Error: File not found: {abs_path}"

    max_bytes = ragconfig.get('read_file_max_bytes', 102400)
    start_line = arguments.get("start_line")
    end_line = arguments.get("end_line")
    try:
        with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
            all_lines = f.readlines()
    except Exception as e:
        return f"Error reading file: {e}"

    total_lines = len(all_lines)
    file_size = os.path.getsize(abs_path)
    if start_line or end_line:
        s = max(1, start_line or 1)
        e = min(total_lines, end_line or total_lines)
        selected = all_lines[s - 1:e]
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

    numbered = [f"{i:4d} | {line}" for i, line in enumerate(content.splitlines(), start=first_line_num)]
    meta = f"**{abs_path}** | Lines {line_range} | {total_lines} total lines | {file_size} bytes"
    if truncated:
        meta += (f"\n*File truncated at {len(numbered)} lines ({max_bytes} bytes). "
                 "Use start_line/end_line to read specific sections.*")
    return f"{meta}\n```\n" + '\n'.join(numbered) + "\n```"


def _delete_by_pattern(pattern: str, dry_run: bool, project_root: str, db: str) -> str:
    indexed = rag_milvus.list_indexed_files(db_path=db)
    all_paths = [p for paths in indexed.values() for p in paths]
    matches = [p for p in all_paths
               if fnmatch.fnmatch(os.path.relpath(p, project_root), pattern) or fnmatch.fnmatch(p, pattern)]
    if not matches:
        return f"No indexed files match pattern: {pattern}"
    if dry_run:
        out = f"**Dry run** — {len(matches)} files would be deleted:\n\n"
        out += "".join(f"- {p}\n" for p in sorted(matches))
        return out + "\nRe-run with dry_run=false to delete."
    deleted = sum(1 for p in matches if rag_milvus.delete_by_path(p, db_path=db) > 0)
    return f"Deleted {deleted} files matching pattern: {pattern}"


def _watcher_text(project_root: str) -> str:
    status = file_watcher.get_watcher_status().get(project_root)
    if status is None:
        return "No active file watcher for this project."
    s = status['stats']
    lines = [
        "**File Watcher:** active",
        f"**Pending changes:** {status['pending']}",
        f"**Currently processing:** {status['processing']}",
        "",
        "### Cumulative Stats",
        f"- Files indexed: {s['files_indexed']}",
        f"- Files deleted: {s['files_deleted']}",
        f"- Batches processed: {s['batches_processed']}",
        f"- Reconciles: {s.get('reconciles', 0)}",
        f"- Errors: {s['errors']}",
    ]
    job = status.get('active_job')
    if job:
        lines += ["", f"### Active job: {job['id']} ({job['kind']}, {job['status']})"]
        p = job.get('progress') or {}
        if p.get('files_total'):
            lines.append(f"- {p.get('files_done', 0)}/{p['files_total']} files")
    return "\n".join(lines)


# --- Tool registration ---

LANGUAGES = ["java", "javascript", "typescript", "yaml", "json", "xml", "markdown",
             "properties", "gradle", "graphql", "protobuf"]


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
                        "query": {"type": "string",
                                  "description": "Natural language search query (e.g., 'user authentication logic')"},
                        "n": {"type": "integer", "description": "Number of results to return (default: 5)", "default": 5},
                        "language": {"type": "string", "description": "Filter by language", "enum": LANGUAGES},
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="search_docs",
                description="Search the YAML documentation for information about components and architecture",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string",
                                  "description": "Natural language query about the system (e.g., 'how does UserGrid work?')"},
                        "n": {"type": "integer", "description": "Number of results to return (default: 5)", "default": 5},
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="search_all",
                description="Search everything - code, documentation, and config files",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language search query"},
                        "n": {"type": "integer", "description": "Number of results to return (default: 10)", "default": 10},
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="index_file",
                description="Index or re-index a single file (must be inside the project root)",
                inputSchema={
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "Absolute path to file"}},
                    "required": ["path"],
                },
            ),
            types.Tool(
                name="index_directory",
                description=("Index all supported files in a directory inside the project root. Runs in the "
                             "background; returns a job id to poll with index_status."),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to directory (default: project root)"},
                        "full": {"type": "boolean", "description": "Re-embed every file, ignoring content hashes",
                                 "default": False},
                    },
                    "required": [],
                },
            ),
            types.Tool(
                name="index_status",
                description="Status of the running or most recent indexing/reconcile job for this project",
                inputSchema={
                    "type": "object",
                    "properties": {"job_id": {"type": "string", "description": "Optional job id"}},
                    "required": [],
                },
            ),
            types.Tool(
                name="verify_index",
                description=("Compare disk, vector index and keyword index for this project and report "
                             "drift. With repair=true, fix it in a background job."),
                inputSchema={
                    "type": "object",
                    "properties": {"repair": {"type": "boolean", "default": False,
                                              "description": "Start a reconcile job that repairs any drift"}},
                    "required": [],
                },
            ),
            types.Tool(
                name="list_indexed",
                description="List indexed files grouped by type. Up to 1000 per type; pass `path_glob` to narrow.",
                inputSchema={
                    "type": "object",
                    "properties": {"path_glob": {"type": "string",
                                                 "description": "Optional fnmatch glob to filter paths, e.g. 'docs/**/*.md'"}},
                    "required": [],
                },
            ),
            types.Tool(
                name="get_stats",
                description="Get index statistics (file count, chunk count by language/type)",
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
            types.Tool(
                name="watcher_status",
                description="Get file watcher status (pending changes, indexing stats, active job)",
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
            types.Tool(
                name="read_file",
                description="Read the full content of a file in the project. Use after searching to see complete file context.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute path to the file to read"},
                        "start_line": {"type": "integer", "description": "1-based start line (optional, for reading a range)"},
                        "end_line": {"type": "integer", "description": "1-based end line, inclusive (optional)"},
                    },
                    "required": ["path"],
                },
            ),
            types.Tool(
                name="delete_by_pattern",
                description="Delete indexed entries matching a glob pattern. Use dry_run=true (default) to preview.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string",
                                    "description": "Glob pattern matched against indexed file paths (e.g. '**/*.yaml', '.playwright-mcp/*')"},
                        "dry_run": {"type": "boolean", "default": True,
                                    "description": "If true (default), show what would be deleted without deleting"},
                    },
                    "required": ["pattern"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        arguments = arguments or {}
        try:
            db = get_db_path()
            project_root = get_current_project_root()
        except ProjectNotConfiguredError as e:
            return [types.TextContent(type="text", text=str(e))]

        # Ensure the file watcher (and its startup reconcile) is running for this project
        try:
            await file_watcher.ensure_watcher(project_root, db)
        except Exception as e:
            logger.warning("Could not start watcher for %s: %s", project_root, e)

        ragconfig = rag_milvus.load_ragconfig(project_root)
        min_relevance = ragconfig.get('min_relevance', 0.0)

        def text(t: str) -> list[types.TextContent]:
            return [types.TextContent(type="text", text=t)]

        try:
            if name == "search_code":
                results = await rag_milvus.search_async(
                    arguments["query"], arguments.get("n", 5),
                    type_filter="code", language_filter=arguments.get("language"), db_path=db)
                return text(_format_search_results(results, min_relevance))

            if name == "search_docs":
                results = await rag_milvus.search_async(
                    arguments["query"], arguments.get("n", 5), type_filter="documentation", db_path=db)
                return text(_format_search_results(results, min_relevance))

            if name == "search_all":
                results = await rag_milvus.search_async(arguments["query"], arguments.get("n", 10), db_path=db)
                return text(_format_search_results(results, min_relevance, grouped=True))

            if name == "index_file":
                path = resolve_under_root(arguments["path"], project_root)
                count = await asyncio.to_thread(rag_milvus.add_file, path, True, db)
                return text(f"Indexed {path}\n\n**Chunks created:** {count}")

            if name == "index_directory":
                path = resolve_under_root(arguments.get("path") or project_root, project_root)
                job = jobs.start_index_job(project_root, db, path=path, full=bool(arguments.get("full", False)))
                heading = ("Indexing started in the background." if job.kind == "index" and job.status in ("queued", "running")
                           else "A job is already running for this project.")
                return text(format_job(job, heading))

            if name == "index_status":
                job_id = arguments.get("job_id")
                if job_id:
                    job = jobs.get_job(job_id)
                else:
                    job = jobs.active_job(project_root) or next(iter(jobs.list_jobs(project_root)), None)
                return text(format_job(job))

            if name == "verify_index":
                if arguments.get("repair"):
                    job = jobs.start_reconcile_job(project_root, db, reason="verify_index repair")
                    return text(format_job(job, "Repair started in the background."))
                report = await asyncio.to_thread(rag_milvus.verify_index, project_root, db)
                return text(format_verify(report))

            if name == "list_indexed":
                files = await asyncio.to_thread(rag_milvus.list_indexed_files, db)
                return text(format_file_list(files, path_glob=arguments.get("path_glob")))

            if name == "get_stats":
                stats = await asyncio.to_thread(rag_milvus.get_stats, db)
                return text(format_stats(stats, db))

            if name == "watcher_status":
                return text(_watcher_text(project_root))

            if name == "read_file":
                return text(await asyncio.to_thread(_read_file, arguments, project_root, ragconfig))

            if name == "delete_by_pattern":
                return text(await asyncio.to_thread(
                    _delete_by_pattern, arguments["pattern"], arguments.get("dry_run", True), project_root, db))

            raise ValueError(f"Unknown tool: {name}")

        except ValueError as e:
            return text(f"Error: {e}")
        except Exception as e:
            logger.exception("Tool %s failed", name)
            return text(f"Error executing {name}: {e}")
