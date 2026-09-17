#!/usr/bin/env python3
"""
CLI tool to index a codebase with Milvus + MLX for semantic search.

The index is stored at {target_path}/.code-rag/milvus.db alongside the project.

If the code-rag HTTP server is running, indexing is handed to it over HTTP
(POST /index) and this process just shows progress. The server is the only
writer to the Milvus Lite files, so there is no stop/index/start dance. With
no server running, indexing happens in this process.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def progress(current: int, total: int, filename: str = ""):
    """Print progress bar."""
    if total <= 0:
        return
    pct = int((current / total) * 100)
    bar_len = 30
    filled = int((current / total) * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    d = filename[-35:] if len(filename) > 35 else filename
    sys.stdout.write(f"\r  [{bar}] {current}/{total} ({pct}%) {d:<35}")
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")


def _server_url() -> str:
    return f"http://127.0.0.1:{os.getenv('CODE_RAG_PORT', '7101')}"


def server_health() -> dict | None:
    try:
        with urllib.request.urlopen(urllib.request.Request(f"{_server_url()}/health"), timeout=3) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def server_pid_alive() -> int | None:
    pid_file = Path.home() / ".code-rag" / "server.pid"
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError, ProcessLookupError):
        return None


def index_via_server(path: Path, full: bool, clear: bool, extensions) -> dict:
    body = json.dumps({"path": str(path), "project_root": str(path), "full": full,
                       "clear": clear, "extensions": extensions}).encode()
    req = urllib.request.Request(f"{_server_url()}/index", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            job = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Error: server refused index request: {e.read().decode(errors='replace')}")
    if job.get("kind") != "index" or job.get("status") not in ("queued", "running"):
        print(f"  Server already has a {job.get('kind')} job ({job.get('status')}) for this project; waiting for it.")
    print(f"  Server job {job['id']} started; the server keeps serving searches while it runs.")

    last = None
    while True:
        time.sleep(1.0)
        try:
            with urllib.request.urlopen(f"{_server_url()}/jobs/{job['id']}", timeout=5) as r:
                job = json.loads(r.read())
        except (urllib.error.URLError, OSError):
            print("\n  Lost contact with the server while waiting for the job.")
            raise SystemExit(1)
        p = job.get("progress") or {}
        if p.get("files_total"):
            progress(p.get("files_done", 0), p["files_total"], p.get("current", ""))
            last = p
        if job["status"] not in ("queued", "running"):
            break
    if last and last.get("files_done", 0) != last.get("files_total", 0):
        sys.stdout.write("\n")
    if job["status"] == "failed":
        raise SystemExit(f"Error: indexing failed: {job.get('error')}")
    return job.get("result") or {}


def main():
    parser = argparse.ArgumentParser(description="Index a codebase for semantic search")
    parser.add_argument("--path", required=True, help="Path to the project root")
    parser.add_argument("--extensions", help="Comma-separated extensions (e.g., .java,.js,.ts)")
    parser.add_argument("--exclude-dirs", help="Extra directories to exclude (comma-separated, merged with .ragignore or defaults)")
    parser.add_argument("--no-jaxb-filter", action="store_true", help="Disable JAXB-generated Java file detection")
    parser.add_argument("--clear", action="store_true", help="Clear existing index before indexing")
    parser.add_argument("--full", action="store_true", help="Full re-index (ignore incremental hashes)")
    parser.add_argument("--limit", type=int, default=0, help="Max files to index (for profiling; local mode only)")
    parser.add_argument("--local", action="store_true", help="Index in this process even if the server is running (unsafe if the server has this project open)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show INFO logs")
    args = parser.parse_args()

    from logging_setup import configure_logging
    configure_logging("INFO" if args.verbose else "WARNING")

    path = Path(args.path).resolve()
    if not path.exists() or not path.is_dir():
        print(f"Error: {path} is not a valid directory")
        sys.exit(1)

    db_path = str(path / ".code-rag" / "milvus.db")
    extensions = None
    if args.extensions:
        extensions = [ext.strip() if ext.strip().startswith('.') else f'.{ext.strip()}'
                      for ext in args.extensions.split(',')]
    extra_excludes = [d.strip() for d in args.exclude_dirs.split(',')] if args.exclude_dirs else None
    incremental = not args.full

    start = time.time()
    health = None if args.local else server_health()

    if health is not None:
        unsupported = [name for name, val in (("--limit", args.limit), ("--exclude-dirs", extra_excludes),
                                              ("--no-jaxb-filter", args.no_jaxb_filter)) if val]
        if unsupported:
            print(f"Note: {', '.join(unsupported)} only apply in local mode; ignored while the server is running.")
        print(f"Indexing {path} via running server ({'incremental' if incremental else 'full re-index'})...")
        stats = index_via_server(path, full=args.full, clear=args.clear, extensions=extensions)
    else:
        pid = server_pid_alive()
        if pid is not None and not args.local:
            print(f"Error: code-rag server process {pid} is alive but not answering /health "
                  f"(starting up or busy). Wait and retry, or pass --local if you are sure it "
                  f"does not have this project open.")
            sys.exit(1)

        import rag_milvus
        if args.clear:
            print("Clearing existing index...")
            rag_milvus.clear_collection(db_path=db_path)

        excluded = rag_milvus.get_excluded_dirs(extra_excludes, project_root=str(path))
        if (path / '.ragignore').exists():
            print(f"Using {path / '.ragignore'} ({len(excluded)} exclusions)")
        else:
            print(f"Using default exclusions ({len(excluded)} dirs)")

        limit_msg = f", limit={args.limit}" if args.limit > 0 else ""
        print(f"Indexing {path} ({'incremental' if incremental else 'full re-index'}{limit_msg})...")
        stats = rag_milvus.index_directory(
            str(path), extensions, incremental=incremental,
            progress_callback=progress, max_files=args.limit,
            extra_excludes=extra_excludes, jaxb_filter=not args.no_jaxb_filter,
            db_path=db_path,
        )
    elapsed = time.time() - start

    if incremental and stats.get('files_indexed', 0) == 0 and stats.get('files_removed', 0) == 0:
        print("\n  All files up to date (use --full to force re-index)")

    print(f"\n{'=' * 40}")
    print(f"  Files indexed:  {stats.get('files_indexed', 0)}")
    if stats.get('files_skipped'):
        print(f"  Files skipped:  {stats['files_skipped']} (unchanged)")
    if stats.get('files_removed'):
        print(f"  Files removed:  {stats['files_removed']} (deleted/moved/excluded)")
    if stats.get('errors'):
        print(f"  Errors:         {stats['errors']}")
    if stats.get('cancelled'):
        print("  Cancelled before completion")
    print(f"  Total chunks:   {stats.get('chunks_created', 0)}")
    print(f"  Time:           {elapsed:.1f}s")
    if elapsed > 0 and stats.get('files_indexed'):
        print(f"  Speed:          {stats['files_indexed'] / elapsed:.1f} files/s, "
              f"{stats.get('chunks_created', 0) / elapsed:.1f} chunks/s")
    if stats.get('by_language'):
        print("\n  By Language:")
        for lang, count in sorted(stats['by_language'].items(), key=lambda x: x[1], reverse=True):
            print(f"    {lang:15} {count:5} chunks")
    print(f"\n  Index: {db_path}")


if __name__ == "__main__":
    main()
