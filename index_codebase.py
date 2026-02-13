#!/usr/bin/env python3
"""
CLI tool to index a codebase with Milvus + MLX for semantic search.

The index is stored at {target_path}/.code-rag/milvus.db alongside the project.
"""

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path
import rag_milvus


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


def main():
    parser = argparse.ArgumentParser(description="Index a codebase for semantic search")
    parser.add_argument("--path", required=True, help="Path to codebase directory")
    parser.add_argument("--extensions", help="Comma-separated extensions (e.g., .java,.js,.ts)")
    parser.add_argument("--exclude-dirs", help="Extra directories to exclude (comma-separated, merged with .ragignore or defaults)")
    parser.add_argument("--no-jaxb-filter", action="store_true", help="Disable JAXB-generated Java file detection")
    parser.add_argument("--clear", action="store_true", help="Clear existing index before indexing")
    parser.add_argument("--full", action="store_true", help="Full re-index (ignore incremental hashes)")
    parser.add_argument("--limit", type=int, default=0, help="Max files to index (for profiling)")

    args = parser.parse_args()

    path = Path(args.path).resolve()
    if not path.exists() or not path.is_dir():
        print(f"Error: {path} is not a valid directory")
        sys.exit(1)

    # DB lives in the project directory
    db_path = str(path / ".code-rag" / "milvus.db")

    # Check if HTTP server is running (it holds an exclusive lock on the DB)
    pid_file = Path.home() / ".code-rag" / "server.pid"
    port = os.getenv("CODE_RAG_PORT", "7101")
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)  # Check process is alive
            # Process alive — check health
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
                urllib.request.urlopen(req, timeout=2)
                print(f"Error: code-rag server is running (PID {pid}) and holds the DB lock.")
                print(f"  Stop it first:  ./code-rag-server.sh stop")
                print(f"  Or index via MCP: use the index_directory tool in Claude Code")
                sys.exit(1)
            except (urllib.error.URLError, OSError):
                pass  # Process alive but not healthy — stale PID, OK to proceed
        except (ProcessLookupError, ValueError):
            pass  # Stale PID file, OK to proceed

    extensions = None
    if args.extensions:
        extensions = [ext.strip() if ext.startswith('.') else f'.{ext.strip()}'
                     for ext in args.extensions.split(',')]

    extra_excludes = None
    if args.exclude_dirs:
        extra_excludes = [d.strip() for d in args.exclude_dirs.split(',')]

    if args.clear:
        print(f"Clearing existing index...")
        rag_milvus.clear_collection(db_path=db_path)

    # Show active exclusion config
    excluded = rag_milvus.get_excluded_dirs(extra_excludes, project_root=str(path))
    ragignore_path = path / '.ragignore'
    if ragignore_path.exists():
        print(f"Using {ragignore_path} ({len(excluded)} exclusions)")
    else:
        print(f"Using default exclusions ({len(excluded)} dirs)")

    incremental = not args.full
    jaxb_filter = not args.no_jaxb_filter
    limit_msg = f", limit={args.limit}" if args.limit > 0 else ""
    print(f"Indexing {path} ({'incremental' if incremental else 'full re-index'}{limit_msg})...")

    start = time.time()
    stats = rag_milvus.index_directory(
        str(path), extensions, incremental=incremental,
        progress_callback=progress, max_files=args.limit,
        extra_excludes=extra_excludes, jaxb_filter=jaxb_filter,
        db_path=db_path,
    )
    elapsed = time.time() - start

    if incremental and stats['files_indexed'] == 0 and stats.get('files_removed', 0) == 0:
        print("\n  All files up to date (use --full to force re-index)")

    # Summary
    print(f"\n{'=' * 40}")
    print(f"  Files indexed:  {stats['files_indexed']}")
    if stats.get('files_skipped', 0):
        print(f"  Files skipped:  {stats['files_skipped']} (unchanged)")
    if stats.get('files_removed', 0):
        print(f"  Files removed:  {stats['files_removed']} (deleted/moved)")
    if stats.get('errors', 0):
        print(f"  Errors:         {stats['errors']}")
    print(f"  Total chunks:   {stats['chunks_created']}")
    print(f"  Time:           {elapsed:.1f}s")

    if elapsed > 0 and stats['files_indexed'] > 0:
        print(f"  Speed:          {stats['files_indexed'] / elapsed:.1f} files/s, "
              f"{stats['chunks_created'] / elapsed:.1f} chunks/s")

    if stats['by_language']:
        print(f"\n  By Language:")
        for lang, count in sorted(stats['by_language'].items(), key=lambda x: x[1], reverse=True):
            print(f"    {lang:15} {count:5} chunks")

    print(f"\n  Index: {db_path}")


if __name__ == "__main__":
    main()
