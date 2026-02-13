#!/bin/bash
# Wrapper script to run indexer with venv Python
# Defaults to parent directory if no path specified
# DB is stored at {target_path}/.code-rag/milvus.db

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Use provided path or default to project root
TARGET_PATH="${1:-$PROJECT_ROOT}"

export PYTHONPATH="$SCRIPT_DIR"

"$SCRIPT_DIR/venv/bin/python" "$SCRIPT_DIR/index_codebase.py" --path "$TARGET_PATH" "${@:2}"
