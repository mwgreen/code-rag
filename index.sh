#!/bin/bash
# Index a project: ./index.sh /path/to/project [--full] [--clear] [...]
# The index is stored at {project}/.code-rag/milvus.db.
# If the code-rag server is running, indexing is delegated to it (no need to stop it).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 1 ] || [[ "$1" == --* ]]; then
    echo "Usage: $0 /path/to/project [--full] [--clear] [--extensions .java,.ts] [--local] [-v]" >&2
    echo "  (a project path is required; the old default of indexing the parent directory was removed)" >&2
    exit 2
fi

TARGET_PATH="$1"
shift

export PYTHONPATH="$SCRIPT_DIR"
exec "$SCRIPT_DIR/venv/bin/python" "$SCRIPT_DIR/index_codebase.py" --path "$TARGET_PATH" "$@"
