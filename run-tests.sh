#!/bin/bash
# Run the code-rag test suite (stdlib unittest; no extra packages needed).
# Tests use a fake 64-dim embedder, so they run without the GPU or model downloads.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR"
exec "$SCRIPT_DIR/venv/bin/python" -m unittest discover -s tests -t . "$@"
