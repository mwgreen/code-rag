#!/bin/bash
# Downloads an embedding model from HuggingFace and quantizes it to Q8 for MLX.
#
# Usage: ./download-embed-model.sh [MODEL_KEY]
#
# MODEL_KEY is a key from model_config.EMBED_MODELS (see `python model_config.py --list`):
#   qwen3-embed-4b      Qwen3-Embedding-4B   (default, 2560 dims, ~4.5 GB at Q8)
#   qwen3-embed-0.6b    Qwen3-Embedding-0.6B (1024 dims, ~0.7 GB)
#   sfr-embed-code-2b   SFR-Embedding-Code-2B_R (legacy, needs codexembed2b patch)
#   qodo-embed-1.5b     Qodo-Embed-1-1.5B (legacy, needs qwen2 patch)
#
# With no argument, downloads whatever CODE_RAG_PROFILE / CODE_RAG_EMBED_MODEL
# resolve to (default profile: high -> qwen3-embed-4b).
#
# Prerequisites: run setup.sh first (creates venv with mlx-embeddings).
# Switching embedding models requires re-indexing every project.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/venv/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "Error: Python venv not found. Run ./setup.sh first."
    exit 1
fi

# Resolve which model to fetch. An explicit argument wins; otherwise ask the
# registry, ignoring any already-downloaded model so a fresh machine gets the
# configured default rather than "nothing".
if [ $# -ge 1 ]; then
    MODEL_KEY="$1"
else
    MODEL_KEY="$(CODE_RAG_PROFILE="${CODE_RAG_PROFILE:-high}" "$PYTHON" "$SCRIPT_DIR/model_config.py" --embed-key)"
fi

read -r HF_ID LOCAL_DIR PATCH <<< "$("$PYTHON" "$SCRIPT_DIR/model_config.py" --embed-info "$MODEL_KEY")"
MODEL_DIR="$SCRIPT_DIR/models/$LOCAL_DIR"

if [ -f "$MODEL_DIR/model.safetensors" ] || [ -f "$MODEL_DIR/model-00001-of-00002.safetensors" ]; then
    echo "Model already exists at $MODEL_DIR"
    echo "Delete it first if you want to re-download."
    exit 0
fi

echo "======================================================================"
echo "Downloading $HF_ID and quantizing to Q8 (key: $MODEL_KEY)"
echo "======================================================================"
echo ""
echo "  1. Download the full-precision model from HuggingFace"
echo "  2. Quantize to 8-bit (Q8, group_size=64)"
echo "  3. Save to $MODEL_DIR"
if [ "$PATCH" != "none" ]; then
    echo ""
    echo "  Note: this model needs patches/mlx_embeddings_${PATCH}.py installed"
    echo "  into mlx-embeddings (setup.sh does this)."
fi
echo ""

# Allow network for this one-off download; the server forces offline mode.
env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE "$PYTHON" -c "
try:
    from mlx_embeddings import convert            # mlx-embeddings >= 0.1.0
except ImportError:
    from mlx_embeddings.utils import convert      # mlx-embeddings 0.0.x

# mlx-embeddings 0.1.0 copies *.json out of the HF cache with shutil.copy, which
# preserves the cache's read-only mode; tokenizer.save_pretrained() then fails with
# 'Permission denied' overwriting tokenizer.json and config.json (with the
# quantization block) is never written. Copy file contents only, never mode bits.
import shutil
from pathlib import Path
def _copy_writable(src, dst, *args, **kwargs):
    dst = Path(dst)
    if dst.is_dir():
        dst = dst / Path(src).name
    if dst.exists():
        dst.chmod(0o644)
    shutil.copyfile(src, dst)
    return str(dst)
shutil.copy = _copy_writable

print('Downloading and quantizing (Q8, group_size=64)...')
convert(
    hf_path='$HF_ID',
    mlx_path='$MODEL_DIR',
    quantize=True,
    q_bits=8,
    q_group_size=64,
)
print('Done!')
"

if [ -f "$MODEL_DIR/model.safetensors" ] || [ -f "$MODEL_DIR/model-00001-of-00002.safetensors" ]; then
    echo ""
    echo "======================================================================"
    echo "Model ready at: $MODEL_DIR"
    echo "Size: $(du -sh "$MODEL_DIR" | cut -f1)"
    echo "======================================================================"
    echo ""
    echo "Select it with:  export CODE_RAG_EMBED_MODEL=$MODEL_KEY"
    echo "(or CODE_RAG_PROFILE; with nothing set, the best downloaded model is used)"
    echo ""
    echo "Note: switching embedding models requires re-indexing: ./index.sh /path/to/project --clear"
else
    echo ""
    echo "Error: model weights not found after conversion."
    echo "Check output above for errors."
    exit 1
fi
