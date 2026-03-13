#!/bin/bash
# Downloads SFR-Embedding-Code-2B_R from HuggingFace and quantizes to Q8 for MLX.
#
# The full-precision model is ~8 GB. After Q8 quantization it's ~2.1 GB
# with minimal quality loss.
#
# Prerequisites: run setup.sh first (creates venv with mlx-embeddings + codexembed2b patch)
#
# Usage: ./download-sfr-embed.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/models/sfr-embed-code-2b-mlx-q8"
PYTHON="$SCRIPT_DIR/venv/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "Error: Python venv not found. Run ./setup.sh first."
    exit 1
fi

# Check for existing model (also handle sharded output)
if [ -f "$MODEL_DIR/model.safetensors" ] || [ -f "$MODEL_DIR/model-00001-of-00002.safetensors" ]; then
    echo "Model already exists at $MODEL_DIR"
    echo "Delete it first if you want to re-download."
    exit 0
fi

echo "======================================================================"
echo "Downloading SFR-Embedding-Code-2B_R and quantizing to Q8"
echo "======================================================================"
echo ""
echo "This will:"
echo "  1. Download the full model from HuggingFace (~8 GB)"
echo "  2. Quantize to 8-bit (Q8, group_size=64) → ~2.1 GB"
echo "  3. Save to $MODEL_DIR"
echo ""
echo "This may take several minutes depending on your connection."
echo ""

# Download and quantize using mlx_embeddings (understands embedding model architectures)
"$PYTHON" -c "
from mlx_embeddings.utils import convert

print('Downloading and quantizing (Q8, group_size=64)...')
convert(
    hf_path='Salesforce/SFR-Embedding-Code-2B_R',
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
    echo "To use this model, set the environment variable:"
    echo "  export EMBED_MODEL_PATH=$MODEL_DIR"
    echo ""
    echo "Note: Switching models requires re-indexing your projects."
else
    echo ""
    echo "Error: model weights not found after conversion."
    echo "Check output above for errors."
    exit 1
fi
