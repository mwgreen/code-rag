#!/bin/bash
# Downloads Qodo-Embed-1-1.5B from HuggingFace and quantizes to Q8 for MLX.
#
# The full-precision model is ~5.8 GB. After Q8 quantization it's ~1.6 GB
# with minimal quality loss.
#
# Prerequisites: run setup.sh first (creates venv with mlx_lm)
#
# Usage: ./download-model.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/models/qodo-embed-1-1.5b-mlx-q8"
PYTHON="$SCRIPT_DIR/venv/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "Error: Python venv not found. Run ./setup.sh first."
    exit 1
fi

# Check mlx_lm is available
if ! "$PYTHON" -c "import mlx_lm" 2>/dev/null; then
    echo "Installing mlx-lm for quantization..."
    "$SCRIPT_DIR/venv/bin/pip" install --quiet mlx-lm
fi

if [ -f "$MODEL_DIR/model.safetensors" ]; then
    echo "Model already exists at $MODEL_DIR"
    echo "Delete it first if you want to re-download."
    exit 0
fi

mkdir -p "$MODEL_DIR"

echo "======================================================================"
echo "Downloading Qodo-Embed-1-1.5B and quantizing to Q8"
echo "======================================================================"
echo ""
echo "This will:"
echo "  1. Download the full model from HuggingFace (~5.8 GB)"
echo "  2. Quantize to 8-bit (Q8, group_size=64) → ~1.6 GB"
echo "  3. Save to $MODEL_DIR"
echo ""
echo "This may take several minutes depending on your connection."
echo ""

# Download and quantize in one step using mlx_lm.convert
"$PYTHON" -c "
from mlx_lm.convert import convert

print('Downloading and quantizing (Q8, group_size=64)...')
convert(
    hf_path='Qodo/Qodo-Embed-1-1.5B',
    mlx_path='$MODEL_DIR',
    quantize=True,
    q_bits=8,
    q_group_size=64,
)
print('Done!')
"

if [ -f "$MODEL_DIR/model.safetensors" ]; then
    echo ""
    echo "======================================================================"
    echo "Model ready at: $MODEL_DIR"
    echo "Size: $(du -sh "$MODEL_DIR" | cut -f1)"
    echo "======================================================================"
else
    echo ""
    echo "Error: model.safetensors not found after conversion."
    echo "Check output above for errors."
    exit 1
fi
