#!/bin/bash
# Pre-downloads an NL description (generator) model into the HuggingFace cache
# so the server can load it with HF_HUB_OFFLINE=1.
#
# Usage: ./download-description-model.sh [MODEL_KEY]
#
# MODEL_KEY is a key from model_config.DESCRIPTION_MODELS (see `python model_config.py --list`):
#   gemma-4-e4b        mlx-community/gemma-4-e4b-it-OptiQ-4bit   (default, ~7.5 GB)
#   gemma-4-e2b        mlx-community/gemma-4-e2b-it-OptiQ-4bit   (~4 GB)
#   qwen3.6-35b-a3b    mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit  (~22 GB)
#   qwen3-4b-2507      mlx-community/Qwen3-4B-Instruct-2507-4bit (~2.5 GB)
#   gemma-3-4b         mlx-community/gemma-3-4b-it-4bit          (legacy)
#
# With no argument, downloads whatever CODE_RAG_PROFILE / CODE_RAG_DESCRIPTION_MODEL_KEY
# resolve to (default profile: high -> gemma-4-e4b).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/venv/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "Error: Python venv not found. Run ./setup.sh first."
    exit 1
fi

if [ $# -ge 1 ]; then
    MODEL_KEY="$1"
else
    MODEL_KEY="$(CODE_RAG_PROFILE="${CODE_RAG_PROFILE:-high}" "$PYTHON" "$SCRIPT_DIR/model_config.py" --description-key)"
fi

read -r HF_ID SIZE_GB MIN_MLX_LM <<< "$("$PYTHON" "$SCRIPT_DIR/model_config.py" --description-info "$MODEL_KEY")"

INSTALLED_MLX_LM="$("$PYTHON" -c 'import importlib.metadata as m; print(m.version("mlx-lm"))' 2>/dev/null || echo 0)"
if [ "$(printf '%s\n%s\n' "$MIN_MLX_LM" "$INSTALLED_MLX_LM" | sort -V | head -1)" != "$MIN_MLX_LM" ]; then
    echo "Warning: $HF_ID needs mlx-lm >= $MIN_MLX_LM but $INSTALLED_MLX_LM is installed."
    echo "         Upgrade with: venv/bin/pip install --upgrade 'mlx-lm>=$MIN_MLX_LM' mlx"
fi

echo "======================================================================"
echo "Pre-downloading $HF_ID (~${SIZE_GB} GB, key: $MODEL_KEY)"
echo "======================================================================"

env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE "$PYTHON" -c "
from mlx_lm import load
model, tokenizer = load('$HF_ID', tokenizer_config={'trust_remote_code': False})
print('Loaded OK; model is cached for offline use.')
"

echo ""
echo "Select it with:  export CODE_RAG_DESCRIPTION_MODEL_KEY=$MODEL_KEY"
echo "(or CODE_RAG_PROFILE; with nothing set, the best cached model is used)"
echo "Descriptions are on by default; disable with CODE_RAG_DESCRIPTIONS=0."
echo "Existing cached descriptions stay valid; only new/changed chunks use the new model."
