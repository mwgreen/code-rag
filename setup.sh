#!/bin/bash
# Setup script for code-rag on Apple Silicon Mac
# Single command to install everything: deps, model, and verify.

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "Code-RAG Setup for Apple Silicon Mac"
echo "======================================================================"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

# Check macOS version
if [[ $(uname) != "Darwin" ]]; then
    echo "Error: This script is for macOS only"
    exit 1
fi

# Check for Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "Error: This script requires Apple Silicon (M1/M2/M3/M4)"
    exit 1
fi

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Install with: brew install python@3.12"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "  Python $PYTHON_VERSION"

# Check macOS version
MACOS_VERSION=$(sw_vers -productVersion | cut -d'.' -f1)
if [[ $MACOS_VERSION -lt 15 ]]; then
    echo "  Warning: macOS 15+ recommended for optimal MLX performance (you have macOS $MACOS_VERSION)"
else
    echo "  macOS $MACOS_VERSION"
fi

echo ""

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "Error: node not found. Install with: brew install node"
    exit 1
fi
NODE_VERSION=$(node --version)
echo "  Node.js $NODE_VERSION"

# Install Node.js dependencies (code-chunk for AST-aware chunking)
echo ""
echo "Installing Node.js dependencies..."
if [ -f "package.json" ]; then
    npm install --silent 2>/dev/null || npm install
    echo "  code-chunk installed"
else
    echo "  package.json not found, skipping Node.js deps"
fi

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "  venv already exists, using existing"
else
    python3 -m venv venv
    echo "  venv created"
fi

# Activate virtual environment
source venv/bin/activate

echo ""
echo "Installing dependencies..."
echo "  This will take 1-2 minutes..."

# Install dependencies with correct versions
pip install --quiet --upgrade pip

echo "  Installing Milvus Lite..."
pip install --quiet "setuptools>=70.0,<82.0" "pymilvus[milvus-lite]"

echo "  Installing MLX embeddings..."
pip install --quiet mlx-embeddings mlx mlx-metal

echo "  Installing MLX LM (NL descriptions)..."
pip install --quiet mlx-lm

echo "  Installing transformers <5.0 (required for compatibility)..."
pip install --quiet "transformers<5.0"

echo "  Installing utilities..."
pip install --quiet python-dotenv pyyaml

echo "  Installing tree-sitter (AST chunking)..."
pip install --quiet "tree-sitter>=0.25.0" "tree-sitter-java>=0.23.0" "tree-sitter-python>=0.25.0" "tree-sitter-typescript>=0.23.0"

echo "  Installing watchdog (file watcher)..."
pip install --quiet "watchdog>=4.0.0"

echo "  Installing MCP (optional - for Claude Code integration)..."
pip install --quiet mcp || echo "  MCP install failed (optional, can skip)"

echo "  All dependencies installed"

# Install architecture patches for mlx-embeddings
echo ""
echo "Installing architecture patches for mlx-embeddings..."
MLX_MODELS_DIR=$(python3 -c "import mlx_embeddings.models; import os; print(os.path.dirname(mlx_embeddings.models.__file__))")
if [ -n "$MLX_MODELS_DIR" ]; then
    if [ -f "patches/mlx_embeddings_qwen2.py" ]; then
        cp patches/mlx_embeddings_qwen2.py "$MLX_MODELS_DIR/qwen2.py"
        echo "  Qwen2 architecture installed (for Qodo-Embed-1-1.5B)"
    else
        echo "  Warning: patches/mlx_embeddings_qwen2.py not found"
    fi
    if [ -f "patches/mlx_embeddings_codexembed2b.py" ]; then
        cp patches/mlx_embeddings_codexembed2b.py "$MLX_MODELS_DIR/codexembed2b.py"
        echo "  CodexEmbed2B architecture installed (for SFR-Embedding-Code-2B)"
    else
        echo "  Warning: patches/mlx_embeddings_codexembed2b.py not found"
    fi
else
    echo "  Warning: Could not find mlx-embeddings models directory"
fi

# Create data directory
mkdir -p data

# Download and quantize embedding model
echo ""
"$SCRIPT_DIR/download-model.sh"

# Pre-download NL description models (used with CODE_RAG_DESCRIPTIONS=1)
echo ""
echo "Pre-downloading NL description models..."
"$PYTHON" << 'DESCEOF'
from mlx_lm import load

# Default: Qwen3-4B
try:
    print("  Downloading Qwen3-4B-MLX-4bit (~2.5 GB)...")
    model, tokenizer = load("Qwen/Qwen3-4B-MLX-4bit", tokenizer_config={"trust_remote_code": False})
    print("  Qwen3-4B cached for offline use")
    del model, tokenizer
except Exception as e:
    print(f"  Qwen3-4B download skipped: {e}")

# Alternative: Gemma 3 4B
try:
    print("  Downloading gemma-3-4b-it-4bit (~2.5 GB)...")
    model, tokenizer = load("mlx-community/gemma-3-4b-it-4bit", tokenizer_config={"trust_remote_code": False})
    print("  Gemma-3-4B cached for offline use")
    del model, tokenizer
except Exception as e:
    print(f"  Gemma-3-4B download skipped: {e}")

print("  (enable descriptions with CODE_RAG_DESCRIPTIONS=1)")
DESCEOF

# Test installation
echo ""
echo "Testing installation..."

python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')

# Test imports
try:
    import rag_milvus
    import mlx.core as mx
    from mlx_embeddings.utils import load
    print("  All imports successful")
except Exception as e:
    print(f"  Import failed: {e}")
    sys.exit(1)

# Test embedding
try:
    print("  Testing Qodo-Embed-1-1.5B (Q8) embedding...")
    model, tokenizer = rag_milvus.get_mlx_model()
    test_emb = rag_milvus.embed_texts(["test"])
    print(f"  Embedding works ({len(test_emb[0])} dimensions)")
except Exception as e:
    print(f"  Embedding test failed: {e}")
    sys.exit(1)

print("")
print("  All tests passed!")
PYEOF

if [ $? -ne 0 ]; then
    echo "Installation test failed"
    exit 1
fi

# Generate MCP config template
echo ""
echo "Generating MCP server configuration..."

VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"
MCP_SERVER="$SCRIPT_DIR/mcp_server.py"
MILVUS_DB="$SCRIPT_DIR/data/milvus.db"

cat > mcp-config-template.json << MCPEOF
{
  "mcpServers": {
    "code-rag": {
      "command": "$VENV_PYTHON",
      "args": ["-u", "$MCP_SERVER"],
      "env": {
        "PYTHONPATH": "$SCRIPT_DIR",
        "MILVUS_DB_PATH": "$MILVUS_DB"
      }
    }
  }
}
MCPEOF

echo "  MCP config template created: mcp-config-template.json"

# Success!
echo ""
echo "======================================================================"
echo "Installation Complete!"
echo "======================================================================"
echo ""
echo "Default embedding model: Qodo-Embed-1-1.5B (Q8, 1536 dims, MLX)"
echo ""
echo "Next steps:"
echo ""
echo "1. Index your codebase:"
echo "   ./index.sh /path/to/your/project"
echo ""
echo "2. Start the server:"
echo "   ./code-rag-server.sh start"
echo ""
echo "3. For Claude Code integration:"
echo "   - Copy mcp-config-template.json contents to your .mcp.json"
echo "   - Or merge into existing .mcp.json"
echo "   - Restart Claude Code"
echo ""
echo "Optional: Alternative models"
echo ""
echo "  Embedding (SFR-Embedding-Code-2B, Gemma 2, 2304 dims):"
echo "    ./download-sfr-embed.sh"
echo "    export EMBED_MODEL_PATH=./models/sfr-embed-code-2b-mlx-q8"
echo ""
echo "  NL descriptions (Gemma 3 4B instead of Qwen3 4B):"
echo "    export CODE_RAG_DESCRIPTION_MODEL=mlx-community/gemma-3-4b-it-4bit"
echo ""
echo "  Note: Switching embedding models requires re-indexing."
echo ""
echo "======================================================================"
