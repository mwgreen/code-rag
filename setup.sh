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
    echo "Error: This script requires Apple Silicon (M1 or later)"
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
if [[ $MACOS_VERSION -lt 26 ]] && sysctl -n machdep.cpu.brand_string 2>/dev/null | grep -q 'M5'; then
    echo "  Note: MLX uses the M5 GPU neural accelerators only on macOS 26.2+"
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
pip install --quiet "mlx>=0.32.2" mlx-metal "mlx-embeddings>=0.1.0"

echo "  Installing MLX LM (NL descriptions)..."
pip install --quiet "mlx-lm>=0.31.3"

echo "  Installing transformers 5.x (required by mlx-embeddings>=0.1.0 and mlx-lm>=0.31)..."
pip install --quiet "transformers[sentencepiece]>=5.0.0"

echo "  Installing utilities..."
pip install --quiet python-dotenv pyyaml

echo "  Installing tree-sitter (AST chunking)..."
pip install --quiet "tree-sitter>=0.25.0" "tree-sitter-java>=0.23.0" "tree-sitter-python>=0.25.0" "tree-sitter-typescript>=0.23.0"

echo "  Installing watchdog (file watcher)..."
pip install --quiet "watchdog>=4.0.0"

echo "  Installing MCP (optional - for Claude Code integration)..."
pip install --quiet "mcp>=1.0.0,<2.0" || echo "  MCP install failed (optional, can skip)"

echo "  All dependencies installed"

# Install architecture patches for mlx-embeddings.
# Only the legacy models (Qodo-Embed = qwen2, SFR-Embedding-Code = codexembed2b) need
# these; Qwen3-Embedding is supported natively. Installed anyway so the legacy
# profile keeps working. A file already shipped by the package is never overwritten.
echo ""
echo "Installing legacy architecture patches for mlx-embeddings..."
MLX_MODELS_DIR=$(python3 -c "import mlx_embeddings.models; import os; print(os.path.dirname(mlx_embeddings.models.__file__))")
if [ -n "$MLX_MODELS_DIR" ]; then
    for patch in qwen2 codexembed2b; do
        if [ ! -f "patches/mlx_embeddings_${patch}.py" ]; then
            echo "  Warning: patches/mlx_embeddings_${patch}.py not found"
        elif [ -f "$MLX_MODELS_DIR/${patch}.py" ] && ! cmp -s "patches/mlx_embeddings_${patch}.py" "$MLX_MODELS_DIR/${patch}.py"; then
            echo "  mlx-embeddings already ships ${patch}.py; leaving it alone"
        else
            cp "patches/mlx_embeddings_${patch}.py" "$MLX_MODELS_DIR/${patch}.py"
            echo "  ${patch} architecture installed (legacy models only)"
        fi
    done
else
    echo "  Warning: Could not find mlx-embeddings models directory"
fi

# Create data directory
mkdir -p data

# Download models for the selected profile (CODE_RAG_PROFILE, default: high).
# Set CODE_RAG_PROFILE=medium|low for smaller machines, or legacy for the old defaults.
echo ""
echo "Model profile: ${CODE_RAG_PROFILE:-high}   (change with CODE_RAG_PROFILE=max|high|medium|low|legacy)"
echo ""
"$SCRIPT_DIR/download-embed-model.sh"

echo ""
"$SCRIPT_DIR/download-description-model.sh" || echo "  Description model download failed; descriptions will be disabled until ./download-description-model.sh succeeds."

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
    import model_config
    print(f"  Testing embedding with {model_config.summary()['embed_model_key']}...")
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
python3 model_config.py
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
echo "Changing models (see .env.example):"
echo ""
echo "  Whole profile:        export CODE_RAG_PROFILE=medium      # max|high|medium|low|legacy"
echo "  Embedding only:       ./download-embed-model.sh qwen3-embed-0.6b"
echo "                        export CODE_RAG_EMBED_MODEL=qwen3-embed-0.6b"
echo "  Descriptions only:    ./download-description-model.sh qwen3-4b-2507"
echo "                        export CODE_RAG_DESCRIPTION_MODEL_KEY=qwen3-4b-2507"
echo "  List everything:      venv/bin/python model_config.py --list"
echo ""
echo "  Note: switching embedding models requires re-indexing (./index.sh --force)."
echo ""
echo "======================================================================"
