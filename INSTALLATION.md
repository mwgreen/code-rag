# Code-RAG Installation Guide

## Prerequisites

- **macOS 15+** with **Apple Silicon** (M1/M2/M3/M4)
- **Python 3.12+**
- **Node.js 18+** (for code-chunk AST-aware chunking)

```bash
python3 --version   # 3.12+
node --version      # 18+
uname -m            # arm64
```

## Quick Start (Automated)

```bash
cd code-rag

# 1. Run setup (creates venv, installs deps, checks model)
./setup.sh

# 2. Start the server
./code-rag-server.sh start

# 3. Verify
curl http://127.0.0.1:7101/health
# {"status": "ok"}
```

## Manual Installation

### 1. Node.js Dependencies

```bash
cd code-rag
npm install
```

Installs `code-chunk` (from supermemory) for AST-aware semantic chunking.

### 2. Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Key dependencies: `pymilvus[milvus-lite]`, `mlx-embeddings`, `mlx`, `starlette`, `uvicorn`, `mcp`.

**Important:** `transformers` must be version 4.x (not 5.x) for mlx-embeddings compatibility.

### 3. Qwen2 Architecture Patch

mlx-embeddings doesn't natively support Qwen2. Install the patch:

```bash
MLX_MODELS_DIR=$(python3 -c "import mlx_embeddings.models; import os; print(os.path.dirname(mlx_embeddings.models.__file__))")
cp patches/mlx_embeddings_qwen2.py "$MLX_MODELS_DIR/qwen2.py"
```

### 4. Embedding Model

The model is Qodo-Embed-1-1.5B quantized to MLX Q8 (~1.6GB). It's too large for git and lives in the gitignored `models/` directory:

```bash
mkdir -p models/qodo-embed-1-1.5b-mlx-q8
# Place model files here: model.safetensors, config.json, tokenizer.json, tokenizer_config.json
```

Once placed, the system runs fully offline (`HF_HUB_OFFLINE=1` is auto-set).

## Claude Code Configuration

### 1. Project `.mcp.json`

Add to your project root's `.mcp.json`:

```json
{
  "mcpServers": {
    "code-rag": {
      "type": "http",
      "url": "http://127.0.0.1:7101/mcp/",
      "headers": {
        "X-Project-Root": "/absolute/path/to/your/project"
      }
    }
  }
}
```

### 2. SessionStart Hook (Auto-Start)

Add to `~/.claude/settings.json` so the server starts automatically:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/absolute/path/to/code-rag/code-rag-server.sh start",
            "timeout": 45000
          }
        ]
      }
    ]
  }
}
```

### 3. Initial Index

```bash
# Stop server (CLI needs exclusive DB access)
./code-rag-server.sh stop

# Index your project
./index.sh /path/to/your/project

# Restart server
./code-rag-server.sh start
```

Add `.code-rag/` to your project's `.gitignore`.

## Verification

```bash
# Server running?
./code-rag-server.sh status

# Health check
curl http://127.0.0.1:7101/health

# Test search via MCP
curl -s -X POST http://127.0.0.1:7101/mcp/ \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "X-Project-Root: /path/to/your/project" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_stats","arguments":{}}}' \
  | python3 -m json.tool
```

## Directory Structure

```
code-rag/                           # Tool directory (lives in your dev env repo)
├── http_server.py                  # Persistent HTTP server
├── mcp_server.py                   # Stdio transport (debug/backward compat)
├── tools.py                        # Shared tool definitions + project context
├── rag_milvus.py                   # Core RAG engine
├── chunking.py                     # Chunking dispatcher
├── codechunk_wrapper.py            # code-chunk Node.js wrapper
├── chunker.mjs / chunker_batch.mjs # Node.js chunkers
├── ast_chunking.py                 # Tree-sitter fallback
├── index_codebase.py               # CLI indexer
├── index.sh                        # CLI wrapper script
├── code-rag-server.sh              # Server launcher
├── setup.sh                        # Automated installer
├── requirements.txt                # Python deps
├── package.json                    # Node.js deps
├── patches/                        # MLX architecture patches
├── models/                         # Embedding model (gitignored, ~1.6GB)
├── venv/                           # Python venv (gitignored)
├── node_modules/                   # Node.js deps (gitignored)
├── README.md                       # Architecture overview
├── USAGE.md                        # Usage guide
└── INSTALLATION.md                 # This file

~/.code-rag/                        # Server runtime (global)
├── server.pid                      # PID file
└── server.log                      # Server log

{project}/.code-rag/                # Per-project index
└── milvus.db                       # Milvus Lite vector DB
```

## Troubleshooting

### code-chunk Not Working

```bash
ls node_modules/code-chunk/    # Should exist
npm install                    # If missing
```

Falls back to tree-sitter, then regex chunking if code-chunk fails.

### Model Not Found

```bash
ls models/qodo-embed-1-1.5b-mlx-q8/
# Should contain: model.safetensors, config.json, tokenizer.json, tokenizer_config.json
```

### transformers Version Error

```bash
pip install "transformers<5.0"
```

### Port Already in Use

```bash
lsof -i :7101
# Kill stale process or set CODE_RAG_PORT=7102 in environment
```
