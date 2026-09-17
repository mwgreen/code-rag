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

**Important:** `transformers` must be 5.x. `mlx-embeddings>=0.1.0` and `mlx-lm>=0.31` both require it (older code-rag installs pinned `<5.0` for mlx-embeddings 0.0.x; upgrading the venv means upgrading transformers too).

### 3. Legacy Architecture Patches (automatic)

The default Qwen3-Embedding models are supported natively by `mlx-embeddings>=0.1.0`. The legacy
models (Qodo-Embed = Qwen2, SFR-Embedding-Code = CodexEmbed2B) need the implementations under
`patches/`; `rag_milvus` registers them into `sys.modules` at load time, so nothing is copied into
site-packages and pip upgrades cannot break them.

### 4. Models

Pick a hardware profile (default `high`, for a 48 GB machine; see [USAGE.md](USAGE.md#model-configuration)),
then download both models. Embedding models are quantized to Q8 into the gitignored `models/` directory;
the description model is cached in `~/.cache/huggingface/hub`.

```bash
export CODE_RAG_PROFILE=high       # or max | medium | low | legacy
./download-embed-model.sh          # e.g. models/qwen3-embed-4b-mlx-q8 (~4.5 GB)
./download-description-model.sh    # e.g. mlx-community/gemma-4-e4b-it-OptiQ-4bit (~7.5 GB)
venv/bin/python model_config.py    # confirm what resolved
```

Both downloads are one-time and need network access; at runtime the server forces `HF_HUB_OFFLINE=1`.
Moving from another machine: copying `models/` over works, but per-project indexes under
`{project}/.code-rag/` must be rebuilt if the embedding model changed (`./index.sh --force`).

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
├── model_config.py                 # Model registry + profiles
├── download-embed-model.sh         # Download + Q8 quantize an embedding model
├── download-description-model.sh   # Pre-cache a description model
├── models/                         # Embedding models (gitignored)
├── venv/                           # Python venv (gitignored)
├── node_modules/                   # Node.js deps (gitignored)
├── README.md                       # Architecture overview
├── USAGE.md                        # Usage guide
└── INSTALLATION.md                 # This file

~/.code-rag/                        # Server runtime (global)
├── server.pid                      # PID file
└── server.log                      # Server log

{project}/.code-rag/                # Per-project index
├── milvus.db                       # Milvus Lite vector DB
└── fts.db                          # FTS5 keyword index
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
venv/bin/python model_config.py --list   # shows which models are downloaded and which resolved
./download-embed-model.sh                # fetch the resolved embedding model
./download-description-model.sh          # fetch the resolved description model
```

If descriptions silently stop appearing, check `~/.code-rag/server.log` for "Failed to load description
model": usually the model is not cached or `mlx-lm` is older than the model needs.

### transformers Version Error

`mlx-embeddings>=0.1.0` and `mlx-lm>=0.31` need transformers 5.x. If an older venv still has the 4.x pin:

```bash
venv/bin/pip install --upgrade "transformers[sentencepiece]>=5.0.0" "mlx-embeddings>=0.1.0" "mlx-lm>=0.31.3" "mlx>=0.32.2"
```

(Only the `legacy` profile on an un-upgraded venv still works with transformers 4.x and mlx-embeddings 0.0.5.)

### Port Already in Use

```bash
lsof -i :7101
# Kill stale process or set CODE_RAG_PORT=7102 in environment
```
