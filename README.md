# Code-RAG: Semantic Code Search for Apple Silicon

Semantic code search using local MLX embeddings (default: Qwen3-Embedding-4B) on the Apple Silicon GPU, Milvus Lite as the vector database, and a local LLM (default: Gemma 4 E4B) that writes one-sentence descriptions of each code chunk. Models are selectable by hardware profile; see [Model Configuration](USAGE.md#model-configuration).

## What Is This?

A persistent HTTP server that provides semantic code search via MCP (Model Context Protocol). Ask natural language questions about your codebase:
- "Find JWT authentication implementation"
- "Show examples of REST API controllers"
- "Find database migration configuration"

Instead of grep/text matching, it understands **semantic meaning** and finds relevant code even when exact keywords don't match.

## Architecture Overview

```
Claude Code ──HTTP/MCP──> code-rag server (persistent, port 7101)
                              │
                              ├── MLX model (loaded once, shared across projects)
                              │
                              ├── Project A: /path/to/project-a/.code-rag/milvus.db
                              └── Project B: /path/to/project-b/.code-rag/milvus.db
```

**Key design decisions:**
- **Persistent HTTP server** — starts once, stays running across Claude Code sessions
- **Per-project indexes** — each project stores its DB at `{project}/.code-rag/milvus.db`
- **Shared model** — the MLX embedding model loads once, serves all projects
- **Concurrent access** — multiple Claude Code sessions can search/index simultaneously
- **Project identification** — `X-Project-Root` HTTP header tells the server which project's DB to use

## Quick Start

```bash
cd code-rag

# 1. Setup (Python venv + Node.js deps + model check)
./setup.sh

# 2. Start the persistent server
./code-rag-server.sh start

# 3. Index your codebase (via CLI, server must be stopped)
./code-rag-server.sh stop
./index.sh /path/to/your/project
./code-rag-server.sh start

# 4. Configure Claude Code (see USAGE.md for .mcp.json setup)
```

## Prerequisites

- macOS 15+ with Apple Silicon (M1/M2/M3/M4)
- Python 3.12+
- Node.js 18+ (for code-chunk semantic chunking)

## Documentation

- **USAGE.md** — How to configure, index, search, and troubleshoot
- **INSTALLATION.md** — Complete setup guide (automated & manual)
- This **README.md** — Architecture overview and file reference

## How It Works

1. **Chunking**: Splits code into semantic chunks (classes, methods, functions)
   - Uses **code-chunk** (Node.js, from supermemory) for AST-aware context
   - Each chunk includes scope chain, imports, and method signatures
   - Falls back to tree-sitter, then regex for unsupported files

2. **Description** (optional, on by default): A local LLM writes a one-sentence summary of each
   code chunk, which is prepended to the code before embedding to bridge the NL-to-code vocabulary gap
   - Default: Gemma 4 E4B (`mlx-community/gemma-4-e4b-it-OptiQ-4bit`) via mlx-lm
   - Cached in SQLite by content hash, so each chunk is described once

3. **Embedding**: Converts chunks (description + code) to vectors with an MLX embedding model
   - Default: Qwen3-Embedding-4B, Q8 quantized (~4.5GB, 2560 dims); smaller/legacy models selectable
   - Queries get a task instruction prefix; documents are embedded raw
   - Runs fully offline after model setup

4. **Indexing**: Stores vectors in Milvus Lite (embedded SQLite-based DB)
   - Incremental updates (only changed files via hash detection)
   - Automatic cleanup of deleted/moved files
   - Per-project DB files at `{project}/.code-rag/milvus.db`

5. **Search**: Vector similarity search with cosine distance
   - Natural language queries via MCP tools
   - Filter by language or type
   - Sub-second results

## File Reference

### Server & Transport
| File | Purpose |
|------|---------|
| `http_server.py` | Persistent HTTP server (Starlette + uvicorn + StreamableHTTPSessionManager) |
| `mcp_server.py` | Stdio MCP transport (for debugging/backward compat) |
| `tools.py` | Shared MCP tool definitions and project context (ContextVar) |
| `code-rag-server.sh` | Server launcher (start/stop/status/restart) |

### Core RAG Engine
| File | Purpose |
|------|---------|
| `rag_milvus.py` | Embedding, search, indexing, client management, concurrency |
| `chunking.py` | Chunking dispatcher (code-chunk -> tree-sitter -> regex) |
| `codechunk_wrapper.py` | Python wrapper for code-chunk Node.js process |
| `chunker.mjs` | Node.js single-file chunker |
| `chunker_batch.mjs` | Node.js batch chunker (NDJSON streaming) |
| `ast_chunking.py` | Tree-sitter chunking fallback |

### CLI Tools
| File | Purpose |
|------|---------|
| `index_codebase.py` | CLI tool for batch indexing (progress bars, full/incremental) |
| `index.sh` | Shell wrapper for index_codebase.py |
| `setup.sh` | Automated installation script |

### Configuration
| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies |
| `package.json` | Node.js dependencies (code-chunk) |
| `.ragignore` | Per-project directory exclusion list (placed in project root) |
| `model_config.py` | Model registry, hardware profiles, and env-var resolution |
| `.env.example` | Documented model/profile environment variables |
| `patches/` | MLX architecture patches for the legacy embedding models (Qwen2, CodexEmbed2B) |

### Runtime Locations
| Path | Purpose |
|------|---------|
| `~/.code-rag/server.pid` | Server PID file (global, one server process) |
| `~/.code-rag/server.log` | Server log file |
| `{project}/.code-rag/milvus.db` | Project's vector index (per-project) |
| `code-rag/models/` | MLX embedding models (gitignored, 0.7-4.5GB each) |
| `~/.cache/huggingface/hub/` | Description model cache (pre-downloaded by `download-description-model.sh`) |
| `code-rag/venv/` | Python virtual environment |

## Concurrency Model

The HTTP server uses asyncio primitives for safe concurrent access:
- **`asyncio.Semaphore(1)`** — serializes MLX embedding (single GPU thread)
- **`asyncio.Lock()`** — serializes Milvus writes (SQLite-backed)
- **Reads are lock-free** — multiple searches run in parallel

Persistent Milvus clients are cached per `db_path` and reused across requests.

## License

All dependencies are open source:
- **MLX**: Apache 2.0 (Apple)
- **Milvus**: Apache 2.0 (Linux Foundation)
- **Qwen3-Embedding**: Apache 2.0 (Alibaba)
- **Gemma 4**: Apache 2.0 (Google)
- Legacy models: SFR-Embedding-Code-2B_R is CC-BY-NC-4.0, Qodo-Embed is OpenRAIL++-M, Gemma 3 is under the Gemma terms
- **code-chunk**: MIT (supermemory)
- **transformers**: Apache 2.0 (Hugging Face)
