# Code-RAG: Semantic Code Search for Apple Silicon

Semantic code search using Qodo-Embed-1-1.5B embeddings with MLX (Apple Silicon GPU) and Milvus Lite vector database.

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
- **Shared model** — the 1.6GB MLX embedding model loads once, serves all projects
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

2. **Embedding**: Converts chunks to 1536-dim vectors using Qodo-Embed-1-1.5B
   - Q8 quantized (~1.6GB) running on Apple Silicon GPU via MLX
   - Runs fully offline after model setup

3. **Indexing**: Stores vectors in Milvus Lite (embedded SQLite-based DB)
   - Incremental updates (only changed files via hash detection)
   - Automatic cleanup of deleted/moved files
   - Per-project DB files at `{project}/.code-rag/milvus.db`

4. **Search**: Vector similarity search with cosine distance
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
| `patches/` | MLX architecture patches (Qwen2 support) |

### Runtime Locations
| Path | Purpose |
|------|---------|
| `~/.code-rag/server.pid` | Server PID file (global, one server process) |
| `~/.code-rag/server.log` | Server log file |
| `{project}/.code-rag/milvus.db` | Project's vector index (per-project) |
| `code-rag/models/` | MLX embedding model (gitignored, ~1.6GB) |
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
- **Qodo-Embed**: Apache 2.0 (Qodo)
- **code-chunk**: MIT (supermemory)
- **transformers**: Apache 2.0 (Hugging Face)
