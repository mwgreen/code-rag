# Code-RAG: Architecture & Design Document

## 1. Executive Summary

Code-RAG is a semantic code search system that enables natural language queries against codebases. It converts source code into vector embeddings using a local AI model and stores them in an embedded vector database, both running entirely on the developer's machine. The system integrates with Claude Code via the Model Context Protocol (MCP) as a persistent HTTP server.

**Core design principles:**

- **Fully local execution** -- all code content, embeddings, and indexes remain on the developer's machine. After initial model download, zero network access is required at runtime.
- **Apple Silicon native** -- leverages MLX for GPU-accelerated embedding generation on M-series chips.
- **Persistent server** -- a single long-lived process serves all projects, sharing the expensive embedding model across sessions.
- **Incremental indexing** -- file-level SHA-256 hashing avoids redundant re-embedding, and a file watcher provides automatic live reindexing.

---

## 2. System Architecture

### 2.1 High-Level Overview

```
                                ┌─────────────────────────────────────────────┐
                                │         Developer's Machine (localhost)      │
                                │                                             │
  ┌──────────────┐   HTTP/MCP   │   ┌─────────────────────────────────────┐   │
  │ Claude Code  │─────────────────>│  code-rag HTTP Server (port 7101)  │   │
  │  (MCP Client)│  X-Project-  │   │  Starlette + uvicorn               │   │
  └──────────────┘  Root header │   └──────────┬──────────────────────────┘   │
                                │              │                              │
                                │      ┌───────┴────────┐                     │
                                │      │                │                     │
                                │   ┌──▼──────────┐  ┌──▼──────────────────┐  │
                                │   │ MLX Model   │  │ Milvus Lite (SQLite)│  │
                                │   │ (Qodo-Embed)│  │ Per-project DB      │  │
                                │   │ Apple GPU   │  │ .code-rag/milvus.db │  │
                                │   └─────────────┘  └─────────────────────┘  │
                                │                                             │
                                │   ┌─────────────────────────────────────┐   │
                                │   │ File Watcher (FSEvents via watchdog)│   │
                                │   │ Detects changes → auto-reindex      │   │
                                │   └─────────────────────────────────────┘   │
                                │                                             │
                                └─────────────────────────────────────────────┘
                                         Nothing leaves this box.
```

### 2.2 Component Architecture

```
┌───────────────────────────────────────────────────────────┐
│                     Transport Layer                        │
│   http_server.py (Starlette/uvicorn, StreamableHTTP)      │
│   mcp_server.py  (stdio, for debugging)                   │
│   tools.py       (shared MCP tool definitions, ContextVar)│
└────────────────────────┬──────────────────────────────────┘
                         │
┌────────────────────────▼──────────────────────────────────┐
│                     RAG Engine                             │
│   rag_milvus.py                                           │
│   ├── Embedding:  MLX + Qodo-Embed-1-1.5B (1536-dim)     │
│   ├── Storage:    Milvus Lite (SQLite-backed)             │
│   ├── Hybrid:     Vector (cosine) + FTS5 (BM25) via RRF  │
│   ├── Indexing:   Incremental (SHA-256 hash tracking)     │
│   └── Concurrency: Semaphore(1) + Lock() + lock-free reads│
└────────────────────────┬──────────────────────────────────┘
                         │
┌────────────────────────▼──────────────────────────────────┐
│                   Chunking Pipeline                        │
│   chunking.py (dispatcher)                                │
│   ├── Tier 1: code-chunk (Node.js, AST + context)         │
│   │           codechunk_wrapper.py → chunker.mjs           │
│   ├── Tier 2: tree-sitter AST (ast_chunking.py)           │
│   └── Tier 3: Language-specific regex / character split    │
└───────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼──────────────────────────────────┐
│                   File Watcher                             │
│   file_watcher.py                                         │
│   ├── FSEvents (macOS native) via watchdog                │
│   ├── Debounced batch processing (2s default)             │
│   ├── Git-aware settling (defers during git operations)   │
│   └── Per-project ProjectWatcher instances                │
└───────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow

### 3.1 Indexing Pipeline

```
Source File
    │
    ▼
┌──────────────────┐
│ File Discovery    │  rglob with extension filter, exclusion list,
│                   │  .ragignore, size cap (1MB), JAXB detection
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Change Detection  │  SHA-256 hash comparison against stored hash
│                   │  Skip if unchanged (incremental mode)
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Chunking          │  code-chunk → tree-sitter → regex → character split
│                   │  Respects semantic boundaries (classes, methods)
│                   │  Adds scope chain, imports, signatures (code-chunk)
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Embedding         │  Qodo-Embed-1-1.5B via MLX (Apple Silicon GPU)
│                   │  Input: text chunk → Output: 1536-dim float vector
│                   │  ~460ms per chunk, serialized via Semaphore(1)
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Storage           │  Milvus Lite insert (SQLite-backed)
│                   │  Fields: id, vector, document, doc_id, path,
│                   │          language, type, content_hash
│                   │  Serialized via Lock()
└──────────────────┘
```

### 3.2 Search Pipeline

```
Natural Language Query (e.g., "JWT authentication logic")
    │
    ▼
┌──────────────────┐
│ Query Embedding   │  Same Qodo-Embed-1-1.5B model
│                   │  Query → 1536-dim vector
│                   │  Serialized via Semaphore(1)
└────────┬─────────┘
         │
    ┌────┴────────────────┐
    ▼                     ▼
┌──────────────┐  ┌──────────────────┐
│ Vector Search │  │ Keyword Search    │  SQLite FTS5
│ Milvus cosine │  │ BM25 ranking      │  Full-text match
│ ~50-80ms      │  │ ~5ms              │  on code content
└──────┬───────┘  └────────┬─────────┘
       │                   │
       └───────┬───────────┘
               ▼
┌──────────────────┐
│ RRF Merge         │  Reciprocal Rank Fusion (k=60)
│ + Deduplication   │  Content fingerprint dedup
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Result Formatting │  File path, line range, language, type
│                   │  Relevance score (1 - cosine distance)
└──────────────────┘
```

---

## 4. Security Analysis

### 4.1 Data Locality: Nothing Leaves the Machine

The single most important security property of Code-RAG is that **no source code, embeddings, queries, or index data ever leaves the developer's machine** during normal operation.

| Component | Location | Network Access |
|-----------|----------|----------------|
| Source code (original files) | Local filesystem | None |
| Embedding model (Qodo-Embed-1-1.5B Q8) | `code-rag/models/` (~1.6 GB) | None at runtime |
| Embedding computation | Apple Silicon GPU via MLX | None |
| Vector database (Milvus Lite) | `{project}/.code-rag/milvus.db` | None |
| HTTP server | `127.0.0.1:7101` (localhost only) | Loopback only |
| Search queries | In-process embedding + DB lookup | None |
| File watcher | macOS FSEvents (kernel-level) | None |

**The only network access occurs during initial setup:**
- `setup.sh` downloads the Qodo-Embed-1-1.5B model from HuggingFace (~5.8 GB, quantized to ~1.6 GB).
- `npm install` downloads the `code-chunk` Node.js package.
- After setup, the system auto-enables `HF_HUB_OFFLINE=1` to prevent any HuggingFace network calls.

**No API keys, accounts, subscriptions, or cloud services are required at runtime.** The model is Apache 2.0 licensed and runs entirely on-device.

### 4.2 Network Binding

The HTTP server is explicitly bound to `127.0.0.1` (not `0.0.0.0`), meaning it only accepts connections from the local machine:

```python
# http_server.py:34
HOST = os.getenv("CODE_RAG_HOST", "127.0.0.1")
```

This prevents any remote access, even on the local network. The binding address is configurable via the `CODE_RAG_HOST` environment variable -- changing this to `0.0.0.0` would expose the server to the network and is **not recommended**.

### 4.3 Authentication

The HTTP server does not implement authentication. Any process running on localhost can send requests to port 7101 and:

- Read any indexed code content via search queries.
- Trigger indexing of arbitrary file paths.
- Read index statistics.

**Assessment:** This is an acceptable trade-off for a single-user developer tool running on localhost. The threat model assumes that any process with localhost access is already within the user's trust boundary (any such process could read the source files directly from disk). However, users should be aware that:

- Other local applications or browser-based exploits (SSRF) could theoretically access the endpoint.
- If the `CODE_RAG_HOST` is changed to a non-loopback address, authentication would become critical.

**Recommendation for hardened environments:** If the server were ever exposed beyond localhost, adding bearer token authentication or mTLS would be necessary.

### 4.4 Input Handling

**Milvus filter expressions** use Python string interpolation to construct queries:

```python
# rag_milvus.py:272-274
filters.append(f'type == "{type_filter}"')
filters.append(f'language == "{language_filter}"')
```

The `language` parameter is constrained by an enum in the MCP tool schema, and `type_filter` is set by application code (never directly from user input). File paths used in filter expressions come from `Path().absolute()` or the local filesystem. One path escaping site (`_get_indexed_paths_under`) properly escapes double quotes; the remaining call sites (`delete_by_path`, `file_needs_indexing`, `add_file`) do not, but these only receive paths from the local filesystem where double-quote characters in file names are extremely rare.

**Assessment:** Low risk. The MCP layer validates inputs, and all filter values are either constrained enums or derived from local file paths. This is not an injection surface in practice.

### 4.5 File System Access

The `index_file` MCP tool accepts an arbitrary file path and reads its contents for embedding. Combined with the lack of authentication:

- Any local process could instruct the server to read and index an arbitrary file.
- The indexed content would then be searchable.

**Assessment:** Low risk. Any local process with socket access already has the same filesystem permissions as the user. The tool does not expose file contents directly -- it stores embedding vectors and code chunks in the Milvus database, which is also on the local filesystem.

### 4.6 Model Supply Chain

The embedding model is downloaded from HuggingFace during setup. The model files are:

- **Source:** `Qodo/Qodo-Embed-1-1.5B` on HuggingFace Hub
- **License:** Apache 2.0
- **Format:** SafeTensors (a safe serialization format that prevents arbitrary code execution, unlike pickle)
- **Quantized locally** to Q8 via `mlx_embeddings.utils.convert`

SafeTensors is specifically designed to be a safe format for model weights -- it does not support arbitrary code execution, unlike Python pickle files. This is a strong positive for supply chain security.

### 4.7 Dependency Surface

| Category | Dependencies | Risk Notes |
|----------|-------------|------------|
| Vector DB | pymilvus, milvus-lite | Apache 2.0, Linux Foundation project |
| ML Runtime | mlx, mlx-embeddings, transformers | Apple (MLX), HuggingFace |
| Code Parsing | tree-sitter, code-chunk | Mature C library (tree-sitter), Node.js AST library |
| Web Framework | starlette, uvicorn | Well-established ASGI ecosystem |
| File Watching | watchdog | Established Python library, uses macOS FSEvents |
| Protocol | mcp | Anthropic's Model Context Protocol SDK |

All dependencies are open-source with permissive licenses. The `patches/mlx_embeddings_qwen2.py` file is a custom Qwen2 architecture implementation that is copied into the mlx-embeddings package directory during setup.

### 4.8 Summary of Potential Concerns

| Concern | Severity | Mitigation |
|---------|----------|------------|
| No authentication on localhost HTTP | Low | Acceptable for single-user localhost tool. Would need auth if exposed to network. |
| String interpolation in Milvus filters | Low | Inputs are constrained enums or local paths. Not exploitable in practice. |
| Model download over network (setup only) | Low | One-time, from HuggingFace. SafeTensors format prevents code execution. `HF_HUB_OFFLINE=1` enforced at runtime. |
| `CODE_RAG_HOST` misconfiguration | Medium | Default is safe (`127.0.0.1`). Documentation should warn against changing to `0.0.0.0`. |
| Arbitrary file path in `index_file` tool | Low | Localhost-only; any local process already has equivalent filesystem access. |

---

## 5. Embedding Model

### 5.1 Current Model: Qodo-Embed-1-1.5B

| Property | Value |
|----------|-------|
| Model | Qodo-Embed-1-1.5B |
| Architecture | Qwen2 (transformer-based encoder) |
| Embedding Dimensions | 1536 |
| Full Precision Size | ~5.8 GB |
| Quantized Size (Q8) | ~1.6 GB |
| Quantization | 8-bit, group_size=64 |
| Framework | MLX (Apple Silicon Metal GPU) |
| License | Apache 2.0 |
| Specialization | Code and text embedding |

The model is loaded once at server startup (~3-4 seconds) and shared across all projects. It runs entirely on the Apple Silicon GPU via MLX's Metal backend. Embedding generation is serialized via an `asyncio.Semaphore(1)` to prevent GPU memory contention.

GPU memory is periodically cleared (`mx.clear_cache()`) every 10 files during batch indexing and every 20 files during watcher-triggered reindexing to prevent memory pressure.

### 5.2 Architecture Patch

The Qwen2 architecture is not natively supported by `mlx-embeddings`. Code-RAG includes a custom implementation at `patches/mlx_embeddings_qwen2.py` that is installed into the mlx-embeddings package during setup. Key differences from Qwen3:

- Attention has bias on Q/K/V projections (not O projection)
- No QK normalization
- Last-token pooling with L2 normalization for embedding generation

### 5.3 Alternative Embedding Providers

The current implementation uses a fully local embedding model. This could be replaced with cloud-based embedding APIs if the user is willing to send code content off-machine. Two notable alternatives:

#### OpenAI Embeddings

| Property | Value |
|----------|-------|
| Model | `text-embedding-3-large` |
| Dimensions | 3072 (or configurable down to 256) |
| Pricing | ~$0.13 per 1M tokens |
| Latency | ~200-500ms per request (network-dependent) |

**Integration impact:** Replace `embed_texts()` in `rag_milvus.py` with OpenAI API calls. Requires an `OPENAI_API_KEY` environment variable. Batch embedding is supported. Would need to update `_EMBED_DIM` to match the chosen dimension. Removes the MLX dependency entirely, making the system cross-platform.

#### Voyage AI Embeddings

| Property | Value |
|----------|-------|
| Model | `voyage-code-3` |
| Dimensions | 1024 |
| Pricing | ~$0.06 per 1M tokens |
| Specialization | Purpose-built for code search and retrieval |

**Integration impact:** Similar to OpenAI -- replace `embed_texts()` with Voyage API calls. Requires a `VOYAGE_API_KEY`. Voyage's code-specific model may provide superior code search relevance. The lower dimensionality (1024 vs 1536) would reduce storage requirements.

#### Trade-offs

| Factor | Local (Qodo-Embed) | Cloud (OpenAI/Voyage) |
|--------|--------------------|-----------------------|
| **Privacy** | Code never leaves machine | Code sent to external API |
| **Cost** | Free (after model download) | Per-token pricing |
| **Latency** | ~460ms/chunk (GPU-bound) | ~200-500ms/chunk (network-bound) |
| **Platform** | Apple Silicon only | Any platform |
| **Offline** | Fully offline capable | Requires internet |
| **Quality** | State-of-the-art code embeddings | Comparable or better |
| **Setup** | ~1.6 GB model download | API key only |

**Recommendation:** The local Qodo-Embed model is the correct default for privacy-sensitive codebases (proprietary code, enterprise environments). Cloud providers are a viable option for open-source projects or environments where code sharing is acceptable, and would make the system cross-platform.

To support provider switching, the embedding layer could be abstracted behind an interface:

```python
class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dimensions(self) -> int: ...
```

---

## 6. Vector Database

### 6.1 Current: Milvus Lite

| Property | Value |
|----------|-------|
| Engine | Milvus Lite (embedded mode) |
| Storage Backend | SQLite |
| Location | `{project}/.code-rag/milvus.db` |
| Distance Metric | Cosine |
| Max Queryable Rows (offset) | 16,384 |
| License | Apache 2.0 |

Milvus Lite is an embedded vector database that requires no separate server process. It stores all data in a single SQLite file per project. It supports vector similarity search, metadata filtering, and standard CRUD operations.

**Concurrency model:**
- Reads are lock-free (multiple concurrent searches)
- Writes are serialized via `asyncio.Lock()` (SQLite limitation)
- Persistent client caching per `db_path` in HTTP server mode
- Ephemeral connections for CLI and stdio modes

**Schema:**

| Field | Type | Purpose |
|-------|------|---------|
| `id` | uint64 | Hash of doc_id |
| `vector` | FloatVector[1536] | Embedding vector |
| `document` | String | Code chunk text |
| `doc_id` | String | `{filepath}::{chunk_index}` |
| `path` | String | Absolute file path |
| `language` | String | Detected language |
| `type` | String | `code`, `documentation`, or `config` |
| `content_hash` | String | SHA-256 of source file |

### 6.2 Alternative Vector Databases

Several alternative vector databases could replace Milvus Lite, each with different trade-offs:

#### Local/Embedded Options

| Database | Storage | Setup Complexity | Notes |
|----------|---------|-----------------|-------|
| **PostgreSQL + pgvector** | Server-based (local) | Medium | Full SQL capabilities, HNSW and IVFFlat indexes. Excellent if Postgres is already in the dev stack. Requires running a Postgres instance. |
| **SQLite + sqlite-vss** | Embedded file | Low | Stays closest to current architecture. Faiss-backed vector search in a SQLite extension. No separate process needed. |
| **ChromaDB** | Embedded or client/server | Low | Python-native, simple API. Originally designed for RAG workloads. Can run embedded (in-process) or as a server. |
| **LanceDB** | Embedded file | Low | Columnar format (Lance), very fast for analytical queries. Good compression. No server process needed. |
| **Qdrant** | Server-based (local) | Medium | Rich filtering, payload indexing. Can run as a Docker container or embedded via `qdrant-client[local]`. |
| **FAISS** | In-memory / file | Low | Meta's library. Extremely fast but no built-in metadata filtering -- would need a sidecar store. Best for pure vector search. |

#### Cloud Options (if data leaving the machine is acceptable)

| Database | Hosting | Notes |
|----------|---------|-------|
| **Pinecone** | Managed cloud | Serverless option, automatic scaling. Per-vector pricing. |
| **Weaviate Cloud** | Managed cloud | GraphQL API, hybrid search (vector + keyword). |
| **PostgreSQL (RDS/Cloud SQL) + pgvector** | Managed cloud | Familiar SQL interface, managed Postgres. |

#### Recommendation

For the use case of per-developer, per-project code search:

- **Milvus Lite** (current) is a solid choice -- zero-config, embedded, sufficient performance.
- **PostgreSQL + pgvector** is the strongest alternative for users who already run Postgres locally. It provides mature indexing (HNSW), full SQL filtering, and a well-understood operational model. Stays local.
- **ChromaDB** or **LanceDB** would be the simplest migration paths if Milvus Lite limitations are encountered (e.g., the 16,384 offset query cap).
- **SQLite + sqlite-vss** would preserve the single-file-per-project architecture while potentially improving query capabilities.

All local options preserve the core security property: **nothing leaves the machine.**

---

## 7. Chunking Pipeline

### 7.1 Multi-Tier Strategy

Code-RAG uses a three-tier fallback strategy for splitting source files into semantically meaningful chunks:

```
Tier 1: code-chunk (Node.js)      ← Best quality, AST + context
  │ Supported: Java, Python, TypeScript, JavaScript, Rust, Go
  │ Features: scope chain, imports, sibling signatures
  │ Falls through on: unsupported language, parse failure, timeout
  ▼
Tier 2: tree-sitter AST            ← Good quality, pure AST
  │ Supported: Java, Python, TypeScript, JavaScript
  │ Features: semantic node extraction (classes, methods, functions)
  │ Falls through on: unsupported language, parse failure
  ▼
Tier 3: Language-specific regex / character split  ← Fallback
  │ Java: class/interface/enum boundary detection
  │ JavaScript: Ext.define/function boundary detection
  │ TypeScript: interface/type/class/enum detection
  │ YAML: document separator (---) splitting
  │ Default: character-based splitting (MAX_CHUNK_SIZE=2000)
  ▼
  Chunks (with metadata: path, language, type, start_line, end_line)
```

### 7.2 code-chunk (Tier 1)

The highest-quality chunker, provided by the `code-chunk` npm package. It performs full AST parsing and produces **contextualized** chunks that include:

- **Scope chain:** The enclosing class/function/module hierarchy
- **Imports:** Relevant import statements for the chunk
- **Sibling signatures:** Method signatures of sibling methods in the same class

This context is prepended to each chunk, giving the embedding model a richer understanding of the code's role within the codebase. The chunker runs as a Node.js subprocess, with two modes:

- **Single file** (`chunker.mjs`): One Node.js process per file. Used for MCP tool calls.
- **Batch** (`chunker_batch.mjs`): Long-lived Node.js process with NDJSON streaming. Used for bulk indexing to avoid cold-start overhead.

Configuration: `maxChunkSize=2000`, `contextMode='full'`, `siblingDetail='signatures'`.

### 7.3 tree-sitter AST (Tier 2)

A pure Python fallback using tree-sitter grammars. Extracts semantic units:

- **Java:** `method_declaration`, `class_declaration`, `interface_declaration`, `enum_declaration`
- **Python:** `function_definition`, `class_definition`
- **TypeScript/JavaScript:** `function_declaration`, `method_definition`, `class_declaration`, `interface_declaration`, `type_alias_declaration`, `enum_declaration`

Large AST nodes (>2000 chars) are recursively split by extracting child nodes.

### 7.4 Chunk Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_CHUNK_SIZE` | 2000 characters | Maximum chunk size before splitting |
| `MIN_CHUNK_SIZE` | 100 characters | Minimum chunk size (smaller chunks discarded) |

---

## 8. Server Architecture

### 8.1 HTTP Server

The primary transport is a persistent HTTP server built on Starlette/uvicorn:

```
┌─────────────────────────────────────────────────┐
│               Starlette Application              │
│                                                  │
│  GET /health       → Health check + watcher stats│
│  POST /mcp/*       → MCP protocol (via manager)  │
│                                                  │
│  ProjectMiddleware → Extracts X-Project-Root     │
│                     → Sets ContextVar             │
│                                                  │
│  StreamableHTTPSessionManager                    │
│     → Stateless mode (no session affinity)       │
│     → JSON response format                       │
└─────────────────────────────────────────────────┘
```

**Lifecycle:**
1. **Startup:** Create `~/.code-rag/` directory, write PID file, pre-load MLX model, initialize server mode (concurrency primitives, persistent client cache).
2. **Running:** Process MCP requests, manage file watchers, serve health checks.
3. **Shutdown:** Stop all file watchers, close all Milvus clients, remove PID file.

**Project isolation:** Each request carries an `X-Project-Root` header. A `contextvars.ContextVar` propagates this through the async call chain, and the DB path is derived as `{project_root}/.code-rag/milvus.db`.

### 8.2 Concurrency Model

```
┌──────────────────────────────────────────────────────┐
│                  Concurrency Primitives               │
│                                                       │
│  asyncio.Semaphore(1)  ── Embedding serialization     │
│  │   Only one embedding operation at a time           │
│  │   Prevents GPU memory contention                   │
│  │                                                    │
│  asyncio.Lock()        ── Milvus write serialization  │
│  │   Only one write operation at a time               │
│  │   SQLite cannot handle concurrent writes           │
│  │                                                    │
│  (no lock)             ── Milvus reads are parallel   │
│      Multiple searches can execute simultaneously     │
└──────────────────────────────────────────────────────┘
```

### 8.3 Client Management

| Mode | Client Lifecycle | Use Case |
|------|-----------------|----------|
| HTTP Server | Persistent cache per `db_path` (reused across requests) | Production use |
| Stdio MCP | Fresh client per request | Debugging, backward compatibility |
| CLI | Ephemeral per invocation (closed after use) | Bulk indexing |

### 8.4 Server Management

The `code-rag-server.sh` script manages the server lifecycle:

- **Start:** Launches via `nohup`, waits up to 30 seconds for health check to pass.
- **Stop:** Sends SIGTERM, waits 10 seconds for graceful shutdown, then SIGKILL if needed.
- **Status:** Checks PID file and health endpoint.
- **Idempotent:** Calling `start` when already running is a no-op.

Runtime files: `~/.code-rag/server.pid`, `~/.code-rag/server.log`.

---

## 9. File Watcher

### 9.1 Architecture

```
macOS FSEvents (kernel)
    │
    ▼
watchdog.Observer (background thread)
    │
    ▼
FileChangeHandler._should_handle() ── fast pre-filter
    │   Extension check, dotfile exclusion, size cap,
    │   excluded directory check, .d.ts exclusion
    ▼
asyncio.Queue (thread-safe bridge)
    │
    ▼
ProjectWatcher._drain_queue() ── event aggregation
    │   Merge logic: delete always wins
    │   deleted → recreated = modified
    ▼
Debounce Timer (2 seconds, resets on each new event)
    │
    ▼
Git Check (.git/index.lock detection)
    │   If git active: defer 3 seconds
    ▼
Batch Processing (max 100 files)
    ├── Phase 1: Deletes (fast, no embedding)
    └── Phase 2: Upserts (embed + write, serialized)
```

### 9.2 Configuration

| Parameter | Default | Env Var |
|-----------|---------|---------|
| Enabled | `true` | `CODE_RAG_WATCH` |
| Debounce interval | 2.0 seconds | `CODE_RAG_WATCH_DEBOUNCE` |
| Max batch size | 100 files | `CODE_RAG_WATCH_MAX_BATCH` |
| Git settle time | 3.0 seconds | `CODE_RAG_WATCH_GIT_SETTLE` |

### 9.3 Design Decisions

- **Debouncing** prevents fragmented processing during burst scenarios (e.g., `git checkout` touching many files).
- **Git awareness** detects `.git/index.lock` and defers processing to avoid indexing partial states.
- **Batch overflow** caps processing at 100 files per batch, re-queuing excess for the next cycle.
- **Cooperative scheduling** via `await asyncio.sleep(0)` between upserts yields control so search requests can proceed during long indexing batches.

---

## 10. MCP Tool Interface

The server exposes eight tools via the Model Context Protocol:

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `search_code` | Search code files only | `query` (string), `n` (int, default 5), `language` (enum filter) |
| `search_docs` | Search documentation/YAML files | `query` (string), `n` (int, default 5) |
| `search_all` | Search all file types | `query` (string), `n` (int, default 10) |
| `index_file` | Index or re-index a single file | `path` (string, absolute) |
| `index_directory` | Index all supported files in a directory | `path` (string, absolute) |
| `list_indexed` | List indexed files grouped by type | -- |
| `get_stats` | Index statistics (counts by language/type) | -- |
| `watcher_status` | File watcher status and cumulative stats | -- |

All tools require the `X-Project-Root` header. Missing header returns an error with setup instructions.

---

## 11. Supported Languages & File Types

| Extension(s) | Language | Chunk Type | Chunking Tier Available |
|-------------|----------|------------|------------------------|
| `.java` | Java | code | 1 (code-chunk), 2 (tree-sitter), 3 (regex) |
| `.ts`, `.tsx` | TypeScript | code | 1, 2, 3 |
| `.js`, `.jsx` | JavaScript | code | 1, 2, 3 |
| `.py` | Python | code | 1, 2 |
| `.rs` | Rust | code | 1 |
| `.go` | Go | code | 1 |
| `.json` | JSON | config | 3 (character split) |
| `.xml` | XML | config | 3 |
| `.yaml`, `.yml` | YAML | documentation | 3 (document separator) |
| `.md` | Markdown | documentation | 3 |
| `.properties` | Properties | config | 3 |
| `.gradle` | Gradle | config | 3 |

**Auto-exclusions:** Files >1 MB, hidden files/directories, `.d.ts` declaration files, `.js`/`.jsx` when corresponding `.ts`/`.tsx` exists, JAXB-generated Java files.

---

## 12. Configuration Reference

### 12.1 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CODE_RAG_HOST` | `127.0.0.1` | Server bind address. **Do not change to `0.0.0.0` without adding authentication.** |
| `CODE_RAG_PORT` | `7101` | Server port |
| `CODE_RAG_WATCH` | `true` | Enable file watcher |
| `CODE_RAG_WATCH_DEBOUNCE` | `2.0` | Debounce interval (seconds) |
| `CODE_RAG_WATCH_MAX_BATCH` | `100` | Max files per watcher batch |
| `CODE_RAG_WATCH_GIT_SETTLE` | `3.0` | Wait after git operations (seconds) |
| `CODE_RAG_PROJECT_ROOT` | `$CWD` | Project root (stdio mode only) |
| `EMBED_MODEL_PATH` | `./models/qodo-embed-1-1.5b-mlx-q8` | Path to embedding model |
| `HF_HUB_OFFLINE` | Auto-set to `1` if model cached | Prevents HuggingFace network calls |
| `MAX_CHUNK_SIZE` | `2000` | Maximum chunk size (characters) |
| `MIN_CHUNK_SIZE` | `100` | Minimum chunk size (characters) |

### 12.2 Per-Project Configuration

| File | Location | Description |
|------|----------|-------------|
| `.ragignore` | Project root | Directory exclusion list (replaces defaults if present) |
| `.code-rag/milvus.db` | Project root | Vector index database (add `.code-rag/` to `.gitignore`) |
| `.code-rag/fts.db` | Project root | Full-text search index (SQLite FTS5) |
| `.mcp.json` | Project root | MCP server configuration for Claude Code |

### 12.3 Global Runtime Files

| Path | Description |
|------|-------------|
| `~/.code-rag/server.pid` | Server process ID |
| `~/.code-rag/server.log` | Server log output |

---

## 13. Performance Characteristics

| Operation | Typical Latency | Notes |
|-----------|----------------|-------|
| Model load | ~3-4 seconds | Once at server startup |
| Embedding (per chunk) | ~460ms | Apple Silicon GPU via MLX |
| Search query | ~50-80ms | After first search (~100ms) |
| File watcher debounce | 2 seconds | Configurable |
| Incremental index (unchanged file) | <1ms | SHA-256 hash comparison |
| Index size (vector + FTS) | ~300 MB | For ~6,000 files / ~27,000 chunks |

---

## 14. Dependency Licenses

All dependencies are open source with permissive licenses:

| Component | License | Maintainer |
|-----------|---------|------------|
| MLX | Apache 2.0 | Apple |
| Milvus / Milvus Lite | Apache 2.0 | Linux Foundation / Zilliz |
| Qodo-Embed-1-1.5B | Apache 2.0 | Qodo |
| code-chunk | MIT | supermemory |
| transformers | Apache 2.0 | Hugging Face |
| tree-sitter | MIT | tree-sitter |
| Starlette | BSD 3-Clause | Encode |
| uvicorn | BSD 3-Clause | Encode |
| watchdog | Apache 2.0 | gorakhargosh |
| MCP SDK | MIT | Anthropic |

---

## 15. Prerequisites & Platform Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Platform | macOS with Apple Silicon | macOS 15+ |
| Chip | M1 | M2/M3/M4 (faster GPU) |
| Python | 3.12+ | 3.12+ |
| Node.js | 18+ | 20+ |
| Disk Space | ~2 GB (model + venv) | ~3 GB (with indexes) |
| RAM | 8 GB | 16+ GB |

---

## 17. Future Considerations

### Pluggable Embedding Providers

Abstracting the embedding layer behind a provider interface would allow switching between:
- **Local (default):** Qodo-Embed via MLX (current, nothing leaves machine)
- **OpenAI:** `text-embedding-3-large` (cross-platform, requires API key, code sent to OpenAI)
- **Voyage AI:** `voyage-code-3` (code-specialized, requires API key, code sent to Voyage)
- **Ollama:** Local embedding models via Ollama (cross-platform, nothing leaves machine, but slower than MLX)

### Pluggable Vector Stores

The vector database could similarly be abstracted:
- **Milvus Lite (default):** Current embedded SQLite-backed store
- **PostgreSQL + pgvector:** For users with existing Postgres infrastructure (local)
- **ChromaDB:** Python-native alternative with simpler API (local)
- **LanceDB:** High-performance columnar alternative (local)

### Cross-Platform Support

Replacing MLX with a cross-platform embedding solution (ONNX Runtime, or cloud API) would enable Linux/Windows support at the cost of either performance (CPU-only inference) or privacy (cloud embeddings).
