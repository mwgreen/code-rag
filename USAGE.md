# Code-RAG Usage Guide

## Claude Code Integration (Primary Use)

Code-rag is designed as a persistent MCP server for Claude Code. Once configured, Claude Code automatically has access to semantic search tools.

### 1. Configure `.mcp.json`

Add to your project's `.mcp.json`:

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

The `X-Project-Root` header tells the server where to find (or create) the project's index at `{project}/.code-rag/milvus.db`. If the header is missing, all tool calls return an error with setup instructions.

### 2. Auto-Start via SessionStart Hook

Add to `~/.claude/settings.json` to auto-start the server on every Claude Code session:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/code-rag/code-rag-server.sh start",
            "timeout": 45000
          }
        ]
      }
    ]
  }
}
```

### 3. Available MCP Tools

Once configured, Claude Code can use these tools:

| Tool | Description | Key Arguments |
|------|-------------|---------------|
| `search_code` | Search code files | `query`, `n` (default 5), `language` filter |
| `search_docs` | Search documentation/YAML files | `query`, `n` (default 5) |
| `search_all` | Search everything (code + docs + config) | `query`, `n` (default 10) |
| `index_file` | Index or re-index a single file | `path` (absolute) |
| `index_directory` | Index all supported files in a directory | `path` (absolute) |
| `list_indexed` | List all indexed files grouped by type | — |
| `get_stats` | Get index statistics | — |

Just ask Claude Code naturally — it will use the tools automatically:
- "Search for JWT authentication code"
- "Find examples of REST API controllers in Java"
- "Index the new feature I added in src/features/payment"
- "How many files are indexed?"

## Model Configuration

Two local models are used: an **embedding model** (vectors for chunks and queries) and a
**description model** (an LLM that writes a one-sentence summary of each chunk before it is embedded).
Both are chosen by `model_config.py`. Run `venv/bin/python model_config.py --list` to see every
option and what your machine currently resolves to.

### Profiles

Pick a profile for your hardware with `CODE_RAG_PROFILE`:

| Profile | RAM | Embedding model | Description model | Notes |
|---------|-----|-----------------|-------------------|-------|
| `max` | 64 GB+ | Qwen3-Embedding-4B (2560 dims) | Qwen3.6-35B-A3B (~22 GB) | Best descriptions; ~29 GB peak while indexing |
| `high` | 48 GB | Qwen3-Embedding-4B (2560 dims) | Gemma 4 E4B (~7.5 GB) | **Default.** ~12 GB peak while indexing, ~5 GB serving |
| `medium` | 24-32 GB | Qwen3-Embedding-0.6B (1024 dims) | Gemma 4 E4B | Fast indexing, smaller indexes |
| `low` | 16 GB | Qwen3-Embedding-0.6B (1024 dims) | Gemma 4 E2B (~4 GB) | Smallest footprint with descriptions on |
| `legacy` | 32 GB | SFR-Embedding-Code-2B (2304 dims) | Gemma 3 4B | Pre-Sept-2026 defaults; needs the mlx-embeddings patches |

```bash
export CODE_RAG_PROFILE=medium
./download-embed-model.sh          # fetches the profile's embedding model, quantizes to Q8
./download-description-model.sh    # caches the profile's description model for offline use
./index.sh --force /path/to/project   # embedding model changed -> full re-index
```

### Resolution order

For each model, the first of these that is set wins:

1. Direct override: `EMBED_MODEL_PATH` (a local MLX model dir) / `CODE_RAG_DESCRIPTION_MODEL` (any mlx-lm HF id)
2. Registry key: `CODE_RAG_EMBED_MODEL` / `CODE_RAG_DESCRIPTION_MODEL_KEY` (keys from `model_config.py --list`)
3. Profile: `CODE_RAG_PROFILE`
4. Auto-detect: the best model that is already downloaded on this machine
5. The `high` profile

Step 4 means an existing machine keeps using whatever it has after a code update, and a fresh machine
picks up the new defaults once the download scripts have run. Put these in `code-rag/.env` (see
`.env.example`) or export them before starting the server. `curl localhost:7101/health` shows the
resolved `embed_model`, `description_model` and `model_profile`.

### Mixing and matching

```bash
./download-embed-model.sh qwen3-embed-0.6b
export CODE_RAG_EMBED_MODEL=qwen3-embed-0.6b            # switching embedders requires re-indexing

./download-description-model.sh qwen3.6-35b-a3b
export CODE_RAG_DESCRIPTION_MODEL_KEY=qwen3.6-35b-a3b   # cached descriptions stay valid

export CODE_RAG_DESCRIPTIONS=0                          # descriptions off: faster indexing
```

Requirements: `mlx-embeddings>=0.1.0` (native Qwen3-Embedding support), `mlx-lm>=0.31.3` (Gemma 4, Qwen3.5/3.6).
On an M5 Mac, MLX uses the GPU neural accelerators (3-4x faster prompt processing) on macOS 26.2 or later.

## Server Management

### Starting and Stopping

```bash
./code-rag-server.sh start    # Start (or confirm already running)
./code-rag-server.sh stop     # Graceful shutdown
./code-rag-server.sh status   # Check if running
./code-rag-server.sh restart  # Stop + start
```

The server writes its PID to `~/.code-rag/server.pid` and logs to `~/.code-rag/server.log`.

### Health Check

```bash
curl http://127.0.0.1:7101/health
# {"status": "ok"}
```

### Viewing Logs

```bash
tail -f ~/.code-rag/server.log
```

## Indexing

### Via CLI (Bulk Indexing)

The CLI is best for initial indexing of large codebases. If the server is running, the CLI
posts the job to it (`POST /index`) and shows progress while the server keeps serving searches;
the server stays the only writer to the Milvus Lite files. With no server running the CLI
indexes in-process.

```bash
# Index (incremental by default — only changed files)
./index.sh /path/to/your/project

# Full re-index (re-embeds everything)
./index.sh /path/to/your/project --full

# Restart server
./code-rag-server.sh start
```

The index is stored at `/path/to/your/project/.code-rag/milvus.db`. Add `.code-rag/` to your project's `.gitignore`.

### Via MCP (Live Updates)

While the server is running, use `index_file` or `index_directory` tools through Claude Code for incremental updates. This is slower than CLI for large batches but doesn't require stopping the server.

### CLI Options

```bash
python3 index_codebase.py --path /path/to/code [OPTIONS]

Options:
  --full              Full re-index (ignore file hashes, re-embed everything)
  --clear             Clear existing index before indexing
  --extensions        Comma-separated extensions (e.g., .java,.js,.ts)
  --exclude-dirs      Extra directories to exclude (merged with .ragignore)
  --no-jaxb-filter    Disable JAXB-generated Java file detection
  --limit N           Max files to index (for testing)
```

### What Gets Indexed

**Default extensions:** `.java`, `.js`, `.ts`, `.tsx`, `.jsx`, `.json`, `.xml`, `.yaml`, `.yml`, `.md`, `.gradle`, `.properties`

**Auto-excluded directories (defaults):**
`node_modules`, `build`, `dist`, `target`, `bin`, `test`, `tests`, `ext`, `bower_components`, `.sencha`, `locale`, `packages`, `sass`, `lib`, `libs`, `vendor`, `vendors`, `data`, `venv`, `cdk.out`, `generated`

**Also excluded:** hidden dirs (`.git`, `.idea`, etc.), files >1MB, hidden files, `.d.ts` files, `.js`/`.jsx` when `.ts`/`.tsx` exists, JAXB-generated Java files.

### Customizing Exclusions with `.ragignore`

Place a `.ragignore` file in your **project root** (the directory you're indexing) to customize exclusions:

```
# Directories to exclude (one per line)
node_modules
build
dist
target
vendor
__pycache__

# Project-specific
my_legacy_code
generated_protos
```

If `.ragignore` exists, it **replaces** the default exclusion list entirely. Lines starting with `#` are comments.

### Stale File Cleanup

When files are deleted or moved, their old index entries are automatically cleaned up during the next indexing run. The cleanup is scoped to the directory being indexed.

### Background jobs, reconcile and verification

`index_directory` (MCP) and `POST /index` return immediately with a job id. Poll with the
`index_status` tool, `GET /jobs/<id>`, or `./code-rag-server.sh status`. One index or reconcile
job runs per project at a time; a second request returns the running job.

The index is kept consistent by a **reconcile** pass that compares the files on disk, the
vector store and the keyword store, then removes deleted/excluded files, repairs keyword rows
from the vector rows (no re-embedding), drops orphans, and indexes new or changed files. It runs
when the watcher starts for a project, after `.git/HEAD` changes (checkout, rebase), after
`.ragignore`/`.ragconfig` changes, and every `CODE_RAG_RECONCILE_INTERVAL` seconds (default 6h).
Run it by hand with the `verify_index` tool: without arguments it reports drift; with
`repair=true` it starts a reconcile job.

### What gets indexed

`indexing_rules.py` is the single predicate used by the CLI, the watcher and reconcile:
configured extensions, inside the project root, no hidden path component (`.git/`, `.nuxt/`,
`.claude/`...), not in an excluded directory, `.ragconfig` exclusions, no `.d.ts`, no `.js` with a
`.ts` sibling, no minified/bundled files (by name, or js/ts/json with lines over 2000 chars), under
1 MB, not JAXB-generated.

### Search scores

`Relevance` is cosine similarity between the query and the chunk (1.0 = identical). Hits found
only by the keyword index show `Match: keyword` and no score; hits found by both show
`Match: semantic+keyword`. `min_relevance` in `.ragconfig` drops semantic hits below that
similarity (keyword hits always pass).

## Troubleshooting

### Server Won't Start

```bash
# Check if already running
./code-rag-server.sh status

# Check logs (timestamped; rotated at 20 MB by the launcher)
./code-rag-server.sh logs 100

# Check if port is in use
lsof -i :7101

# Force cleanup and restart
rm -f ~/.code-rag/server.pid
./code-rag-server.sh start
```

### "No project configured" Error

Every MCP request needs the `X-Project-Root` header. Check your `.mcp.json`:
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

### CLI Says "Server is running and holds the DB lock"

The CLI can't access the DB while the server is running (Milvus Lite exclusive lock). Either:
1. Stop the server: `./code-rag-server.sh stop`, run CLI, then restart
2. Use the MCP `index_directory` tool through Claude Code instead

### No Results / Empty Index

```bash
# Check stats via curl
curl -s -X POST http://127.0.0.1:7101/mcp/ \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "X-Project-Root: /path/to/project" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_stats","arguments":{}}}' \
  | python3 -m json.tool
```

If 0 chunks, you need to index first. See [Indexing](#indexing) above.

### Slow Indexing

Embedding is the bottleneck (~460ms per chunk on Apple Silicon). Use `--limit` to test:
```bash
./index.sh /path/to/project --limit 10 --full
```

### Search Quality

- Use descriptive phrases: "user authentication with JWT tokens" not just "auth"
- Use filters to narrow: `language="java"` or `type_filter="code"`
- If results seem stale, run an incremental re-index

## Performance

- **First search** in a new session: ~100ms (client connection cached)
- **Subsequent searches**: ~50-80ms
- **Embedding speed**: ~460ms per chunk (MLX on Apple Silicon)
- **Model load time**: ~3-4s (once, at server startup)
- **Index size**: ~300MB for ~6,000 files / ~27,000 chunks
