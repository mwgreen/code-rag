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

The CLI is best for initial indexing of large codebases. **The server must be stopped** because Milvus Lite uses exclusive SQLite locks.

```bash
# Stop server first
./code-rag-server.sh stop

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

## Troubleshooting

### Server Won't Start

```bash
# Check if already running
./code-rag-server.sh status

# Check logs
cat ~/.code-rag/server.log

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
