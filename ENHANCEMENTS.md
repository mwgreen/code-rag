# Code-RAG Enhancement Spec

These enhancements address real issues found during production use. All changes must preserve current default behavior — existing projects with no `.ragconfig` should work identically.

Read `ARCHITECTURE.md` and `USAGE.md` before starting. The codebase is small (~8 Python files). Key files: `rag_milvus.py` (indexing + search), `tools.py` (MCP tool definitions), `file_watcher.py` (live reindex), `fts_hybrid.py` (hybrid search), `chunking.py` (file type detection).

## 1. Project-level `.ragconfig` file

**Problem:** There's no way to configure code-rag behavior per-project beyond `.ragignore`. Users need to tune relevance thresholds, exclude file patterns (not just directories), and control type classification without changing code-rag source.

**Design:** YAML file at `{project_root}/.ragconfig`. Loaded alongside `.ragignore`. If absent, all current defaults apply unchanged.

```yaml
# .ragconfig — per-project code-rag configuration
# All fields are optional. Omitted fields use built-in defaults.

# Minimum relevance score (0.0–1.0). Results below this are dropped.
# Default: 0.0 (no filtering — current behavior)
min_relevance: 0.35

# File extension exclusions (in addition to .ragignore directory exclusions)
# These apply globally across the project.
exclude_extensions:
  - .yml
  - .log
  - .png
  - .jpg

# Path-based exclusions — glob patterns matched against the file's path
# relative to the project root. More flexible than .ragignore directory names.
exclude_patterns:
  - ".playwright-mcp/page-*"
  - ".playwright-mcp/console-*"
  - "**/test-fixtures/**"

# Override type classification for specific paths.
# Default classification: .yml/.yaml/.md → "documentation", .json/.xml/.properties/.gradle → "config", everything else → "code"
# Use this when the default is wrong (e.g., YAML files that aren't docs).
type_overrides:
  - pattern: ".playwright-mcp/*.md"
    type: documentation
  - pattern: "deploy/**/*.yaml"
    type: config

# read_file tool configuration
# Maximum bytes returned by the read_file MCP tool (always available).
# Default: 102400 (100KB)
read_file_max_bytes: 102400
```

**Implementation notes:**
- Load in `rag_milvus.py` at the same point `.ragignore` is loaded (around line 736)
- `exclude_extensions` checked in `index_directory()` alongside the existing dotfile/size/JAXB checks
- `exclude_patterns` uses `fnmatch` against the relative path from project root
- `min_relevance` applied in `tools.py` before formatting results — filter where `(1 - distance) < min_relevance`
- `type_overrides` checked in `chunking.py:detect_language()` before the default extension-based classification
- `.ragconfig` changes should be picked up by the file watcher (hot reload) — see Enhancement 3
- Use `yaml.safe_load()` — it's already a dependency via Milvus

## 2. `read_file` MCP tool

**Problem:** After searching, models need to read full file content. Without a dedicated tool, they either hallucinate MCP resource calls (`read_mcp_resource` on nonexistent servers — observed repeatedly in production) or fall back to shell `cat`. A discoverable tool in the tool list prevents hallucination.

**Design:** Add to `tools.py` alongside the existing search tools. Always available — models already have file access via shell commands, so the tool just makes the correct path discoverable and prevents hallucinated MCP resource calls.

```
Tool: read_file
Input:
  path: string (required) — absolute path to file
  start_line: integer (optional) — 1-based start line
  end_line: integer (optional) — 1-based end line (inclusive)
Output:
  File content with line numbers, truncated to read_file_max_bytes
  Metadata: path, language, total lines, byte size
Error if:
  - Path is outside the project root (directory traversal prevention)
  - File doesn't exist
  - File exceeds max_bytes and no line range specified
```

**Implementation notes:**
- Path must be under `get_current_project_root()` — reject anything with `..` that escapes
- Return content with `{start_line}-{end_line}` metadata so the model knows what it got
- If file exceeds `read_file_max_bytes` and no line range given, return the first N lines that fit plus a message: `"File truncated at {n} lines ({bytes} bytes). Use start_line/end_line to read specific sections."`
- `read_file_max_bytes` is read from `.ragconfig` if present, otherwise defaults to 102400

## 3. `.ragignore` and `.ragconfig` hot reload

**Problem:** Changes to `.ragignore` require a server restart. The file watcher already watches the project root.

**Design:** In `file_watcher.py`, detect changes to `.ragignore` and `.ragconfig`. On change:
1. Reload the exclusion/config state
2. For newly-excluded paths: call `delete_by_path` for each indexed file that now matches an exclusion
3. For newly-included paths: queue them for indexing in the next batch

**Implementation notes:**
- In `FileChangeHandler._should_handle()`, check if the changed file is `.ragignore` or `.ragconfig`
- If so, set a flag that triggers a config reload + cleanup pass in `_process_batch()`
- Cleanup: iterate `list_indexed()` output, check each path against new exclusions, delete matches
- This is bounded by the number of indexed files, not the filesystem — should be fast

## 4. Relevance floor

**Problem:** Searches return results at 0.26 relevance (barely above random). The model treats all returned results as authoritative, leading to answers based on irrelevant content. Currently there's only a warning text at distance >0.5.

**Design:** In `tools.py`, after search results are returned from `rag_milvus.search()`, filter out results where `(1 - distance) < min_relevance` before formatting.

**Implementation notes:**
- Default `min_relevance: 0.0` preserves current behavior exactly
- Applied in the formatting functions (`_format_results` in tools.py around line 54)
- If ALL results are below the floor, return the warning message (don't return empty)
- The low-similarity warning (line 62-68) should still appear but now it's supplemented by actual filtering when configured

## 5. `delete_by_pattern` MCP tool

**Problem:** When junk files get indexed (119 Playwright YAML snapshots in our case), there's no way to clean up without CLI access and a server restart.

**Design:** New MCP tool:

```
Tool: delete_by_pattern
Input:
  pattern: string (required) — glob pattern matched against indexed file paths
  dry_run: boolean (optional, default true) — if true, return what would be deleted without deleting
Output:
  List of deleted (or would-be-deleted) file paths and chunk counts
```

**Implementation notes:**
- Use `list_indexed()` to get all paths, filter with `fnmatch`
- Paths must be within the project root (safety check)
- Default `dry_run: true` prevents accidental mass deletion
- Call existing `delete_by_path()` for each match
- Log deletions at INFO level

## 6. Stale entry cleanup on startup

**Problem:** When files are deleted while the server is stopped, their index entries persist as orphans. The file watcher only catches live deletions.

**Design:** On server startup (or first request for a project), scan all indexed paths and delete entries for files that no longer exist on disk.

**Implementation notes:**
- In `rag_milvus.py`, add a `cleanup_stale_entries(db_path)` function
- Call it lazily on first search/index for a project, not on server boot (avoids slow startup)
- Use `list_indexed()` to get all paths, check `os.path.exists()`, delete missing ones
- Track last-cleanup timestamp per project to avoid running on every request — once per server lifetime is enough
- Log count of cleaned entries at INFO level

## 7. Search result hints

**Problem:** After getting search snippets, models don't know how to read the full file. They hallucinate MCP resource calls because it seems like the "right" MCP way to fetch content. Even with `read_file` available, a hint in search output guides better behavior.

**Design:** Append a one-line hint to search result output in `tools.py`:

```
To read the full file, use the read_file tool with the path shown above.
```

**Implementation notes:**
- Add to `_format_results()` in tools.py after all result blocks
- Single line, not per-result — just once at the end

---

## Priority Order

1. **`.ragconfig` file** (foundation — all other features read from it)
2. **`read_file` tool** (stops the most common hallucination)
3. **Relevance floor** (stops garbage results, simple filter)
4. **`delete_by_pattern` tool** (cleanup without restart)
5. **Search result hints** (one line of text, high impact)
6. **Hot reload** (convenience, moderate complexity)
7. **Stale cleanup** (nice-to-have, prevents slow drift)

## Testing

For each enhancement, verify:
- A project with NO `.ragconfig` behaves identically to today
- A project with an empty `.ragconfig` (`{}`) behaves identically to today
- Each config field works independently (don't require all fields)
- The file watcher doesn't crash on config file changes
- `read_file` rejects paths outside the project root
- `delete_by_pattern` with `dry_run: true` doesn't delete anything
