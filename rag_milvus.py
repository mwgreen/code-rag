"""
RAG with Milvus Lite + MLX embeddings.

Each project gets its own DB file at {project_root}/.code-rag/milvus.db.
The HTTP server shares the MLX model across projects and caches Milvus clients per DB.
CLI and stdio MCP use ephemeral connections.

Full-text search via SQLite FTS5 sidecar for hybrid search (vector + keyword).
Results merged with Reciprocal Rank Fusion (RRF).

Concurrency model (all plain threads; the asyncio layer only dispatches):
  - mlx_gpu.GPU        re-entrant lock around every MLX call (embedding + description model)
  - _DB_LOCK           re-entrant lock around every mutation of Milvus and FTS, so the
                       watcher, background index jobs and MCP tool calls can never
                       interleave writes ("database is locked" in the old logs)
  - reads (search, query) run concurrently and retry on transient Milvus errors
Slow work (chunking, descriptions, embedding) is done before the write lock is taken.

Score semantics: Milvus returns cosine SIMILARITY for the COSINE metric (1.0 = identical).
Results expose it as `similarity`; keyword-only hits have similarity None.
"""

import json
import os

# Block all HuggingFace network access at runtime.
# Models must be pre-downloaded via setup.sh / download-*.sh.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import asyncio
import hashlib
import importlib
import importlib.util
import logging
import re
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from pymilvus import MilvusClient, MilvusException
from mlx_embeddings.utils import load as mlx_load, generate as mlx_generate

# pymilvus installs its own logging config at import time, which would undo any
# level set earlier. Its query_iterator warns on every call against milvus-lite
# ("failed to get mvccTs"), a known harmless limitation of the embedded server.
for _name in ("pymilvus", "pymilvus.orm.iterator"):
    logging.getLogger(_name).setLevel(logging.ERROR)
import mlx.core as mx

from mlx_gpu import GPU
import model_config
import chunking
import nl_descriptions
from indexing_rules import (  # noqa: F401 - re-exported for callers that import them from here
    IndexRules, get_excluded_dirs, load_ragconfig, invalidate_ragconfig, is_jaxb_generated,
    DEFAULT_EXTENSIONS,
)
from fts_hybrid import FTSIndex, rrf_merge

logger = logging.getLogger("code-rag.milvus")

# --- Embedding model configuration ---
# Resolved by model_config.py: EMBED_MODEL_PATH > CODE_RAG_EMBED_MODEL >
# CODE_RAG_PROFILE > first downloaded model > default profile (Qwen3-Embedding-4B).
_SCRIPT_DIR = Path(__file__).parent
_MODEL_PATH = model_config.resolve_embed_model_path()
_EMBED_DIM = None  # Auto-detected from model config.json

# Texts longer than this many tokens are truncated before embedding. mlx-embeddings
# defaults to 512, which cut the method body out of most contextualized chunks
# (they run 3,000-7,000 chars with a 350-1,000 char preamble). Qwen3-Embedding
# accepts 32k, SFR 8k; 2048 covers essentially every chunk.
EMBED_MAX_TOKENS = int(os.getenv("CODE_RAG_EMBED_MAX_TOKENS", "2048"))
# Chunks embedded per forward pass. Bounded so a 200-chunk file does not become one
# padded batch; sorted by length so padding is small.
EMBED_BATCH_SIZE = max(1, int(os.getenv("CODE_RAG_EMBED_BATCH", "8")))

COLLECTION_NAME = "codebase"
RESULT_FIELDS = ["document", "path", "language", "type", "doc_id",
                 "description", "start_line", "end_line", "class_name", "component"]
FTS_COLUMNS = ["path", "language", "type", "start_line", "end_line", "class_name", "component", "description"]

# Architectures mlx-embeddings does not ship. Registered into sys.modules from patches/
# on demand instead of being copied into site-packages (which any pip upgrade wiped).
_LEGACY_ARCHITECTURES = {
    "qwen2": "mlx_embeddings_qwen2.py",
    "codexembed2b": "mlx_embeddings_codexembed2b.py",
}


# =====================================================================
# Model
# =====================================================================

def _read_model_config(model_path: str) -> dict:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"Model config not found at {config_path}")
    with open(config_path) as f:
        return json.load(f)


def _detect_embed_dim(model_path: str) -> int:
    """Read hidden_size from the model's config.json to determine embedding dimension."""
    dim = _read_model_config(model_path).get("hidden_size")
    if not dim:
        raise RuntimeError(f"hidden_size not found in {Path(model_path) / 'config.json'}")
    return dim


def _get_embed_dim() -> int:
    global _EMBED_DIM
    if _EMBED_DIM is None:
        _EMBED_DIM = _detect_embed_dim(_MODEL_PATH)
    return _EMBED_DIM


def register_legacy_architecture(model_type: str) -> None:
    """Make `mlx_embeddings.models.<model_type>` importable from patches/ if the
    installed mlx-embeddings does not provide it. mlx-embeddings resolves classes
    with importlib.import_module, so a sys.modules entry is all it needs."""
    fname = _LEGACY_ARCHITECTURES.get(model_type)
    if not fname:
        return
    name = f"mlx_embeddings.models.{model_type}"
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return  # shipped by the package (or copied there by an older setup.sh)
    except ImportError:
        pass
    path = _SCRIPT_DIR / "patches" / fname
    if not path.exists():
        raise RuntimeError(f"Embedding model type {model_type!r} needs {path}, which is missing")
    import mlx_embeddings.models  # ensure the parent package exists for relative imports
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    logger.info("Registered legacy embedding architecture %s from %s", model_type, path)


def compute_file_hash(file_path: str) -> str:
    """SHA256 of file content for change detection. Empty string if unreadable."""
    try:
        h = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''):
                h.update(block)
        return h.hexdigest()
    except Exception:
        return ""


_mlx_model = None
_mlx_tokenizer = None


def get_mlx_model():
    """Get or load the MLX embedding model (one-time load, under the GPU lock)."""
    global _mlx_model, _mlx_tokenizer, _EMBED_DIM

    if _mlx_model is not None:
        return _mlx_model, _mlx_tokenizer

    if not Path(_MODEL_PATH).exists():
        raise RuntimeError(
            f"Embedding model not found at {_MODEL_PATH}. "
            f"Run ./download-embed-model.sh (or ./setup.sh) to download it, "
            f"or set CODE_RAG_PROFILE / CODE_RAG_EMBED_MODEL / EMBED_MODEL_PATH. "
            f"See `python model_config.py --list`."
        )

    with GPU:
        if _mlx_model is not None:
            return _mlx_model, _mlx_tokenizer
        config = _read_model_config(_MODEL_PATH)
        _EMBED_DIM = config.get("hidden_size") or _detect_embed_dim(_MODEL_PATH)
        register_legacy_architecture(config.get("model_type", ""))
        model_name = Path(_MODEL_PATH).name
        logger.info("Loading embedding model %s from %s", model_name, _MODEL_PATH)
        t0 = time.perf_counter()
        _mlx_model, _mlx_tokenizer = mlx_load(_MODEL_PATH)
        logger.info("%s ready (%d dims, %.1fs, max %d tokens, batch %d)",
                    model_name, _EMBED_DIM, time.perf_counter() - t0, EMBED_MAX_TOKENS, EMBED_BATCH_SIZE)

    return _mlx_model, _mlx_tokenizer


_QUERY_INSTRUCTION = None


def _get_query_instruction() -> str:
    """Query-side instruction prefix (Qwen3-Embedding, SFR). Documents are embedded raw."""
    global _QUERY_INSTRUCTION
    if _QUERY_INSTRUCTION is None:
        _QUERY_INSTRUCTION = model_config.query_instruction_for_model(_MODEL_PATH)
    return _QUERY_INSTRUCTION


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed documents. Batched, length-sorted, explicit token limit, GPU lock per batch
    so a search query only ever waits for one batch."""
    if not texts:
        return []
    model, tokenizer = get_mlx_model()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    out: List[Optional[List[float]]] = [None] * len(texts)
    for start in range(0, len(order), EMBED_BATCH_SIZE):
        idx = order[start:start + EMBED_BATCH_SIZE]
        batch = [texts[i] for i in idx]
        with GPU:
            output = mlx_generate(model, tokenizer, texts=batch,
                                  max_length=EMBED_MAX_TOKENS, padding=True, truncation=True)
            vectors = output.text_embeds.tolist()
            mx.clear_cache()
        for i, vec in zip(idx, vectors):
            out[i] = vec
    return out  # type: ignore[return-value]


def embed_query(query: str) -> List[float]:
    """Embed a search query, prepending the instruction prefix if the model uses one."""
    instruction = _get_query_instruction()
    text = f"{instruction}{query}" if instruction else query
    return embed_texts([text])[0]


# =====================================================================
# Clients, locks, retries
# =====================================================================

_DB_LOCK = threading.RLock()       # every Milvus/FTS mutation
_CLIENT_LOCK = threading.RLock()   # client cache bookkeeping

_active_client: Optional[MilvusClient] = None      # CLI batch session
_persistent_clients: Dict[str, MilvusClient] = {}  # server mode: db_path -> client
_prepared_dbs: Set[str] = set()                    # collection ensured + model checked
_server_mode = False

_fts = FTSIndex("chunks_fts", FTS_COLUMNS, indexed_metadata={"description"})

_GRPC_OPTIONS = {
    # Relaxed keepalive: pymilvus defaults to 10s pings, which made the embedded
    # milvus-lite server answer GOAWAY/too_many_pings after a while.
    "grpc.keepalive_time_ms": 300_000,
    "grpc.keepalive_timeout_ms": 10_000,
    "grpc.keepalive_permit_without_calls": False,
}

# Strong consistency: a query issued right after an insert must see it. With the
# default (bounded) level, verify/reconcile and search-after-index could observe
# rows the server had not yet made visible.
_STRONG = {"consistency_level": "Strong"}

_TRANSIENT_MARKERS = ("locked", "unavailable", "failed to connect", "goaway", "connection refused",
                      "deadline", "timeout", "too_many_pings", "socket", "broken pipe", "rpc")
_CONNECTION_MARKERS = ("failed to connect", "goaway", "connection refused", "socket", "unavailable",
                       "broken pipe")


def init_server_mode():
    """Server mode: persistent clients per project, persistent FTS connections."""
    global _server_mode
    _server_mode = True
    _fts.set_server_mode(True)
    logger.info("Server mode initialized (persistent clients are created per project)")


def close_server_mode():
    global _server_mode
    with _CLIENT_LOCK:
        for path, client in list(_persistent_clients.items()):
            try:
                client.close()
                logger.info("Closed Milvus client: %s", path)
            except Exception as e:
                logger.warning("Error closing Milvus client %s: %s", path, e)
        _persistent_clients.clear()
        _prepared_dbs.clear()
    _fts.close_all()
    _server_mode = False


def _resolve_db_path(db_path: Optional[str]) -> str:
    if not db_path:
        raise ValueError("db_path is required. Each project stores its index at {project}/.code-rag/milvus.db")
    return db_path


def _open_client(db_path: str) -> MilvusClient:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return MilvusClient(db_path, grpc_options=dict(_GRPC_OPTIONS))


def _persistent_client(db_path: str) -> MilvusClient:
    with _CLIENT_LOCK:
        client = _persistent_clients.get(db_path)
        if client is None:
            client = _open_client(db_path)
            _persistent_clients[db_path] = client
            logger.info("Opened persistent Milvus client: %s", db_path)
        return client


def _evict_client(db_path: str) -> None:
    with _CLIENT_LOCK:
        client = _persistent_clients.pop(db_path, None)
        _prepared_dbs.discard(db_path)
    if client is not None:
        try:
            client.close()
        except Exception:
            pass


def _ensure_collection(client: MilvusClient) -> None:
    if not client.has_collection(COLLECTION_NAME):
        dim = _get_embed_dim()
        logger.info("Creating Milvus collection %s (dim=%d)", COLLECTION_NAME, dim)
        client.create_collection(collection_name=COLLECTION_NAME, dimension=dim, metric_type="COSINE",
                                 consistency_level="Strong")
        return  # create_collection() leaves the new collection loaded
    # An existing collection opens in the "released" state on a fresh client
    # (pymilvus 3 / Milvus Lite): every search/query/get fails with
    # "Collection 'codebase' is in state 'released'; call load() before search"
    # until it is loaded. This is the path every server restart takes.
    client.load_collection(COLLECTION_NAME)


def _write_model_config(meta_path: Path) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, 'w') as f:
        json.dump({"embed_model_path": _MODEL_PATH, "embed_dim": _get_embed_dim()}, f, indent=2)


def _check_model_consistency(db_path: str) -> None:
    """Refuse to mix embedding dimensions in one index; record which model built it."""
    meta_path = Path(db_path).parent / "model_config.json"
    current_dim = _get_embed_dim()
    if not meta_path.exists():
        _write_model_config(meta_path)
        return
    with open(meta_path) as f:
        stored = json.load(f)
    stored_dim = stored.get("embed_dim")
    if stored_dim and stored_dim != current_dim:
        raise RuntimeError(
            f"Embedding dimension mismatch: index was built with {stored_dim}-dim embeddings "
            f"(model: {stored.get('embed_model_path', 'unknown')}), but the current model produces "
            f"{current_dim}-dim embeddings (model: {_MODEL_PATH}). "
            f"Re-index with: ./index.sh {Path(db_path).parent.parent} --clear"
        )
    if stored.get("embed_model_path") != _MODEL_PATH:
        _write_model_config(meta_path)


def _prepare(client: MilvusClient, db_path: str) -> None:
    if db_path in _prepared_dbs:
        return
    _ensure_collection(client)
    _check_model_consistency(db_path)
    _prepared_dbs.add(db_path)


def _is_transient(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _TRANSIENT_MARKERS)


def _is_connection_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _CONNECTION_MARKERS)


def _with_client(db_path: Optional[str], fn: Callable[[MilvusClient], object], retries: int = 2,
                 prepare: bool = True):
    """Run fn(client) against the right client for the current mode.

    prepare=False skips _prepare (collection load + embedding-model consistency
    check). Only clear_collection uses it, so an index built by a different
    embedding model can still be dropped.

    Server mode: persistent client, transient errors retried (reopening the
    client on connection errors). CLI batch session: the session client.
    Otherwise: a fresh ephemeral client that is closed afterwards.
    """
    path = _resolve_db_path(db_path)

    if _server_mode:
        attempt = 0
        while True:
            client = _persistent_client(path)
            try:
                if prepare:
                    _prepare(client, path)
                return fn(client)
            except MilvusException as e:
                if attempt >= retries or not _is_transient(e):
                    raise
                attempt += 1
                logger.warning("Milvus call failed on %s (%s); retry %d/%d",
                               Path(path).parent.parent.name, str(e)[:160], attempt, retries)
                if _is_connection_error(e):
                    _evict_client(path)
                time.sleep(0.5 * attempt)

    if _active_client is not None:
        if prepare:
            _prepare(_active_client, path)
        return fn(_active_client)

    client = _open_client(path)
    try:
        if prepare:
            _prepare(client, path)
        return fn(client)
    finally:
        try:
            client.close()
        except Exception:
            pass


@contextmanager
def milvus_session(db_path: Optional[str] = None):
    """One connection for a whole batch operation (CLI indexing). No-op in server mode,
    where the persistent client is used anyway."""
    global _active_client
    path = _resolve_db_path(db_path)
    if _server_mode or _active_client is not None:
        yield
        return
    client = _open_client(path)
    # No eager _prepare: _with_client prepares on first use, and clear_collection
    # (prepare=False) must be able to drop a dimension-mismatched index first.
    _active_client = client
    try:
        yield
    finally:
        _active_client = None
        _prepared_dbs.discard(path)
        try:
            client.close()
        except Exception:
            pass


def _q(value: str) -> str:
    """Quote a string for a Milvus filter expression."""
    return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"') + '"'


def _doc_pk(doc_id: str) -> int:
    """Deterministic 63-bit primary key. Python's hash() is randomized per process."""
    return int.from_bytes(hashlib.sha256(doc_id.encode('utf-8')).digest()[:8], 'big') & 0x7FFFFFFFFFFFFFFF


def _abs(path: str) -> str:
    return str(Path(path).absolute())


# =====================================================================
# Reads
# =====================================================================

def _query_all(output_fields: List[str], db_path: Optional[str] = None, filter_expr: str = "",
               batch_size: int = 2000) -> List[Dict]:
    """Every row matching filter_expr, via query_iterator. The old offset-paging
    version stopped at 16,384 rows, which was two thirds of a real 25k-chunk index."""
    def run(client: MilvusClient):
        if not client.has_collection(COLLECTION_NAME):
            return []
        rows: List[Dict] = []
        iterator = client.query_iterator(collection_name=COLLECTION_NAME, filter=filter_expr,
                                         output_fields=output_fields, batch_size=batch_size, **_STRONG)
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                rows.extend(batch)
        finally:
            iterator.close()
        return rows
    return _with_client(db_path, run)


def _stored_hash(abs_path: str, db_path: Optional[str]) -> Optional[str]:
    try:
        rows = _with_client(db_path, lambda c: c.query(
            collection_name=COLLECTION_NAME, filter=f'path == {_q(abs_path)}',
            limit=1, output_fields=["content_hash"], **_STRONG))
    except Exception as e:
        logger.warning("Hash lookup failed for %s: %s", abs_path, e)
        return None
    return rows[0].get('content_hash') if rows else None


def file_needs_indexing(file_path: str, db_path: Optional[str] = None) -> bool:
    """True unless the stored content hash matches the file on disk."""
    current = compute_file_hash(file_path)
    if not current:
        return True
    return _stored_hash(_abs(file_path), db_path) != current


_HEADER_RE = re.compile(r'\A(?:#[^\n]*\n)+\n?')


def _content_fingerprint(content: str) -> str:
    """Dedup key: the chunk body without code-chunk's `# ...` context header,
    whitespace-normalized and hashed. The old key (first 500 chars) was mostly
    that header, so different methods of one class deduped into each other."""
    body = _HEADER_RE.sub('', content or '', count=1)
    return hashlib.sha1(' '.join(body.split()).encode('utf-8')).hexdigest()


def _as_int(value, default=0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def search(query: str, n: int = 5, type_filter: Optional[str] = None,
           language_filter: Optional[str] = None, db_path: Optional[str] = None) -> List[Dict]:
    """Hybrid search: vector similarity + FTS5 keyword search, merged with RRF.

    Each result has `similarity` (cosine, 1.0 = identical; None for keyword-only
    hits) and `source` ("semantic", "keyword" or "both").
    """
    fetch_n = max(n * 3, 10)

    # --- Vector search ---
    query_embedding = embed_query(query)
    filters = []
    if type_filter:
        filters.append(f'type == {_q(type_filter)}')
    if language_filter:
        filters.append(f'language == {_q(language_filter)}')
    filter_expr = " && ".join(filters) if filters else None

    raw = _with_client(db_path, lambda c: c.search(
        collection_name=COLLECTION_NAME, data=[query_embedding], limit=fetch_n,
        filter=filter_expr, output_fields=RESULT_FIELDS, **_STRONG))

    vector_results: List[Dict] = []
    for hit in (raw[0] if raw else []):
        entity = hit['entity']
        result = {
            'content': entity.get('document', ''),
            'doc_id': entity.get('doc_id', ''),
            'path': entity.get('path', ''),
            'language': entity.get('language', ''),
            'type': entity.get('type', ''),
            'similarity': float(hit['distance']),
            'source': 'semantic',
            'start_line': _as_int(entity.get('start_line')),
            'end_line': _as_int(entity.get('end_line')),
        }
        for field in ('description', 'class_name', 'component'):
            if entity.get(field):
                result[field] = entity[field]
        vector_results.append(result)

    # --- FTS5 keyword search ---
    fts_filters = {}
    if type_filter:
        fts_filters["type"] = type_filter
    if language_filter:
        fts_filters["language"] = language_filter
    fts_rows = _fts.search(query, n=fetch_n, filters=fts_filters or None, db_path=db_path)
    fts_results: List[Dict] = []
    for row in fts_rows:
        result = {
            'content': row.get('content', ''),
            'doc_id': row.get('doc_id', ''),
            'path': row.get('path', ''),
            'language': row.get('language', ''),
            'type': row.get('type', ''),
            'similarity': None,
            'source': 'keyword',
            'start_line': _as_int(row.get('start_line')),
            'end_line': _as_int(row.get('end_line')),
        }
        for field in ('description', 'class_name', 'component'):
            if row.get(field):
                result[field] = row[field]
        fts_results.append(result)

    # --- Merge with RRF ---
    if fts_results and vector_results:
        merged = rrf_merge(vector_results, fts_results, n=fetch_n)
        keyword_ids = {r['doc_id'] for r in fts_results}
        for r in merged:
            if r['source'] == 'semantic' and r['doc_id'] in keyword_ids:
                r['source'] = 'both'
    else:
        merged = fts_results or vector_results

    # Deduplicate identical code (copies of a file in two directories)
    seen: Set[str] = set()
    deduped = []
    for r in merged:
        fp = _content_fingerprint(r.get('content', ''))
        if fp in seen:
            continue
        seen.add(fp)
        deduped.append(r)

    final = deduped[:n]
    for r in final:
        r.pop("_rrf_score", None)
        r.pop("doc_id", None)
    return final


def get_stats(db_path: Optional[str] = None) -> Dict:
    """Index statistics over the whole index."""
    rows = _query_all(["path", "language", "type"], db_path=db_path)
    by_language: Dict[str, int] = {}
    by_type: Dict[str, int] = {}
    files: Set[str] = set()
    for r in rows:
        files.add(r.get('path', ''))
        by_language[r.get('language', 'unknown')] = by_language.get(r.get('language', 'unknown'), 0) + 1
        by_type[r.get('type', 'unknown')] = by_type.get(r.get('type', 'unknown'), 0) + 1
    return {'total_chunks': len(rows), 'total_files': len(files),
            'by_language': by_language, 'by_type': by_type}


def list_indexed_files(db_path: Optional[str] = None) -> Dict[str, List[str]]:
    """All indexed files grouped by type."""
    by_type: Dict[str, Set[str]] = {}
    for r in _query_all(["path", "type"], db_path=db_path):
        by_type.setdefault(r.get('type', 'unknown'), set()).add(r.get('path', ''))
    return {t: sorted(paths) for t, paths in by_type.items()}


def _get_indexed_paths_under(dir_path: str, db_path: Optional[str] = None) -> Set[str]:
    """Unique indexed file paths that start with dir_path."""
    prefix = dir_path.rstrip(os.sep) + os.sep
    try:
        rows = _query_all(["path"], db_path=db_path, filter_expr=f'path like {_q(prefix + "%")}')
        return {r['path'] for r in rows if r.get('path', '').startswith(prefix)}
    except Exception as e:
        logger.warning("_get_indexed_paths_under failed for %s: %s", dir_path, e)
        return set()


def _index_snapshot(db_path: str) -> Dict[str, Dict]:
    """{path: {'hash': content_hash, 'doc_ids': set}} for the whole index."""
    snapshot: Dict[str, Dict] = {}
    for r in _query_all(["path", "content_hash", "doc_id"], db_path=db_path):
        entry = snapshot.setdefault(r.get('path', ''), {'hash': None, 'doc_ids': set()})
        entry['hash'] = r.get('content_hash') or entry['hash']
        entry['doc_ids'].add(r.get('doc_id', ''))
    return snapshot


# =====================================================================
# Writes
# =====================================================================

def _milvus_delete_path(abs_path: str, db_path: Optional[str]) -> int:
    def run(client: MilvusClient):
        rows = client.query(collection_name=COLLECTION_NAME, filter=f'path == {_q(abs_path)}',
                            output_fields=["id"], **_STRONG)
        if rows:
            client.delete(collection_name=COLLECTION_NAME, filter=f'path == {_q(abs_path)}')
        return len(rows)
    return _with_client(db_path, run)


def _fts_delete_path(abs_path: str, db_path: Optional[str]) -> None:
    conn = _fts.connection(db_path)
    try:
        _fts.delete(conn, "path", abs_path)
    finally:
        _fts.close_ephemeral(conn)


def delete_by_path(file_path: str, db_path: Optional[str] = None) -> int:
    """Delete all chunks for a file from both stores. Returns the Milvus row count removed."""
    abs_path = _abs(file_path)
    removed = 0
    with _DB_LOCK:
        try:
            removed = _milvus_delete_path(abs_path, db_path)
        except Exception as e:
            logger.warning("delete_by_path failed for %s: %s", abs_path, e)
        try:
            if db_path:
                _fts_delete_path(abs_path, db_path)
        except Exception as e:
            logger.warning("FTS delete_by_path failed (non-fatal) for %s: %s", abs_path, e)
    return removed


def _fts_records(documents: List[str], metadatas: List[Dict], ids: List[str]) -> List[Dict]:
    return [{
        "doc_id": doc_id,
        "content": doc,
        "path": meta.get("path", ""),
        "language": meta.get("language", ""),
        "type": meta.get("type", ""),
        "start_line": str(meta.get("start_line", "")),
        "end_line": str(meta.get("end_line", "")),
        "class_name": meta.get("class_name", "") or "",
        "component": meta.get("component", "") or "",
        "description": meta.get("description", "") or "",
    } for doc, meta, doc_id in zip(documents, metadatas, ids)]


def _insert_documents(documents: List[str], metadatas: List[Dict], ids: List[str],
                      embeddings: List[List[float]], content_hash: str, db_path: Optional[str]) -> None:
    """Insert already-embedded chunks into Milvus and FTS. Caller holds _DB_LOCK."""
    data = []
    for doc, meta, doc_id, emb in zip(documents, metadatas, ids, embeddings):
        row = {"id": _doc_pk(doc_id), "vector": emb, "document": doc, "doc_id": doc_id, **meta}
        if content_hash:
            row["content_hash"] = content_hash
        data.append(row)
    _with_client(db_path, lambda c: c.insert(collection_name=COLLECTION_NAME, data=data))

    if db_path:
        try:
            conn = _fts.connection(db_path)
            try:
                _fts.insert(conn, _fts_records(documents, metadatas, ids))
            finally:
                _fts.close_ephemeral(conn)
        except Exception as e:
            logger.warning("FTS insert failed (non-fatal): %s", e)


def _apply_type_overrides(chunks: List[Dict], abs_path: str, db_path: Optional[str]) -> None:
    if not db_path:
        return
    project_root = str(Path(db_path).parent.parent)
    overrides = load_ragconfig(project_root).get('type_overrides', [])
    if not overrides:
        return
    import fnmatch
    rel_path = os.path.relpath(abs_path, project_root)
    for override in overrides:
        if fnmatch.fnmatch(rel_path, override.get('pattern', '')):
            for chunk in chunks:
                chunk['type'] = override['type']
            break


def add_file(path: str, force: bool = False, db_path: Optional[str] = None) -> int:
    """Index one file. Returns the number of chunks written (0 if unchanged or empty).

    Chunking, description generation and embedding happen before the write lock is
    taken, and the delete-then-insert of the file's rows happens under it, so a
    concurrent writer can never interleave with it and a failed embed leaves the
    previous chunks in place rather than an empty gap.
    """
    abs_path = _abs(path)
    if not force and not file_needs_indexing(abs_path, db_path):
        return 0

    content_hash = compute_file_hash(abs_path)
    chunks = chunking.chunk_file(abs_path)
    if not chunks:
        # Nothing to index (empty/unreadable): make sure stale chunks are gone.
        delete_by_path(abs_path, db_path)
        return 0

    _apply_type_overrides(chunks, abs_path, db_path)

    desc_enabled = nl_descriptions.is_enabled(db_path=db_path)
    descriptions = nl_descriptions.describe_chunks(chunks, db_path=db_path) if desc_enabled else [None] * len(chunks)

    docs = [c['content'] for c in chunks]
    docs_for_embed = [f"{desc}\n\n{c['content']}" if desc else c['content']
                      for c, desc in zip(chunks, descriptions)]
    metas = [{
        'path': abs_path,
        'language': c['language'],
        'type': c['type'],
        'start_line': c.get('start_line', 0),
        'end_line': c.get('end_line', 0),
        'class_name': c.get('class_name', '') or '',
        'component': c.get('component', '') or '',
        **({"description": desc} if desc else {}),
    } for c, desc in zip(chunks, descriptions)]
    ids = [f"{abs_path}::{i}" for i in range(len(chunks))]

    embeddings = embed_texts(docs_for_embed)

    with _DB_LOCK:
        delete_by_path(abs_path, db_path)
        _insert_documents(docs, metas, ids, embeddings, content_hash, db_path)
    return len(chunks)


def clear_collection(db_path: Optional[str] = None):
    """Drop all data (Milvus + FTS) for a project.

    Skips the embedding-model consistency check so an index built by a different
    model (dimension mismatch) can still be cleared: this is the recovery path
    for switching embedders. The description cache (descriptions.db) is keyed by
    chunk content, not by embedder, and is kept so the rebuild reuses it."""
    path = _resolve_db_path(db_path)
    with _DB_LOCK:
        def run(client: MilvusClient):
            if client.has_collection(COLLECTION_NAME):
                client.drop_collection(COLLECTION_NAME)
                logger.info("Collection cleared: %s", path)
        _with_client(path, run, prepare=False)
        _prepared_dbs.discard(path)
        _fts.clear(path)
        meta_path = Path(path).parent / "model_config.json"
        if meta_path.exists():
            meta_path.unlink()  # the next _prepare records the current model


def _rebuild_fts_for_path(abs_path: str, db_path: str) -> int:
    """Recreate a file's FTS rows from what Milvus already holds. No re-embedding."""
    rows = _query_all(RESULT_FIELDS, db_path=db_path, filter_expr=f'path == {_q(abs_path)}')
    if not rows:
        return 0
    docs = [r.get('document', '') for r in rows]
    metas = [{k: r.get(k, '') for k in ('path', 'language', 'type', 'start_line', 'end_line',
                                        'class_name', 'component', 'description')} for r in rows]
    ids = [r.get('doc_id', '') for r in rows]
    with _DB_LOCK:
        conn = _fts.connection(db_path)
        try:
            _fts.delete(conn, "path", abs_path)
            _fts.insert(conn, _fts_records(docs, metas, ids))
        finally:
            _fts.close_ephemeral(conn)
    return len(rows)


# =====================================================================
# Bulk operations: index_directory, verify, reconcile
# =====================================================================

def _project_root_for(db_path: Optional[str], fallback: str) -> str:
    return str(Path(db_path).parent.parent) if db_path else fallback


def index_directory(dir_path: str, extensions: Optional[List[str]] = None,
                    incremental: bool = True, progress_callback=None,
                    max_files: int = 0, extra_excludes: Optional[List[str]] = None,
                    jaxb_filter: bool = True, db_path: Optional[str] = None,
                    cancel_event: Optional[threading.Event] = None) -> Dict:
    """Index all indexable files under dir_path (a project root or a subdirectory of one)."""
    dir_path = Path(dir_path).resolve()
    project_root = _project_root_for(db_path, str(dir_path))
    rules = IndexRules(project_root, extensions, extra_excludes, jaxb_filter)

    all_files = list(rules.walk(str(dir_path)))
    if max_files > 0:
        all_files = all_files[:max_files]
    total_files = len(all_files)

    stats = {'files_indexed': 0, 'files_skipped': 0, 'files_removed': 0, 'chunks_created': 0,
             'errors': 0, 'cancelled': False, 'by_language': {}, 'by_type': {}}

    with milvus_session(db_path):
        if max_files == 0:
            disk_paths = set(all_files)
            stale_paths = _get_indexed_paths_under(str(dir_path), db_path) - disk_paths
            for stale_path in sorted(stale_paths):
                delete_by_path(stale_path, db_path)
                stats['files_removed'] += 1
            if stale_paths:
                logger.info("Removed %d stale file(s) from index under %s", len(stale_paths), dir_path)

        for idx, file_path in enumerate(all_files):
            if cancel_event is not None and cancel_event.is_set():
                stats['cancelled'] = True
                logger.info("Indexing cancelled after %d/%d files", idx, total_files)
                break
            try:
                num_chunks = add_file(file_path, force=not incremental, db_path=db_path)
                if num_chunks > 0:
                    stats['files_indexed'] += 1
                    stats['chunks_created'] += num_chunks
                    lang = chunking.detect_language(file_path)
                    t = chunking.detect_type(file_path, lang)
                    stats['by_language'][lang] = stats['by_language'].get(lang, 0) + num_chunks
                    stats['by_type'][t] = stats['by_type'].get(t, 0) + num_chunks
                else:
                    stats['files_skipped'] += 1
            except Exception as e:
                logger.warning("Error indexing %s: %s", file_path, e)
                stats['errors'] += 1
            if progress_callback:
                progress_callback(idx + 1, total_files, Path(file_path).name)

    # In server mode the idle unloader frees the description model; unloading here
    # would just force a reload for the next watcher batch.
    if not _server_mode and nl_descriptions.is_enabled(db_path=db_path):
        nl_descriptions.unload_model()

    return stats


def _diff_index(project_root: str, db_path: str, rules: Optional[IndexRules] = None) -> Dict:
    """Compare disk, Milvus and FTS. Shared by verify_index and reconcile."""
    rules = rules or IndexRules(project_root)
    disk = set(rules.walk())
    milvus = _index_snapshot(db_path)
    fts = _fts.snapshot(db_path) if Path(_fts.db_path(db_path)).exists() else {}
    indexed = set(milvus)

    missing_on_disk = indexed - disk        # deleted, moved, or now excluded by the rules
    not_indexed = disk - indexed
    changed = {p for p in disk & indexed if milvus[p]['hash'] != compute_file_hash(p)}
    fts_orphans = set(fts) - indexed
    fts_mismatch = {p for p in indexed & disk if fts.get(p, set()) != milvus[p]['doc_ids']}
    return {
        'rules': rules,
        'disk_files': len(disk),
        'indexed_files': len(indexed),
        'indexed_chunks': sum(len(v['doc_ids']) for v in milvus.values()),
        'fts_chunks': sum(len(v) for v in fts.values()),
        'missing_on_disk': sorted(missing_on_disk),
        'not_indexed': sorted(not_indexed),
        'changed': sorted(changed),
        'fts_orphans': sorted(fts_orphans),
        'fts_mismatch': sorted(fts_mismatch - changed),
    }


def verify_index(project_root: str, db_path: str, sample: int = 10) -> Dict:
    """Read-only consistency report between disk, Milvus and FTS."""
    d = _diff_index(project_root, db_path)
    report = {k: v for k, v in d.items() if k != 'rules' and not isinstance(v, list)}
    for key in ('missing_on_disk', 'not_indexed', 'changed', 'fts_orphans', 'fts_mismatch'):
        report[key] = len(d[key])
        report[f'{key}_sample'] = d[key][:sample]
    report['consistent'] = all(report[k] == 0 for k in
                               ('missing_on_disk', 'not_indexed', 'changed', 'fts_orphans', 'fts_mismatch'))
    return report


def reconcile(project_root: str, db_path: str, progress_callback=None,
              cancel_event: Optional[threading.Event] = None) -> Dict:
    """Make the index match the disk and the two stores match each other.

    Replaces three overlapping mechanisms from before (the watcher's initial scan,
    the once-per-server stale cleanup that stopped at 16,384 rows, and the config
    reload cleanup). Runs at watcher start, after git checkouts, periodically, and
    on demand through the verify_index tool.
    """
    d = _diff_index(project_root, db_path)
    stats = {'removed_missing': 0, 'fts_orphans_removed': 0, 'fts_rebuilt': 0,
             'files_indexed': 0, 'files_skipped': 0, 'errors': 0, 'cancelled': False,
             'disk_files': d['disk_files']}

    with milvus_session(db_path):
        for path in d['missing_on_disk']:
            delete_by_path(path, db_path)
            stats['removed_missing'] += 1

        if d['fts_orphans']:
            with _DB_LOCK:
                conn = _fts.connection(db_path)
                try:
                    for path in d['fts_orphans']:
                        _fts.delete(conn, "path", path)
                        stats['fts_orphans_removed'] += 1
                finally:
                    _fts.close_ephemeral(conn)

        for path in d['fts_mismatch']:
            try:
                _rebuild_fts_for_path(path, db_path)
                stats['fts_rebuilt'] += 1
            except Exception as e:
                logger.warning("FTS rebuild failed for %s: %s", path, e)
                stats['errors'] += 1

        todo = sorted(set(d['not_indexed']) | set(d['changed']))
        for idx, path in enumerate(todo):
            if cancel_event is not None and cancel_event.is_set():
                stats['cancelled'] = True
                break
            try:
                if add_file(path, force=True, db_path=db_path) > 0:
                    stats['files_indexed'] += 1
                else:
                    stats['files_skipped'] += 1
            except Exception as e:
                logger.warning("Reconcile: error indexing %s: %s", path, e)
                stats['errors'] += 1
            if progress_callback:
                progress_callback(idx + 1, len(todo), Path(path).name)

    if any(stats[k] for k in ('removed_missing', 'fts_orphans_removed', 'fts_rebuilt', 'files_indexed', 'errors')):
        logger.info("Reconcile %s: %s", Path(project_root).name,
                    ", ".join(f"{k}={v}" for k, v in stats.items() if v and k != 'disk_files'))
    return stats


# =====================================================================
# Async wrappers (HTTP server / watcher). Everything runs in worker threads.
# =====================================================================

async def search_async(query: str, n: int = 5, type_filter: Optional[str] = None,
                       language_filter: Optional[str] = None, db_path: Optional[str] = None) -> List[Dict]:
    return await asyncio.to_thread(search, query, n, type_filter, language_filter, db_path)


async def add_file_async(path: str, force: bool = False, db_path: Optional[str] = None) -> int:
    return await asyncio.to_thread(add_file, path, force, db_path)


async def delete_by_path_async(file_path: str, db_path: Optional[str] = None) -> int:
    return await asyncio.to_thread(delete_by_path, file_path, db_path)
