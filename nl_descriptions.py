"""
LLM-generated natural language descriptions for code chunks.

Uses a local MLX LLM (default: Qwen3-4B-MLX-4bit, configurable via
CODE_RAG_DESCRIPTION_MODEL env var) to generate one-sentence summaries of
code chunks, improving semantic search by bridging the vocabulary gap between
natural language queries and code.

Descriptions are cached in SQLite keyed by SHA256 of chunk content.
Enabled via CODE_RAG_DESCRIPTIONS=1 environment variable (off by default).
"""

import hashlib
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

# Block HuggingFace network access (also set by rag_milvus on import)
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

logger = logging.getLogger("code-rag.descriptions")

# --- Configuration ---

MODEL_ID = os.getenv("CODE_RAG_DESCRIPTION_MODEL", "mlx-community/gemma-3-4b-it-4bit")
MAX_GEN_TOKENS = 100
MAX_INPUT_CHARS = 2000
MIN_CHUNK_CHARS = 100  # Skip tiny chunks (imports-only, trivial)

# Types that are already natural language — no description needed
_SKIP_TYPES = {"documentation", "config"}

PROMPT_TEMPLATE = """Summarize this code in one sentence. Be concise - describe WHAT it does, not HOW.

```
{code}
```

One-sentence summary:"""

# --- Model management ---

_model = None
_tokenizer = None


def is_enabled(db_path: str | None = None) -> bool:
    """Check if NL descriptions are enabled.

    Enabled if CODE_RAG_DESCRIPTIONS=1 env var is set, OR if a descriptions.db
    cache already exists for the project (meaning descriptions were previously
    generated and can be reused without a full reindex).
    """
    if os.getenv("CODE_RAG_DESCRIPTIONS", "0") == "1":
        return True
    if db_path:
        cache_path = _get_cache_db_path(db_path)
        return Path(cache_path).exists()
    return False


def load_model():
    """Load Qwen3-4B-MLX-4bit for description generation."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    from mlx_lm import load

    print(f"Loading {MODEL_ID} for NL descriptions...")
    t0 = time.perf_counter()
    _model, _tokenizer = load(MODEL_ID, tokenizer_config={"trust_remote_code": False})
    elapsed = time.perf_counter() - t0
    print(f"Description model ready ({elapsed:.1f}s)")
    return _model, _tokenizer


def unload_model():
    """Free the description model from memory."""
    global _model, _tokenizer
    if _model is not None:
        _model = None
        _tokenizer = None
        # Trigger garbage collection to actually free the memory
        import gc
        import mlx.core as mx
        gc.collect()
        mx.clear_cache()
        print("Description model unloaded")


# --- Description generation ---

def generate_description(code: str) -> str:
    """Generate a one-sentence NL description for a code chunk."""
    from mlx_lm import generate

    model, tokenizer = load_model()
    prompt = PROMPT_TEMPLATE.format(code=code[:MAX_INPUT_CHARS])

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            # Models like Gemma 3 don't support enable_thinking
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
    else:
        formatted = prompt

    response = generate(
        model, tokenizer, prompt=formatted, max_tokens=MAX_GEN_TOKENS, verbose=False
    )
    return response.strip()


# --- SQLite cache ---

def _content_hash(content: str) -> str:
    """SHA256 hash of chunk content string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _get_cache_db_path(db_path: str) -> str:
    """Derive descriptions cache DB path from the Milvus DB path."""
    return str(Path(db_path).parent / "descriptions.db")


def _get_cache_connection(db_path: str) -> sqlite3.Connection:
    """Open (or create) the descriptions cache database."""
    cache_path = _get_cache_db_path(db_path)
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cache_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS descriptions (
            content_hash TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            created_at REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def get_cached(conn: sqlite3.Connection, content_hash: str) -> Optional[str]:
    """Look up a cached description by content hash."""
    row = conn.execute(
        "SELECT description FROM descriptions WHERE content_hash = ?",
        (content_hash,)
    ).fetchone()
    return row[0] if row else None


def put_cached(conn: sqlite3.Connection, content_hash: str, description: str):
    """Store a description in the cache."""
    conn.execute(
        "INSERT OR REPLACE INTO descriptions (content_hash, description, created_at) VALUES (?, ?, ?)",
        (content_hash, description, time.time())
    )
    conn.commit()


# --- Batch API ---

def describe_chunks(chunks: List[Dict], db_path: str) -> List[Optional[str]]:
    """Generate NL descriptions for a list of chunks.

    Returns a list parallel to the input: a description string or None
    (for skipped chunks like docs, config, or tiny code).

    Checks cache first, only generates for cache misses.
    """
    if not chunks:
        return []

    cache_conn = _get_cache_connection(db_path)
    descriptions: List[Optional[str]] = [None] * len(chunks)
    to_generate: List[int] = []  # Indices that need generation

    # Phase 1: Check cache, identify what needs generation
    for i, chunk in enumerate(chunks):
        chunk_type = chunk.get("type", "code")
        content = chunk.get("content", "")

        # Skip non-code types (already natural language)
        if chunk_type in _SKIP_TYPES:
            continue

        # Skip tiny chunks
        if len(content) < MIN_CHUNK_CHARS:
            continue

        h = _content_hash(content)
        cached = get_cached(cache_conn, h)
        if cached is not None:
            descriptions[i] = cached
        else:
            to_generate.append(i)

    # Phase 2: Generate descriptions for cache misses
    if to_generate:
        cache_hits = len(chunks) - len(to_generate) - sum(
            1 for i, c in enumerate(chunks)
            if c.get("type", "code") in _SKIP_TYPES or len(c.get("content", "")) < MIN_CHUNK_CHARS
        )
        logger.info(
            "Describing %d chunks (%d cached, %d skipped)",
            len(to_generate), cache_hits,
            len(chunks) - len(to_generate) - cache_hits
        )

        for idx in to_generate:
            content = chunks[idx]["content"]
            try:
                desc = generate_description(content)
                descriptions[idx] = desc
                put_cached(cache_conn, _content_hash(content), desc)
            except Exception as e:
                logger.warning("Description generation failed for chunk %d: %s", idx, e)

    cache_conn.close()
    return descriptions
