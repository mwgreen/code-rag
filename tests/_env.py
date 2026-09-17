"""Test environment: import this before any code-rag module.

Points the embedding model at a fake config (64 dims) so rag_milvus never loads
MLX weights, disables NL descriptions, and provides a deterministic bag-of-words
embedder that keeps similar texts close in cosine space.
"""

import hashlib
import json
import math
import os
import pathlib
import re
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DIM = 64
FAKE_MODEL_DIR = pathlib.Path(tempfile.mkdtemp(prefix="code-rag-fake-model-"))
(FAKE_MODEL_DIR / "config.json").write_text(json.dumps({"model_type": "fake", "hidden_size": DIM}))

os.environ["EMBED_MODEL_PATH"] = str(FAKE_MODEL_DIR)
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ["CODE_RAG_DESCRIPTIONS"] = "0"
os.environ["CODE_RAG_LOG_LEVEL"] = os.environ.get("CODE_RAG_LOG_LEVEL", "WARNING")
os.environ.pop("CODE_RAG_PROFILE", None)
os.environ.pop("CODE_RAG_EMBED_MODEL", None)

_TOKEN = re.compile(r"[A-Za-z0-9_]+")

EMBED_CALLS = {"count": 0, "texts": 0}


def fake_embed(texts):
    """Deterministic hashed bag-of-words embedding, L2-normalized."""
    EMBED_CALLS["count"] += 1
    EMBED_CALLS["texts"] += len(texts)
    out = []
    for text in texts:
        vec = [0.0] * DIM
        for tok in _TOKEN.findall(text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % DIM] += 1.0
            vec[(h >> 8) % DIM] += 0.5
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        out.append([x / norm for x in vec])
    return out


def install_fake_embedder():
    import rag_milvus
    rag_milvus.embed_texts = fake_embed
    return rag_milvus


def write(path: pathlib.Path, content: str) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
