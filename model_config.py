"""
Model registry and resolution for code-rag.

Two models are configurable:
  - Embedding model (mlx-embeddings): turns code/text chunks and queries into vectors.
  - Description model (mlx-lm): writes one-sentence NL summaries of code chunks.

Resolution order for each model (first match wins):
  1. Direct override:   EMBED_MODEL_PATH / CODE_RAG_DESCRIPTION_MODEL
  2. Named model key:   CODE_RAG_EMBED_MODEL / CODE_RAG_DESCRIPTION_MODEL_KEY
  3. Profile:           CODE_RAG_PROFILE (high | medium | low | max | legacy)
  4. Auto-detect:       first model in preference order that is already downloaded
  5. Default profile:   DEFAULT_PROFILE

Step 4 means a machine that only has the older models downloaded keeps working
after a code update, while a fresh machine picks up the new defaults once
./download-embed-model.sh and ./download-description-model.sh have run.

Run `python model_config.py` to print the resolved configuration.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is in requirements
    pass

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"

# --- Embedding models (local Q8 MLX conversions under models/) ---

EMBED_MODELS: Dict[str, dict] = {
    "qwen3-embed-4b": {
        "hf_id": "Qwen/Qwen3-Embedding-4B",
        "local_dir": "qwen3-embed-4b-mlx-q8",
        "model_type": "qwen3",
        "dims": 2560,
        "size_gb": 4.5,
        "license": "Apache-2.0",
        "patch": None,
        "notes": "Best quality. General-purpose embedder that tops open code-retrieval benchmarks.",
    },
    "qwen3-embed-0.6b": {
        "hf_id": "Qwen/Qwen3-Embedding-0.6B",
        "local_dir": "qwen3-embed-0.6b-mlx-q8",
        "model_type": "qwen3",
        "dims": 1024,
        "size_gb": 0.7,
        "license": "Apache-2.0",
        "patch": None,
        "notes": "Fast. Near the 4B on Java/C++ retrieval at ~10x the throughput.",
    },
    "sfr-embed-code-2b": {
        "hf_id": "Salesforce/SFR-Embedding-Code-2B_R",
        "local_dir": "sfr-embed-code-2b-mlx-q8",
        "model_type": "codexembed2b",
        "dims": 2304,
        "size_gb": 2.6,
        "license": "CC-BY-NC-4.0",
        "patch": "codexembed2b",
        "notes": "Legacy default (Gemma 2 base). Needs patches/mlx_embeddings_codexembed2b.py.",
    },
    "qodo-embed-1.5b": {
        "hf_id": "Qodo/Qodo-Embed-1-1.5B",
        "local_dir": "qodo-embed-1-1.5b-mlx-q8",
        "model_type": "qwen2",
        "dims": 1536,
        "size_gb": 1.6,
        "license": "OpenRAIL++-M",
        "patch": "qwen2",
        "notes": "Legacy alternative (Qwen2 base). Needs patches/mlx_embeddings_qwen2.py.",
    },
}

# Auto-detect preference when nothing is configured: best first.
EMBED_PREFERENCE: List[str] = ["qwen3-embed-4b", "qwen3-embed-0.6b", "sfr-embed-code-2b", "qodo-embed-1.5b"]

# Query-side instruction prefix keyed by model_type (read from the model's config.json).
# Documents are always embedded raw. Empty string means the model is symmetric.
QUERY_INSTRUCTIONS: Dict[str, str] = {
    # Qwen's documented format: "Instruct: {task}\nQuery:{query}" (no space after the colon).
    "qwen3": (
        "Instruct: Given a natural language question about a codebase, "
        "retrieve relevant code snippets and documentation\nQuery:"
    ),
    "codexembed2b": "Instruct: Given Code or Text, retrieval relevant content\nQuery: ",
}

# --- Description (generator) models (HuggingFace IDs loaded by mlx-lm) ---

DESCRIPTION_MODELS: Dict[str, dict] = {
    "gemma-4-e4b": {
        "hf_id": "mlx-community/gemma-4-e4b-it-OptiQ-4bit",
        "size_gb": 7.5,
        "license": "Apache-2.0",
        "min_mlx_lm": "0.31.2",
        "notes": "Recommended. Best 4B-tier model for code understanding; ~84 tok/s on M5 Pro.",
    },
    "gemma-4-e2b": {
        "hf_id": "mlx-community/gemma-4-e2b-it-OptiQ-4bit",
        "size_gb": 4.0,
        "license": "Apache-2.0",
        "min_mlx_lm": "0.31.2",
        "notes": "Lighter Gemma 4 for 16 GB machines.",
    },
    "qwen3.6-35b-a3b": {
        "hf_id": "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit",
        "size_gb": 22.0,
        "license": "Apache-2.0",
        "min_mlx_lm": "0.31.3",
        "notes": "Highest quality. MoE with 3B active params; needs ~22 GB free RAM while indexing.",
    },
    "qwen3-4b-2507": {
        "hf_id": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
        "size_gb": 2.5,
        "license": "Apache-2.0",
        "min_mlx_lm": "0.26.0",
        "notes": "Small and solid. Works on older mlx-lm; no thinking mode.",
    },
    "gemma-3-4b": {
        "hf_id": "mlx-community/gemma-3-4b-it-4bit",
        "size_gb": 2.5,
        "license": "Gemma",
        "min_mlx_lm": "0.22.0",
        "notes": "Legacy default.",
    },
    "qwen3-4b": {
        "hf_id": "Qwen/Qwen3-4B-MLX-4bit",
        "size_gb": 2.5,
        "license": "Apache-2.0",
        "min_mlx_lm": "0.24.0",
        "notes": "Legacy alternative.",
    },
}

DESCRIPTION_PREFERENCE: List[str] = [
    "gemma-4-e4b", "qwen3.6-35b-a3b", "gemma-4-e2b", "qwen3-4b-2507", "gemma-3-4b", "qwen3-4b",
]

# --- Profiles: (embed key, description key) ---

PROFILES: Dict[str, dict] = {
    "max": {
        "embed": "qwen3-embed-4b", "description": "qwen3.6-35b-a3b",
        "ram": "64 GB+", "notes": "Best descriptions; ~29 GB peak while indexing.",
    },
    "high": {
        "embed": "qwen3-embed-4b", "description": "gemma-4-e4b",
        "ram": "48 GB", "notes": "Recommended. ~12 GB peak while indexing, ~5 GB serving.",
    },
    "medium": {
        "embed": "qwen3-embed-0.6b", "description": "gemma-4-e4b",
        "ram": "24-32 GB", "notes": "Fast indexing, 1024-dim vectors.",
    },
    "low": {
        "embed": "qwen3-embed-0.6b", "description": "gemma-4-e2b",
        "ram": "16 GB", "notes": "Smallest footprint with descriptions on.",
    },
    "legacy": {
        "embed": "sfr-embed-code-2b", "description": "gemma-3-4b",
        "ram": "32 GB", "notes": "Pre-Sept-2026 defaults. Requires the mlx-embeddings patches.",
    },
}

DEFAULT_PROFILE = "high"


# --- Helpers ---

def _profile() -> Optional[str]:
    name = os.getenv("CODE_RAG_PROFILE", "").strip().lower()
    if not name:
        return None
    if name not in PROFILES:
        raise ValueError(f"Unknown CODE_RAG_PROFILE={name!r}. Choose from: {', '.join(PROFILES)}")
    return name


def embed_model_dir(key: str) -> Path:
    if key not in EMBED_MODELS:
        raise ValueError(f"Unknown embedding model key {key!r}. Choose from: {', '.join(EMBED_MODELS)}")
    return MODELS_DIR / EMBED_MODELS[key]["local_dir"]


def embed_model_downloaded(key: str) -> bool:
    d = embed_model_dir(key)
    return (d / "model.safetensors").exists() or (d / "model-00001-of-00002.safetensors").exists()


def hf_cache_dir() -> Path:
    if os.getenv("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.getenv("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def hf_model_cached(hf_id: str) -> bool:
    return (hf_cache_dir() / f"models--{hf_id.replace('/', '--')}").is_dir()


def description_model_downloaded(key: str) -> bool:
    if key not in DESCRIPTION_MODELS:
        raise ValueError(f"Unknown description model key {key!r}. Choose from: {', '.join(DESCRIPTION_MODELS)}")
    return hf_model_cached(DESCRIPTION_MODELS[key]["hf_id"])


# --- Resolution ---

def resolve_embed_model_key() -> Optional[str]:
    """Which registry key would be used (None if EMBED_MODEL_PATH points elsewhere)."""
    if os.getenv("EMBED_MODEL_PATH"):
        return None
    key = os.getenv("CODE_RAG_EMBED_MODEL", "").strip()
    if key:
        embed_model_dir(key)  # validate
        return key
    profile = _profile()
    if profile:
        return PROFILES[profile]["embed"]
    for candidate in EMBED_PREFERENCE:
        if embed_model_downloaded(candidate):
            return candidate
    return PROFILES[DEFAULT_PROFILE]["embed"]


def resolve_embed_model_path() -> str:
    override = os.getenv("EMBED_MODEL_PATH")
    if override:
        return override
    return str(embed_model_dir(resolve_embed_model_key()))


def resolve_description_model_key() -> Optional[str]:
    if os.getenv("CODE_RAG_DESCRIPTION_MODEL"):
        return None
    key = os.getenv("CODE_RAG_DESCRIPTION_MODEL_KEY", "").strip()
    if key:
        description_model_downloaded(key)  # validate
        return key
    profile = _profile()
    if profile:
        return PROFILES[profile]["description"]
    for candidate in DESCRIPTION_PREFERENCE:
        if description_model_downloaded(candidate):
            return candidate
    return PROFILES[DEFAULT_PROFILE]["description"]


def resolve_description_model_id() -> str:
    override = os.getenv("CODE_RAG_DESCRIPTION_MODEL")
    if override:
        return override
    return DESCRIPTION_MODELS[resolve_description_model_key()]["hf_id"]


def query_instruction_for_model(model_path: str) -> str:
    """Query instruction prefix based on model_type in the model's config.json."""
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return ""
    try:
        with open(config_path) as f:
            model_type = json.load(f).get("model_type", "")
    except (OSError, json.JSONDecodeError):
        return ""
    for prefix, instruction in QUERY_INSTRUCTIONS.items():
        if model_type.startswith(prefix):
            return instruction
    return ""


def summary() -> dict:
    """Resolved configuration, for /health and the CLI."""
    embed_key = resolve_embed_model_key()
    desc_key = resolve_description_model_key()
    return {
        "profile": _profile() or "auto",
        "embed_model_key": embed_key,
        "embed_model_path": resolve_embed_model_path(),
        "embed_model_downloaded": embed_model_downloaded(embed_key) if embed_key else Path(resolve_embed_model_path()).exists(),
        "description_model_key": desc_key,
        "description_model_id": resolve_description_model_id(),
        "description_model_downloaded": hf_model_cached(resolve_description_model_id()),
    }


# --- CLI (used by the download scripts and setup.sh) ---

def _main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description="Show or query code-rag model configuration.")
    p.add_argument("--embed-key", action="store_true", help="print resolved embedding model key")
    p.add_argument("--embed-info", metavar="KEY", help="print 'hf_id local_dir patch' for an embedding model key")
    p.add_argument("--description-key", action="store_true", help="print resolved description model key")
    p.add_argument("--description-info", metavar="KEY", help="print 'hf_id size_gb min_mlx_lm' for a description model key")
    p.add_argument("--list", action="store_true", help="list profiles and models")
    p.add_argument("--json", action="store_true", help="print resolved config as JSON")
    a = p.parse_args(argv)

    if a.embed_key:
        print(resolve_embed_model_key() or "")
        return 0
    if a.embed_info:
        m = EMBED_MODELS[a.embed_info]
        print(m["hf_id"], m["local_dir"], m["patch"] or "none")
        return 0
    if a.description_key:
        print(resolve_description_model_key() or "")
        return 0
    if a.description_info:
        m = DESCRIPTION_MODELS[a.description_info]
        print(m["hf_id"], m["size_gb"], m["min_mlx_lm"])
        return 0
    if a.json:
        print(json.dumps(summary(), indent=2))
        return 0

    s = summary()
    print("Resolved configuration")
    print(f"  profile:            {s['profile']}")
    print(f"  embedding model:    {s['embed_model_key'] or '(EMBED_MODEL_PATH override)'}")
    print(f"    path:             {s['embed_model_path']}  [{'downloaded' if s['embed_model_downloaded'] else 'NOT downloaded'}]")
    print(f"  description model:  {s['description_model_key'] or '(CODE_RAG_DESCRIPTION_MODEL override)'}")
    print(f"    hf id:            {s['description_model_id']}  [{'cached' if s['description_model_downloaded'] else 'NOT cached'}]")
    if a.list:
        print("\nProfiles (CODE_RAG_PROFILE)")
        for name, prof in PROFILES.items():
            marker = " (default)" if name == DEFAULT_PROFILE else ""
            print(f"  {name:<8} {prof['ram']:<9} embed={prof['embed']:<18} description={prof['description']:<16} {prof['notes']}{marker}")
        print("\nEmbedding models (CODE_RAG_EMBED_MODEL)")
        for key, m in EMBED_MODELS.items():
            state = "downloaded" if embed_model_downloaded(key) else "-"
            print(f"  {key:<18} {m['dims']:>4} dims  {m['size_gb']:>4} GB  {m['license']:<14} {state:<10} {m['notes']}")
        print("\nDescription models (CODE_RAG_DESCRIPTION_MODEL_KEY)")
        for key, m in DESCRIPTION_MODELS.items():
            state = "cached" if description_model_downloaded(key) else "-"
            print(f"  {key:<18} {m['size_gb']:>4} GB  {m['license']:<11} mlx-lm>={m['min_mlx_lm']:<7} {state:<10} {m['notes']}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
