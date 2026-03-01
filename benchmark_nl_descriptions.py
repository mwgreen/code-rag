#!/usr/bin/env python3
"""
Benchmark: NL description generation for code chunks.

Compares indexing time with and without LLM-generated descriptions.
Uses Qwen3-4B-MLX-4bit for local generation on Apple Silicon.
"""

import time
import sys
import os
import statistics
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import chunking
import rag_milvus

# --- Config ---
MODEL_ID = "Qwen/Qwen3-4B-MLX-4bit"
TARGET_DIR = Path("/Users/matt.green/IdeaProjects/docker/repos/icm-internal/server")
NUM_FILES = 100
MAX_TOKENS = 150  # Short descriptions only

DESCRIPTION_PROMPT = """Summarize this code in one sentence. Be concise - describe WHAT it does, not HOW.

```
{code}
```

One-sentence summary:"""


def find_java_files(root: Path, limit: int) -> list[Path]:
    """Find Java files, skipping excluded dirs."""
    excluded = rag_milvus.get_excluded_dirs()
    files = []
    for f in root.rglob("*.java"):
        if excluded & set(f.parts):
            continue
        if f.stat().st_size > 512 * 1024:  # Skip huge files
            continue
        if rag_milvus.is_jaxb_generated(str(f)):
            continue
        files.append(f)
        if len(files) >= limit:
            break
    return files


def benchmark_chunking_only(files: list[Path]) -> dict:
    """Baseline: chunk files, measure time."""
    times = []
    total_chunks = 0
    for f in files:
        t0 = time.perf_counter()
        chunks = chunking.chunk_file(str(f))
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        total_chunks += len(chunks)
    return {
        "label": "Chunking only",
        "total_s": sum(times),
        "mean_ms": statistics.mean(times) * 1000,
        "median_ms": statistics.median(times) * 1000,
        "total_chunks": total_chunks,
    }


def benchmark_embed_only(files: list[Path]) -> dict:
    """Current pipeline: chunk + embed."""
    times = []
    total_chunks = 0
    for f in files:
        chunks = chunking.chunk_file(str(f))
        docs = [c["content"] for c in chunks]
        if not docs:
            continue
        t0 = time.perf_counter()
        rag_milvus.embed_texts(docs)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        total_chunks += len(chunks)
    return {
        "label": "Embed only (current)",
        "total_s": sum(times),
        "mean_ms": statistics.mean(times) * 1000,
        "median_ms": statistics.median(times) * 1000,
        "total_chunks": total_chunks,
    }


def load_qwen3():
    """Load Qwen3-4B for text generation. No remote code execution."""
    from mlx_lm import load

    print(f"Loading {MODEL_ID} (this downloads ~2.5GB on first run)...")
    t0 = time.perf_counter()
    model, tokenizer = load(MODEL_ID, tokenizer_config={"trust_remote_code": False})
    elapsed = time.perf_counter() - t0
    print(f"Qwen3 loaded in {elapsed:.1f}s")
    return model, tokenizer


def generate_description(model, tokenizer, code: str) -> str:
    """Generate a one-sentence NL description for a code chunk."""
    from mlx_lm import generate

    prompt = DESCRIPTION_PROMPT.format(code=code[:2000])  # Truncate long chunks

    # Use chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False,  # Qwen3 thinking mode — disable for speed
        )
    else:
        formatted = prompt

    response = generate(
        model, tokenizer, prompt=formatted, max_tokens=MAX_TOKENS, verbose=False
    )
    return response.strip()


def benchmark_describe_then_embed(files: list[Path], model, tokenizer) -> dict:
    """New pipeline: chunk + describe + embed."""
    gen_times = []
    embed_times = []
    total_chunks = 0
    sample_descriptions = []

    for i, f in enumerate(files):
        chunks = chunking.chunk_file(str(f))
        docs = [c["content"] for c in chunks]
        if not docs:
            continue

        # Generate descriptions
        t0 = time.perf_counter()
        descriptions = []
        for doc in docs:
            desc = generate_description(model, tokenizer, doc)
            descriptions.append(desc)
        gen_elapsed = time.perf_counter() - t0
        gen_times.append(gen_elapsed)

        # Embed (description + code combined)
        augmented = [f"{desc}\n\n{doc}" for desc, doc in zip(descriptions, docs)]
        t0 = time.perf_counter()
        rag_milvus.embed_texts(augmented)
        embed_elapsed = time.perf_counter() - t0
        embed_times.append(embed_elapsed)

        total_chunks += len(chunks)

        # Save a few samples
        if len(sample_descriptions) < 5:
            for desc, doc in zip(descriptions, docs):
                sample_descriptions.append({
                    "file": f.name,
                    "description": desc,
                    "code_preview": doc[:200],
                })
                if len(sample_descriptions) >= 5:
                    break

        if (i + 1) % 10 == 0:
            avg_gen = statistics.mean(gen_times) * 1000
            print(f"  [{i+1}/{len(files)}] avg gen: {avg_gen:.0f}ms/file")

    return {
        "label": "Describe + Embed (new)",
        "gen_total_s": sum(gen_times),
        "gen_mean_ms": statistics.mean(gen_times) * 1000,
        "gen_median_ms": statistics.median(gen_times) * 1000,
        "embed_total_s": sum(embed_times),
        "embed_mean_ms": statistics.mean(embed_times) * 1000,
        "total_s": sum(gen_times) + sum(embed_times),
        "total_chunks": total_chunks,
        "samples": sample_descriptions,
    }


def main():
    print(f"Finding {NUM_FILES} Java files in {TARGET_DIR}...")
    files = find_java_files(TARGET_DIR, NUM_FILES)
    print(f"Found {len(files)} files\n")

    if len(files) < NUM_FILES:
        print(f"Warning: only found {len(files)} files (wanted {NUM_FILES})")

    # Phase 1: Baseline — chunking only
    print("=== Phase 1: Chunking only ===")
    chunk_stats = benchmark_chunking_only(files)
    print(f"  {chunk_stats['total_chunks']} chunks from {len(files)} files")
    print(f"  Total: {chunk_stats['total_s']:.2f}s | Mean: {chunk_stats['mean_ms']:.1f}ms | Median: {chunk_stats['median_ms']:.1f}ms per file\n")

    # Phase 2: Current pipeline — chunk + embed
    print("=== Phase 2: Chunk + Embed (current pipeline) ===")
    embed_stats = benchmark_embed_only(files)
    print(f"  {embed_stats['total_chunks']} chunks")
    print(f"  Total: {embed_stats['total_s']:.2f}s | Mean: {embed_stats['mean_ms']:.1f}ms | Median: {embed_stats['median_ms']:.1f}ms per file\n")

    # Phase 3: New pipeline — chunk + describe + embed
    print("=== Phase 3: Chunk + Describe + Embed (new pipeline) ===")
    qwen_model, qwen_tokenizer = load_qwen3()
    desc_stats = benchmark_describe_then_embed(files, qwen_model, qwen_tokenizer)
    print(f"  {desc_stats['total_chunks']} chunks")
    print(f"  Generation: {desc_stats['gen_total_s']:.2f}s total | {desc_stats['gen_mean_ms']:.1f}ms mean per file")
    print(f"  Embedding:  {desc_stats['embed_total_s']:.2f}s total | {desc_stats['embed_mean_ms']:.1f}ms mean per file")
    print(f"  Combined:   {desc_stats['total_s']:.2f}s total\n")

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    slowdown = desc_stats["total_s"] / embed_stats["total_s"] if embed_stats["total_s"] > 0 else float("inf")
    print(f"  Current pipeline:  {embed_stats['total_s']:.1f}s for {len(files)} files")
    print(f"  With descriptions: {desc_stats['total_s']:.1f}s for {len(files)} files")
    print(f"  Slowdown factor:   {slowdown:.1f}x")
    print(f"  Extra time per file: {(desc_stats['gen_mean_ms']):.0f}ms")
    print()

    # Extrapolate to full index
    files_total = 8747
    current_est = (embed_stats["total_s"] / len(files)) * files_total
    new_est = (desc_stats["total_s"] / len(files)) * files_total
    print(f"  Estimated full index ({files_total} files):")
    print(f"    Current: {current_est/60:.0f} min")
    print(f"    With NL: {new_est/60:.0f} min")
    print()

    # Sample descriptions
    if desc_stats.get("samples"):
        print("=== Sample Descriptions ===")
        for s in desc_stats["samples"]:
            print(f"\n  File: {s['file']}")
            print(f"  Desc: {s['description']}")
            print(f"  Code: {s['code_preview'][:100]}...")


if __name__ == "__main__":
    main()
