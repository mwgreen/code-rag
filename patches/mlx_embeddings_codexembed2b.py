"""
CodexEmbed-2B embedding model for mlx-embeddings.

Gemma 2 architecture for Salesforce/SFR-Embedding-Code-2B_R.
The SFR model's config.json has model_type: "codexembed2b", so mlx-embeddings
dispatches to this file.

Imports building blocks from mlx_lm.models.gemma2 (same pattern as
mlx_embeddings/models/gemma3_text.py). Uses last-token pooling and L2
normalization, matching SFR's own implementation.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gemma2 import ModelArgs, RMSNorm, TransformerBlock

from .base import BaseModelOutput, normalize_embeddings


def last_token_pool(
    last_hidden_states: mx.array, attention_mask: Optional[mx.array] = None
) -> mx.array:
    if attention_mask is None:
        return last_hidden_states[:, -1]

    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(axis=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[mx.arange(batch_size), sequence_lengths]


class Gemma2EmbedModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.config = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args=args) for _ in range(args.num_hidden_layers)]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def _create_causal_mask(self, seq_length: int, dtype: mx.Dtype) -> mx.array:
        # Gemma 2 GQA reshapes scores to 5D (B, n_kv_heads, repeats, L, L)
        # when repeats > 1. Mask must be (1, 1, 1, L, L) to broadcast correctly.
        mask = mx.tril(mx.ones((seq_length, seq_length), dtype=mx.bool_))
        mask = mx.where(mask, 0.0, -mx.inf).astype(dtype)
        return mask.reshape(1, 1, 1, seq_length, seq_length)

    def __call__(self, input_ids: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        batch_size, seq_length = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states * (self.config.hidden_size ** 0.5)

        if attention_mask is None:
            mask = self._create_causal_mask(seq_length, hidden_states.dtype)
        elif attention_mask.ndim == 2:
            # Padding mask: (B, 1, 1, 1, L) to broadcast with 5D GQA scores
            padding_mask = attention_mask[:, None, None, None, :]
            padding_mask = mx.where(padding_mask == 0, -mx.inf, 0.0).astype(hidden_states.dtype)
            causal_mask = self._create_causal_mask(seq_length, hidden_states.dtype)
            mask = causal_mask + padding_mask
        else:
            mask = attention_mask

        for layer in self.layers:
            hidden_states = layer(hidden_states, mask)

        return self.norm(hidden_states)


class Model(nn.Module):
    """CodexEmbed-2B (Gemma 2) model for embedding generation with last-token pooling."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Gemma2EmbedModel(config)

    def __call__(self, input_ids: mx.array, attention_mask: Optional[mx.array] = None) -> BaseModelOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be 2D, got shape {input_ids.shape}")

        batch_size, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)

        last_hidden_state = self.model(input_ids, attention_mask=attention_mask)
        pooled_output = last_token_pool(last_hidden_state, attention_mask)
        text_embeds = normalize_embeddings(pooled_output)

        return BaseModelOutput(text_embeds=text_embeds, last_hidden_state=last_hidden_state)

    def sanitize(self, weights: dict) -> dict:
        sanitized = {}
        for key, value in weights.items():
            if "lm_head.weight" in key:
                continue
            if key.startswith("model."):
                new_key = key
            elif not key.startswith("model.") and "." in key:
                new_key = f"model.{key}"
            else:
                new_key = key
            sanitized[new_key] = value
        return sanitized
