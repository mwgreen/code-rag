"""
Qwen2 embedding model for mlx-embeddings.

Adapted from qwen3.py. Key differences from Qwen3:
- Attention has bias on q/k/v projections (not o_proj)
- No QK normalization (q_norm/k_norm)
- attention_bias defaults to True
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, BaseModelOutput, normalize_embeddings


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


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "qwen2"
    hidden_size: int = 1536
    num_hidden_layers: int = 28
    intermediate_size: int = 8960
    num_attention_heads: int = 12
    num_key_value_heads: Optional[int] = 2
    head_dim: Optional[int] = None
    max_position_embeddings: int = 131072
    vocab_size: int = 151646

    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    rope_theta: float = 1000000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    # Qwen2: attention bias defaults to True (q/k/v have bias, o_proj does not)
    attention_bias: bool = True
    use_sliding_window: bool = False
    sliding_window: Optional[int] = None
    max_window_layers: Optional[int] = 21

    tie_word_embeddings: bool = False
    hidden_act: str = "silu"

    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None

    architectures: List[str] = field(default_factory=lambda: ["Qwen2Model"])
    initializer_range: float = 0.02

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


class Qwen2MLP(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2Attention(nn.Module):
    """Qwen2 attention: has bias on q/k/v projections, no QK normalization."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.rope_theta = config.rope_theta

        # q/k/v have bias=True, o_proj has bias=False (Qwen2 convention)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # No q_norm/k_norm in Qwen2 (unlike Qwen3)

        self.rotary_emb = nn.RoPE(self.head_dim, traditional=False, base=self.rope_theta)

    def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None, **kwargs) -> mx.array:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)

        # No QK norm in Qwen2

        query_states = self.rotary_emb(query_states)
        key_states = self.rotary_emb(key_states)

        if self.num_key_value_groups > 1:
            key_states = mx.repeat(key_states, self.num_key_value_groups, axis=1)
            value_states = mx.repeat(value_states, self.num_key_value_groups, axis=1)

        scale = 1.0 / math.sqrt(self.head_dim)

        try:
            attn_output = mx.fast.scaled_dot_product_attention(
                query_states, key_states, value_states, scale=scale, mask=attention_mask
            )
        except Exception as e:
            logging.warning(f"Fast attention failed, using fallback: {e}")
            attn_weights = (query_states @ key_states.transpose(0, 1, 3, 2)) * scale
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = mx.softmax(attn_weights, axis=-1)
            attn_output = attn_weights @ value_states

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.self_attn = Qwen2Attention(config)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None, **kwargs) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen2Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _create_causal_mask(self, seq_length: int, dtype: mx.Dtype) -> mx.array:
        mask = mx.tril(mx.ones((seq_length, seq_length), dtype=mx.bool_))
        mask = mx.where(mask, 0.0, -mx.inf).astype(dtype)
        return mx.expand_dims(mask, axis=(0, 1))

    def __call__(self, input_ids: mx.array, attention_mask: Optional[mx.array] = None, **kwargs) -> mx.array:
        batch_size, seq_length = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        if attention_mask is None:
            attention_mask = self._create_causal_mask(seq_length, hidden_states.dtype)
        elif attention_mask.ndim == 2:
            padding_mask = attention_mask[:, None, None, :]
            padding_mask = mx.where(padding_mask == 0, -mx.inf, 0.0).astype(hidden_states.dtype)
            causal_mask = self._create_causal_mask(seq_length, hidden_states.dtype)
            attention_mask = causal_mask + padding_mask

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        return self.norm(hidden_states)


class Model(nn.Module):
    """Qwen2 model for embedding generation with last-token pooling."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Qwen2Model(config)

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
