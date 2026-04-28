import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import equinox as eqx
from jax import Array
from jax import numpy as jnp

from .common import (
    Attention,
    DecoderLayer,
    Embedding,
    Linear,
    RMSNorm,
    RotaryEmbedding,
    decoder_attention,
    decoder_layer_forward,
    decoder_only_forward,
)


@dataclass(frozen=True)
class Qwen2Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool

    @classmethod
    def from_hf(cls, path: "str | Path") -> "Qwen2Config":
        p = Path(path)
        if p.is_dir():
            p = p / "config.json"
        c = json.loads(p.read_text())
        n_heads = c["num_attention_heads"]
        return cls(
            vocab_size=c["vocab_size"],
            hidden_size=c["hidden_size"],
            intermediate_size=c["intermediate_size"],
            num_hidden_layers=c["num_hidden_layers"],
            num_heads=n_heads,
            num_kv_heads=c.get("num_key_value_heads", n_heads),
            head_dim=c.get("head_dim", c["hidden_size"] // n_heads),
            rms_norm_eps=c["rms_norm_eps"],
            rope_theta=c.get("rope_theta", 10000.0),
            max_position_embeddings=c["max_position_embeddings"],
            tie_word_embeddings=c.get("tie_word_embeddings", False),
        )


class Qwen2Model(eqx.Module):
    embed_tokens: Embedding
    layers: list[DecoderLayer]
    norm: RMSNorm
    rotary_emb: RotaryEmbedding
    lm_head: Linear
    cfg: Qwen2Config


def attention(a: Attention, hidden: Array, cos: Array, sin: Array, mask: Array) -> Array:
    return decoder_attention(a, hidden, cos, sin, mask)


def decoder_layer(d: DecoderLayer, hidden: Array, cos: Array, sin: Array, mask: Array) -> Array:
    return decoder_layer_forward(d, hidden, cos, sin, mask)


def forward(
    m: Qwen2Model,
    input_ids: Array,
    attention_mask: Optional[Array] = None,
    position_ids: Optional[Array] = None,
) -> Array:
    return decoder_only_forward(m, input_ids, attention_mask, position_ids)
