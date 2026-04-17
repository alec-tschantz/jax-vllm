"""Qwen3 / Qwen3.5 wiring.

Uses the shared `Attention` / `DecoderLayer` types from common.py. Qwen3
populates `q_norm` and `k_norm` (Qwen2 leaves them None) and has bias-free
Q/K/V projections. Weight loading lives in `weights.py::load_qwen3`.
"""
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
    apply_rope,
    attention_kernel,
    embed,
    linear,
    maybe_qk_norm,
    rms_norm,
    rope_cos_sin,
    swiglu,
)


@dataclass(frozen=True)
class Qwen3Config:
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
    def from_hf(cls, path: "str | Path") -> "Qwen3Config":
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
            rope_theta=c.get("rope_theta", 1_000_000.0),
            max_position_embeddings=c["max_position_embeddings"],
            tie_word_embeddings=c.get("tie_word_embeddings", False),
        )


class Qwen3Model(eqx.Module):
    embed_tokens: Embedding
    layers: list[DecoderLayer]
    norm: RMSNorm
    rotary_emb: RotaryEmbedding
    lm_head: Linear
    cfg: Qwen3Config


def attention(a: Attention, hidden: Array, cos: Array, sin: Array, mask: Array) -> Array:
    B, T, _ = hidden.shape
    q = linear(a.q_proj, hidden).reshape(B, T, a.num_heads, a.head_dim).transpose(0, 2, 1, 3)
    k = linear(a.k_proj, hidden).reshape(B, T, a.num_kv_heads, a.head_dim).transpose(0, 2, 1, 3)
    v = linear(a.v_proj, hidden).reshape(B, T, a.num_kv_heads, a.head_dim).transpose(0, 2, 1, 3)
    q, k = maybe_qk_norm(a, q, k)  # Qwen3 applies q_norm / k_norm here
    q, k = apply_rope(q, k, cos, sin)
    out = attention_kernel(q, k, v, mask, a.head_dim, a.num_heads, a.num_kv_heads)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
    return linear(a.o_proj, out)


def decoder_layer(d: DecoderLayer, hidden: Array, cos: Array, sin: Array, mask: Array) -> Array:
    hidden = hidden + attention(d.self_attn, rms_norm(d.input_layernorm, hidden), cos, sin, mask)
    hidden = hidden + swiglu(d.mlp, rms_norm(d.post_attention_layernorm, hidden))
    return hidden


def forward(
    m: Qwen3Model,
    input_ids: Array,
    attention_mask: Optional[Array] = None,
    position_ids: Optional[Array] = None,
) -> Array:
    B, T = input_ids.shape
    if position_ids is None:
        position_ids = jnp.broadcast_to(jnp.arange(T), (B, T))
    if attention_mask is None:
        attention_mask = jnp.ones((B, T), dtype=bool)

    causal = jnp.tril(jnp.ones((T, T), dtype=bool))
    mask = causal[None, None, :, :] & attention_mask.astype(bool)[:, None, None, :]

    hidden = embed(m.embed_tokens, input_ids)
    cos, sin = rope_cos_sin(m.rotary_emb, position_ids, hidden.dtype)
    for layer in m.layers:
        hidden = decoder_layer(layer, hidden, cos, sin, mask)
    hidden = rms_norm(m.norm, hidden)
    return linear(m.lm_head, hidden)
