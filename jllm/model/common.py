"""Shared primitives used by any decoder-only model in jllm.

Data classes and compute functions that are common to Qwen2, Qwen3, and
similar decoder-only transformers. Model-specific modules (qwen2.py,
qwen3.py) compose these with their own `Attention` + `DecoderLayer` variants
and their own `{Arch}Config` / `{Arch}Model` wrappers.

Attention kernel selection is env-driven:

    JLLM_ATTENTION_IMPL = einsum (default) | sdpa | aiter

`einsum` is our reference path (bf16 params/acts, fp32 softmax + RMSNorm).
`sdpa` dispatches to `jax.nn.dot_product_attention`. `aiter` (when the PoC
lands) calls AMD's AITER flash-attention via jax_aiter.
"""
import os
from typing import Optional, Protocol, Sequence

import equinox as eqx
import jax
from jax import Array, nn
from jax import numpy as jnp


# ---------- data-only modules ----------


class Embedding(eqx.Module):
    weight: Array


class Linear(eqx.Module):
    weight: Array
    bias: Optional[Array]


class RMSNorm(eqx.Module):
    weight: Array
    eps: float


class RotaryEmbedding(eqx.Module):
    dim: int
    theta: float


class KVCacheConfig(Protocol):
    num_hidden_layers: int
    num_kv_heads: int
    head_dim: int


# ---------- primitives ----------


def embed(e: Embedding, ids: Array) -> Array:
    return jnp.take(e.weight, ids, axis=0)


def linear(l: Linear, x: Array) -> Array:
    y = x @ l.weight.T
    return y + l.bias if l.bias is not None else y


def rms_norm(r: RMSNorm, x: Array) -> Array:
    dtype = x.dtype
    x32 = x.astype(jnp.float32)
    inv_rms = jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + r.eps)
    return (x32 * inv_rms).astype(dtype) * r.weight


def rope_cos_sin(r: RotaryEmbedding, position_ids: Array, dtype) -> tuple[Array, Array]:
    inv_freq = 1.0 / (r.theta ** (jnp.arange(0, r.dim, 2, dtype=jnp.float32) / r.dim))
    freqs = position_ids.astype(jnp.float32)[..., None] * inv_freq
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


def _rotate_half(u: Array) -> Array:
    u1, u2 = jnp.split(u, 2, axis=-1)
    return jnp.concatenate([-u2, u1], axis=-1)


def apply_rope(q: Array, k: Array, cos: Array, sin: Array) -> tuple[Array, Array]:
    """q,k: [B, H, T, D]; cos,sin: [B, T, D]."""
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


# ---------- SwiGLU MLP (shared Qwen2/Qwen3/Llama) ----------


class SwiGLU(eqx.Module):
    gate_proj: Linear
    up_proj: Linear
    down_proj: Linear


def swiglu(d: SwiGLU, x: Array) -> Array:
    return linear(d.down_proj, nn.silu(linear(d.gate_proj, x)) * linear(d.up_proj, x))


# ---------- Attention + DecoderLayer (shared shape for Qwen2/Qwen3) ----------


class Attention(eqx.Module):
    """Attention block for decoder-only transformers.

    Fields present on all supported archs; Qwen2 leaves q_norm/k_norm as None
    (and carries biases on the Q/K/V projections), Qwen3 populates q_norm/k_norm
    and has bias-free projections. `Linear.bias` is already Optional so the
    bias policy is handled by the loader.
    """
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear
    q_norm: Optional[RMSNorm]   # Qwen3: per-head RMSNorm on Q (pre-RoPE). Qwen2: None.
    k_norm: Optional[RMSNorm]   # Qwen3: per-head RMSNorm on K (pre-RoPE). Qwen2: None.
    num_heads: int
    num_kv_heads: int
    head_dim: int


class DecoderLayer(eqx.Module):
    self_attn: Attention
    mlp: SwiGLU
    input_layernorm: RMSNorm
    post_attention_layernorm: RMSNorm


class DecoderOnlyModel(Protocol):
    embed_tokens: Embedding
    layers: Sequence[DecoderLayer]
    norm: RMSNorm
    rotary_emb: RotaryEmbedding
    lm_head: Linear
    cfg: KVCacheConfig


def maybe_qk_norm(a: Attention, q: Array, k: Array) -> tuple[Array, Array]:
    """Apply Qwen3-style RMSNorm to Q/K if present, else pass through."""
    if a.q_norm is not None:
        q = rms_norm(a.q_norm, q)
        k = rms_norm(a.k_norm, k)
    return q, k


def decoder_attention(a: Attention, hidden: Array, cos: Array, sin: Array, mask: Array) -> Array:
    B, T, _ = hidden.shape
    q = linear(a.q_proj, hidden).reshape(B, T, a.num_heads, a.head_dim).transpose(0, 2, 1, 3)
    k = linear(a.k_proj, hidden).reshape(B, T, a.num_kv_heads, a.head_dim).transpose(0, 2, 1, 3)
    v = linear(a.v_proj, hidden).reshape(B, T, a.num_kv_heads, a.head_dim).transpose(0, 2, 1, 3)
    q, k = maybe_qk_norm(a, q, k)
    q, k = apply_rope(q, k, cos, sin)
    out = attention_kernel(q, k, v, mask, a.head_dim, a.num_heads, a.num_kv_heads)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
    return linear(a.o_proj, out)


def decoder_layer_forward(d: DecoderLayer, hidden: Array, cos: Array, sin: Array, mask: Array) -> Array:
    hidden = hidden + decoder_attention(d.self_attn, rms_norm(d.input_layernorm, hidden), cos, sin, mask)
    hidden = hidden + swiglu(d.mlp, rms_norm(d.post_attention_layernorm, hidden))
    return hidden


def decoder_only_forward(
    m,
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
        hidden = decoder_layer_forward(layer, hidden, cos, sin, mask)
    hidden = rms_norm(m.norm, hidden)
    return linear(m.lm_head, hidden)


# ---------- attention kernel (shared across archs) ----------


def _attn_einsum(q: Array, k: Array, v: Array, mask: Array, head_dim: int) -> Array:
    """q,k,v in [B, H, T, D] with H matched (caller repeats K/V for GQA)."""
    scale = 1.0 / jnp.sqrt(head_dim).astype(jnp.float32)
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k).astype(jnp.float32) * scale
    scores = jnp.where(mask, scores, -jnp.inf)
    probs = nn.softmax(scores, axis=-1).astype(v.dtype)
    return jnp.einsum("bhqk,bhkd->bhqd", probs, v)


def _attn_sdpa(q: Array, k: Array, v: Array, mask: Array, head_dim: int) -> Array:
    """jax.nn.dot_product_attention. Accepts K/V with num_kv_heads (< num_heads);
    SDPA handles GQA internally. Inputs here are [B, H, T, D]; internally transposed."""
    q_t = jnp.transpose(q, (0, 2, 1, 3))  # [B, T, H_q, D]
    k_t = jnp.transpose(k, (0, 2, 1, 3))  # [B, T, H_kv, D]
    v_t = jnp.transpose(v, (0, 2, 1, 3))
    out = jax.nn.dot_product_attention(
        q_t, k_t, v_t, mask=mask, scale=1.0 / jnp.sqrt(head_dim).astype(q.dtype)
    )
    return jnp.transpose(out, (0, 2, 1, 3))


# A/B on MI300X + ROCm JAX 0.9.2: einsum + jnp.repeat beats SDPA by ~5-13% at
# B=1..4 and ties at B=16 — keep einsum as default. Read once at import time
# via JLLM_ATTENTION_IMPL; no hot-swap at runtime because switching under JIT
# would recompile every kernel.
ATTENTION_IMPL = os.environ.get("JLLM_ATTENTION_IMPL", "einsum")


def attention_kernel(
    q: Array, k: Array, v: Array, mask: Array, head_dim: int, num_heads: int, num_kv_heads: int
) -> Array:
    if ATTENTION_IMPL == "sdpa":
        # SDPA handles GQA natively.
        return _attn_sdpa(q, k, v, mask, head_dim)
    # einsum path (also current fallback for "aiter" until the PoC wire-up lands).
    # einsum needs matched head counts.
    if num_kv_heads != num_heads:
        rep = num_heads // num_kv_heads
        k = jnp.repeat(k, rep, axis=1)
        v = jnp.repeat(v, rep, axis=1)
    return _attn_einsum(q, k, v, mask, head_dim)
