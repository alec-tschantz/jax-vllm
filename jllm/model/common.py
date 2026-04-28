import os
from typing import Optional, Protocol, Sequence

import equinox as eqx
import jax
from jax import Array, nn
from jax import numpy as jnp





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
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin





class SwiGLU(eqx.Module):
    gate_proj: Linear
    up_proj: Linear
    down_proj: Linear


def swiglu(d: SwiGLU, x: Array) -> Array:
    return linear(d.down_proj, nn.silu(linear(d.gate_proj, x)) * linear(d.up_proj, x))





class Attention(eqx.Module):
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear
    q_norm: Optional[RMSNorm]
    k_norm: Optional[RMSNorm]
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





def _attn_einsum(q: Array, k: Array, v: Array, mask: Array, head_dim: int) -> Array:
    scale = 1.0 / jnp.sqrt(head_dim).astype(jnp.float32)
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k).astype(jnp.float32) * scale
    scores = jnp.where(mask, scores, -jnp.inf)
    probs = nn.softmax(scores, axis=-1).astype(v.dtype)
    return jnp.einsum("bhqk,bhkd->bhqd", probs, v)


def _attn_sdpa(q: Array, k: Array, v: Array, mask: Array, head_dim: int) -> Array:
    q_t = jnp.transpose(q, (0, 2, 1, 3))
    k_t = jnp.transpose(k, (0, 2, 1, 3))
    v_t = jnp.transpose(v, (0, 2, 1, 3))
    out = jax.nn.dot_product_attention(
        q_t, k_t, v_t, mask=mask, scale=1.0 / jnp.sqrt(head_dim).astype(q.dtype)
    )
    return jnp.transpose(out, (0, 2, 1, 3))


def _repeat_kv_heads(
    k: Array,
    v: Array,
    num_heads: int,
    num_kv_heads: int,
) -> tuple[Array, Array]:
    if num_kv_heads == num_heads:
        return k, v
    rep = num_heads // num_kv_heads
    return jnp.repeat(k, rep, axis=1), jnp.repeat(v, rep, axis=1)


def causal_mask_from_lens(q_lens: Array, k_lens: Array, max_q: int, max_k: int) -> Array:
    q_idx = jnp.arange(max_q, dtype=jnp.int32)[None, :, None]
    k_idx = jnp.arange(max_k, dtype=jnp.int32)[None, None, :]
    q_lens = q_lens.astype(jnp.int32)[:, None, None]
    k_lens = k_lens.astype(jnp.int32)[:, None, None]
    start = k_lens - q_lens
    q_valid = q_idx < q_lens
    k_valid = k_idx < k_lens
    causal = k_idx <= (start + q_idx)
    return (q_valid & k_valid & causal)[:, None, :, :]






ATTENTION_IMPL = os.environ.get("JLLM_ATTENTION_IMPL", "einsum").lower()


def attention_kernel_causal(
    q: Array,
    k: Array,
    v: Array,
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
    q_lens: Optional[Array] = None,
    k_lens: Optional[Array] = None,
) -> Array:
    if q_lens is None:
        q_lens = jnp.full((q.shape[0],), q.shape[2], dtype=jnp.int32)
    if k_lens is None:
        k_lens = jnp.full((k.shape[0],), k.shape[2], dtype=jnp.int32)

    mask = causal_mask_from_lens(q_lens, k_lens, q.shape[2], k.shape[2])
    return attention_kernel(q, k, v, mask, head_dim, num_heads, num_kv_heads)


def attention_kernel(
    q: Array, k: Array, v: Array, mask: Array, head_dim: int, num_heads: int, num_kv_heads: int
) -> Array:
    if ATTENTION_IMPL == "sdpa":

        return _attn_sdpa(q, k, v, mask, head_dim)
    k, v = _repeat_kv_heads(k, v, num_heads, num_kv_heads)
    return _attn_einsum(q, k, v, mask, head_dim)
