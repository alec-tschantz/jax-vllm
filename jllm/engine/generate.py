"""Paged-attention kernels: batched prefill + packed decode.

Two kernels cover the hot path:

- `prefill_step`: advances B prefilling slots by one `block_size` chunk each.
  Each row writes into one fresh physical block, gathers the full prefix via
  block_tables, and returns the next token from the last valid query position.
- `decode_step_cb`: advances B decoding slots by one token each.

Unused padded rows may target the sentinel block 0; those writes are harmless.

This module is model-family agnostic. It only depends on the generic
decoder-only interface defined in `jllm.model.common`.
"""
import equinox as eqx
from jax import Array
from jax import numpy as jnp

from ..model.common import (
    Attention,
    DecoderLayer,
    DecoderOnlyModel,
    apply_rope,
    attention_kernel_causal,
    embed,
    linear,
    maybe_qk_norm,
    rms_norm,
    rope_cos_sin,
    swiglu,
)
from .paged import (
    PagedCache,
    PagedLayerCache,
    gather_kv,
    scatter_kv_decode,
    scatter_kv_prefill_batch,
)

def attention_prefill(
    a: Attention,
    hidden: Array,          # [B, C, D]
    cos: Array,             # [B, C, D_head]
    sin: Array,
    layer: PagedLayerCache,
    q_pos: Array,           # [B, C] int32
    valid_tokens: Array,    # [B] int32
    block_tables: Array,    # [B, NB_MAX] int32
    phys_blocks: Array,     # [B] int32
) -> tuple[Array, PagedLayerCache]:
    B, C, _ = hidden.shape
    q = linear(a.q_proj, hidden).reshape(B, C, a.num_heads, a.head_dim).transpose(0, 2, 1, 3)
    k_new = linear(a.k_proj, hidden).reshape(B, C, a.num_kv_heads, a.head_dim).transpose(0, 2, 1, 3)
    v_new = linear(a.v_proj, hidden).reshape(B, C, a.num_kv_heads, a.head_dim).transpose(0, 2, 1, 3)
    q, k_new = maybe_qk_norm(a, q, k_new)
    q, k_new = apply_rope(q, k_new, cos, sin)

    k_scatter = k_new.transpose(0, 2, 1, 3)
    v_scatter = v_new.transpose(0, 2, 1, 3)
    new_layer = scatter_kv_prefill_batch(layer, k_scatter, v_scatter, phys_blocks)

    k_all, v_all = gather_kv(new_layer, block_tables)
    k_all = k_all.transpose(0, 2, 1, 3)
    v_all = v_all.transpose(0, 2, 1, 3)

    q_lens = valid_tokens.astype(jnp.int32)
    k_lens = q_pos[:, 0].astype(jnp.int32) + q_lens
    out = attention_kernel_causal(
        q,
        k_all,
        v_all,
        a.head_dim,
        a.num_heads,
        a.num_kv_heads,
        q_lens=q_lens,
        k_lens=k_lens,
    )
    query_mask = (jnp.arange(C, dtype=jnp.int32)[None, :] < q_lens[:, None])[:, None, :, None]
    out = jnp.where(query_mask, out, 0)
    out = out.transpose(0, 2, 1, 3).reshape(B, C, -1)
    return linear(a.o_proj, out), new_layer


def decoder_layer_prefill(
    d: DecoderLayer, hidden, cos, sin, layer, q_pos, valid_tokens, block_tables, phys_blocks,
):
    h, new_layer = attention_prefill(
        d.self_attn,
        rms_norm(d.input_layernorm, hidden),
        cos,
        sin,
        layer,
        q_pos,
        valid_tokens,
        block_tables,
        phys_blocks,
    )
    hidden = hidden + h
    hidden = hidden + swiglu(d.mlp, rms_norm(d.post_attention_layernorm, hidden))
    return hidden, new_layer


def prefill_step(
    m: DecoderOnlyModel,
    chunk_ids: Array,         # [B, C] int32
    cache: PagedCache,
    block_tables: Array,      # [B, NB_MAX] int32
    pos_starts: Array,        # [B] int32
    valid_tokens: Array,      # [B] int32
    last_token_idx: Array,    # [B] int32
    phys_blocks: Array,       # [B] int32
) -> tuple[Array, PagedCache]:
    _, C = chunk_ids.shape
    hidden = embed(m.embed_tokens, chunk_ids)
    q_pos = pos_starts[:, None] + jnp.arange(C, dtype=jnp.int32)[None, :]
    cos, sin = rope_cos_sin(m.rotary_emb, q_pos, hidden.dtype)
    new_layers: list[PagedLayerCache] = []
    for layer, layer_cache in zip(m.layers, cache.layers):
        hidden, layer_cache = decoder_layer_prefill(
            layer, hidden, cos, sin, layer_cache, q_pos, valid_tokens, block_tables, phys_blocks
        )
        new_layers.append(layer_cache)
    hidden = rms_norm(m.norm, hidden)
    last_hidden = hidden[jnp.arange(hidden.shape[0], dtype=jnp.int32), last_token_idx]
    next_toks = jnp.argmax(linear(m.lm_head, last_hidden), axis=-1).astype(jnp.int32)
    return next_toks, PagedCache(layers=new_layers, block_size=cache.block_size)


def attention_decode_cb(
    a: Attention,
    hidden: Array,
    cos: Array,
    sin: Array,
    layer: PagedLayerCache,
    positions: Array,       # [B] int32
    valid_rows: Array,      # [B] int32
    block_tables: Array,    # [B, NB_MAX] int32
    phys_block: Array,      # [B] int32
    slot_in_block: Array,   # [B] int32
) -> tuple[Array, PagedLayerCache]:
    B = hidden.shape[0]
    q = linear(a.q_proj, hidden).reshape(B, 1, a.num_heads, a.head_dim).transpose(0, 2, 1, 3)
    k_new = linear(a.k_proj, hidden).reshape(B, 1, a.num_kv_heads, a.head_dim).transpose(0, 2, 1, 3)
    v_new = linear(a.v_proj, hidden).reshape(B, 1, a.num_kv_heads, a.head_dim).transpose(0, 2, 1, 3)
    q, k_new = maybe_qk_norm(a, q, k_new)
    q, k_new = apply_rope(q, k_new, cos, sin)

    k_scatter = k_new.transpose(0, 2, 1, 3)
    v_scatter = v_new.transpose(0, 2, 1, 3)
    new_layer = scatter_kv_decode(layer, k_scatter, v_scatter, phys_block, slot_in_block)

    k_all, v_all = gather_kv(new_layer, block_tables)
    k_all = k_all.transpose(0, 2, 1, 3)
    v_all = v_all.transpose(0, 2, 1, 3)

    q_lens = valid_rows.astype(jnp.int32)
    k_lens = q_lens * (positions.astype(jnp.int32) + 1)
    out = attention_kernel_causal(
        q,
        k_all,
        v_all,
        a.head_dim,
        a.num_heads,
        a.num_kv_heads,
        q_lens=q_lens,
        k_lens=k_lens,
    )
    out = jnp.where(valid_rows[:, None, None, None].astype(bool), out, 0)
    out = out.transpose(0, 2, 1, 3).reshape(B, 1, -1)
    return linear(a.o_proj, out), new_layer


def decoder_layer_decode_cb(
    d: DecoderLayer,
    hidden,
    cos,
    sin,
    layer,
    positions,
    valid_rows,
    block_tables,
    phys_block,
    slot_in_block,
):
    h, new_layer = attention_decode_cb(
        d.self_attn, rms_norm(d.input_layernorm, hidden), cos, sin, layer,
        positions, valid_rows, block_tables, phys_block, slot_in_block,
    )
    hidden = hidden + h
    hidden = hidden + swiglu(d.mlp, rms_norm(d.post_attention_layernorm, hidden))
    return hidden, new_layer


def decode_step_cb(
    m: DecoderOnlyModel,
    last_tokens: Array,
    cache: PagedCache,
    positions: Array,
    valid_rows: Array,
    block_tables: Array,
    phys_block: Array,
    slot_in_block: Array,
) -> tuple[Array, PagedCache]:
    B = last_tokens.shape[0]
    hidden = embed(m.embed_tokens, last_tokens)
    q_pos = positions[:, None]
    cos, sin = rope_cos_sin(m.rotary_emb, q_pos, hidden.dtype)
    new_layers: list[PagedLayerCache] = []
    for layer, layer_cache in zip(m.layers, cache.layers):
        hidden, layer_cache = decoder_layer_decode_cb(
            layer,
            hidden,
            cos,
            sin,
            layer_cache,
            positions,
            valid_rows,
            block_tables,
            phys_block,
            slot_in_block,
        )
        new_layers.append(layer_cache)
    hidden = rms_norm(m.norm, hidden)
    return linear(m.lm_head, hidden), PagedCache(layers=new_layers, block_size=cache.block_size)


prefill_step_jit = eqx.filter_jit(prefill_step, donate="all-except-first")
decode_step_cb_jit = eqx.filter_jit(decode_step_cb, donate="all-except-first")
