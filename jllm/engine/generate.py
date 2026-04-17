"""Paged-attention kernels: extend_step (chunked prefill) + decode_step_cb.

One KV world: a pool of fixed-size blocks (see paged.py) indexed via
`block_tables[slot, NB_MAX]`. Two kernels:

- `extend_step`: advances one slot by `chunk_size == block_size` tokens.
  Writes Q*block_size tokens' K/V into ONE physical block, gathers the full
  prefix via block_tables, computes attention with causal + real-mask, returns
  logits for every query position so the caller can pick the last real token
  on the final chunk.

- `decode_step_cb`: one token per slot across all B decoding slots. Writes at
  (phys_block[b], slot_in_block[b]), gathers prefix, masks by positions[b].

Both rely on the sentinel-block-0 convention: unused block_table entries point
at block 0 and the positional mask (`k_pos <= positions[b]`) drops them.
"""
import equinox as eqx
from jax import Array
from jax import numpy as jnp

from ..model.qwen2 import (
    Attention,
    DecoderLayer,
    Qwen2Model,
    apply_rope,
    attention_kernel,
    dense,
    embed,
    linear,
    rms_norm,
    rope_cos_sin,
)
from .paged import PagedCache, PagedLayerCache, gather_kv, scatter_kv_decode, scatter_kv_prefill


# ---------- chunked prefill (one slot, one chunk) ----------


def attention_extend(
    a: Attention,
    hidden: Array,          # [1, C, D]
    cos: Array,             # [1, C, D_head]
    sin: Array,
    layer: PagedLayerCache,
    q_pos: Array,           # [1, C] int32 — absolute positions of chunk tokens
    block_table_row: Array, # [NB_MAX] int32
    phys_block: Array,      # scalar int32 — the one physical block this chunk writes into
) -> tuple[Array, PagedLayerCache]:
    _, C, _ = hidden.shape
    q = linear(a.q_proj, hidden).reshape(1, C, a.num_heads, a.head_dim).transpose(0, 2, 1, 3)
    k_new = linear(a.k_proj, hidden).reshape(1, C, a.num_kv_heads, a.head_dim).transpose(0, 2, 1, 3)
    v_new = linear(a.v_proj, hidden).reshape(1, C, a.num_kv_heads, a.head_dim).transpose(0, 2, 1, 3)
    q, k_new = apply_rope(q, k_new, cos, sin)

    # Scatter this chunk's C tokens into phys_block[0..C-1]. scatter_kv_prefill
    # expects [1, T, H_kv, D] (heads-third) with T a multiple of block_size; we
    # pass block_indices = [phys_block] so nb = 1 and T = block_size = C.
    k_scatter = k_new.transpose(0, 2, 1, 3)
    v_scatter = v_new.transpose(0, 2, 1, 3)
    new_layer = scatter_kv_prefill(layer, k_scatter, v_scatter, phys_block[None])

    # Gather the full prefix (incl. this chunk which we just wrote).
    k_all, v_all = gather_kv(new_layer, block_table_row[None, :])
    k_all = k_all.transpose(0, 2, 1, 3)
    v_all = v_all.transpose(0, 2, 1, 3)

    # Causal only. Padding-Q rows always attend to at least their own position
    # (never all-False, so no NaN from softmax); their outputs are discarded.
    L = k_all.shape[2]
    k_pos = jnp.arange(L)
    q_abs = q_pos[0]
    mask = k_pos[None, None, None, :] <= q_abs[None, None, :, None]

    out = attention_kernel(q, k_all, v_all, mask, a.head_dim, a.num_heads, a.num_kv_heads)
    out = out.transpose(0, 2, 1, 3).reshape(1, C, -1)
    return linear(a.o_proj, out), new_layer


def decoder_layer_extend(d: DecoderLayer, hidden, cos, sin, layer, q_pos, block_table_row, phys_block):
    h, new_layer = attention_extend(
        d.self_attn, rms_norm(d.input_layernorm, hidden), cos, sin, layer,
        q_pos, block_table_row, phys_block,
    )
    hidden = hidden + h
    hidden = hidden + dense(d.mlp, rms_norm(d.post_attention_layernorm, hidden))
    return hidden, new_layer


def extend_step(
    m: Qwen2Model,
    chunk_ids: Array,         # [1, C] int32
    cache: PagedCache,
    block_table_row: Array,   # [NB_MAX] int32
    pos_start: Array,         # scalar int32 — absolute position of chunk_ids[0]
    phys_block: Array,        # scalar int32 — physical block this chunk writes into
) -> tuple[Array, PagedCache]:
    _, C = chunk_ids.shape
    hidden = embed(m.embed_tokens, chunk_ids)
    q_pos = (pos_start + jnp.arange(C, dtype=jnp.int32))[None, :]
    cos, sin = rope_cos_sin(m.rotary_emb, q_pos, hidden.dtype)
    new_layers: list[PagedLayerCache] = []
    for layer, lc in zip(m.layers, cache.layers):
        hidden, lc = decoder_layer_extend(
            layer, hidden, cos, sin, lc, q_pos, block_table_row, phys_block
        )
        new_layers.append(lc)
    hidden = rms_norm(m.norm, hidden)
    logits = linear(m.lm_head, hidden)  # [1, C, V]
    return logits, PagedCache(layers=new_layers, block_size=cache.block_size)


# ---------- fused continuous-batching decode ----------


def attention_decode_cb(
    a: Attention,
    hidden: Array,
    cos: Array,
    sin: Array,
    layer: PagedLayerCache,
    positions: Array,       # [B] int32
    block_tables: Array,    # [B, NB_MAX] int32
    phys_block: Array,      # [B] int32
    slot_in_block: Array,   # [B] int32
) -> tuple[Array, PagedLayerCache]:
    B = hidden.shape[0]
    q = linear(a.q_proj, hidden).reshape(B, 1, a.num_heads, a.head_dim).transpose(0, 2, 1, 3)
    k_new = linear(a.k_proj, hidden).reshape(B, 1, a.num_kv_heads, a.head_dim).transpose(0, 2, 1, 3)
    v_new = linear(a.v_proj, hidden).reshape(B, 1, a.num_kv_heads, a.head_dim).transpose(0, 2, 1, 3)
    q, k_new = apply_rope(q, k_new, cos, sin)

    # scatter_kv_decode expects [B, 1, H_kv, D] (heads-third).
    k_scatter = k_new.transpose(0, 2, 1, 3)
    v_scatter = v_new.transpose(0, 2, 1, 3)
    new_layer = scatter_kv_decode(layer, k_scatter, v_scatter, phys_block, slot_in_block)

    # gather_kv returns [B, NB_MAX*block_size, H_kv, D]; transpose to heads-second.
    k_all, v_all = gather_kv(new_layer, block_tables)
    k_all = k_all.transpose(0, 2, 1, 3)
    v_all = v_all.transpose(0, 2, 1, 3)

    L = k_all.shape[2]
    k_pos = jnp.arange(L)
    mask = k_pos[None, None, None, :] <= positions[:, None, None, None]

    out = attention_kernel(q, k_all, v_all, mask, a.head_dim, a.num_heads, a.num_kv_heads)
    out = out.transpose(0, 2, 1, 3).reshape(B, 1, -1)
    return linear(a.o_proj, out), new_layer


def decoder_layer_decode_cb(d: DecoderLayer, hidden, cos, sin, layer, positions, block_tables, phys_block, slot_in_block):
    h, new_layer = attention_decode_cb(
        d.self_attn, rms_norm(d.input_layernorm, hidden), cos, sin, layer,
        positions, block_tables, phys_block, slot_in_block,
    )
    hidden = hidden + h
    hidden = hidden + dense(d.mlp, rms_norm(d.post_attention_layernorm, hidden))
    return hidden, new_layer


def decode_step_cb(
    m: Qwen2Model,
    last_tokens: Array,
    cache: PagedCache,
    positions: Array,
    block_tables: Array,
    phys_block: Array,
    slot_in_block: Array,
) -> tuple[Array, PagedCache]:
    B = last_tokens.shape[0]
    hidden = embed(m.embed_tokens, last_tokens)
    q_pos = positions[:, None]
    cos, sin = rope_cos_sin(m.rotary_emb, q_pos, hidden.dtype)
    new_layers: list[PagedLayerCache] = []
    for layer, lc in zip(m.layers, cache.layers):
        hidden, lc = decoder_layer_decode_cb(
            layer, hidden, cos, sin, lc, positions, block_tables, phys_block, slot_in_block
        )
        new_layers.append(lc)
    hidden = rms_norm(m.norm, hidden)
    return linear(m.lm_head, hidden), PagedCache(layers=new_layers, block_size=cache.block_size)


extend_step_jit = eqx.filter_jit(extend_step)
decode_step_cb_jit = eqx.filter_jit(decode_step_cb)
