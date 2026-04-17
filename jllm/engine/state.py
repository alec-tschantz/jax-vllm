"""Functional state machine for the continuous-batching engine.

Pure: every function takes an EngineState and returns a new one. Threading +
queue plumbing lives in engine.py as a thin wrapper.

The KV cache is a paged pool: layers live in PagedCache with blocks of shape
`[num_blocks, block_size, num_kv_heads, head_dim]`. Each slot owns a row of
`block_tables[slot, NB_MAX]` mapping logical block index → physical block id.
Unused trailing entries point at the sentinel block 0; the positional mask
(k_pos <= positions[b]) in the attention kernel drops them.
"""
import equinox as eqx
from jax import Array
from jax import numpy as jnp

from ..model.qwen2 import Qwen2Config, Qwen2Model
from .generate import decode_step_cb_jit, extend_step_jit
from .paged import PagedCache, init_paged_cache


class EngineState(eqx.Module):
    cache: PagedCache
    positions: Array      # [max_num_seqs] int32 — next position to write (= current length)
    last_tokens: Array    # [max_num_seqs, 1] int32 — last-generated token per decoding slot
    block_tables: Array   # [max_num_seqs, NB_MAX] int32


def init_state(
    cfg: Qwen2Config,
    max_num_seqs: int,
    max_model_len: int,
    block_size: int,
    num_blocks: int,
    dtype,
) -> EngineState:
    if max_model_len % block_size != 0:
        raise ValueError(f"max_model_len ({max_model_len}) must be a multiple of block_size ({block_size})")
    nb_max = max_model_len // block_size
    return EngineState(
        cache=init_paged_cache(cfg, num_blocks, block_size, dtype),
        positions=jnp.zeros((max_num_seqs,), dtype=jnp.int32),
        last_tokens=jnp.zeros((max_num_seqs, 1), dtype=jnp.int32),
        block_tables=jnp.zeros((max_num_seqs, nb_max), dtype=jnp.int32),
    )


def prefill_chunk(
    state: EngineState,
    model: Qwen2Model,
    chunk_ids: Array,           # [1, C] int32
    pos_start: int,
    block_table_row: Array,     # [NB_MAX] int32  — current row for this slot
    phys_block: int,            # physical block the chunk writes into
) -> tuple[EngineState, Array]:
    """Run one extend_step chunk. Caller gets full [1, C, V] logits to pick
    the last-real-token logits on the final chunk."""
    pos_start_j = jnp.asarray(pos_start, dtype=jnp.int32)
    phys_block_j = jnp.asarray(phys_block, dtype=jnp.int32)
    logits, new_cache = extend_step_jit(
        model, chunk_ids, state.cache, block_table_row,
        pos_start_j, phys_block_j,
    )
    return EngineState(
        cache=new_cache,
        positions=state.positions,
        last_tokens=state.last_tokens,
        block_tables=state.block_tables,
    ), logits


def set_slot_after_prefill(
    state: EngineState,
    slot: int,
    prompt_len: int,
    first_token: int,
    block_table_row: Array,      # [NB_MAX] int32 — final row
) -> EngineState:
    """Write per-slot bookkeeping after the last prefill chunk: position = prompt_len,
    last_tokens[slot] = first_token, block_tables[slot] = block_table_row."""
    return EngineState(
        cache=state.cache,
        positions=state.positions.at[slot].set(prompt_len),
        last_tokens=state.last_tokens.at[slot, 0].set(first_token),
        block_tables=state.block_tables.at[slot].set(block_table_row),
    )


def update_block_table(state: EngineState, slot: int, col: int, block_id: int) -> EngineState:
    """Write block_tables[slot, col] = block_id. Used during prefill (to plumb
    each chunk's block into the row) and during decode (when a new block is
    allocated on a boundary crossing)."""
    return EngineState(
        cache=state.cache,
        positions=state.positions,
        last_tokens=state.last_tokens,
        block_tables=state.block_tables.at[slot, col].set(block_id),
    )


def write_block_table_row(state: EngineState, slot: int, row: Array) -> EngineState:
    """Write block_tables[slot] = row. Used at admit when a slot inherits a
    prefix of cached block ids from the prefix-cache lookup."""
    return EngineState(
        cache=state.cache,
        positions=state.positions,
        last_tokens=state.last_tokens,
        block_tables=state.block_tables.at[slot].set(row),
    )


def decode(
    state: EngineState,
    model: Qwen2Model,
    phys_block: Array,      # [max_num_seqs] int32
    slot_in_block: Array,   # [max_num_seqs] int32
) -> tuple[EngineState, Array]:
    """One fused decode step; return (new_state, new_tokens [B])."""
    logits, new_cache = decode_step_cb_jit(
        model, state.last_tokens, state.cache, state.positions,
        state.block_tables, phys_block, slot_in_block,
    )
    new_toks = jnp.argmax(logits[:, 0, :], axis=-1).astype(jnp.int32)
    new_state = EngineState(
        cache=new_cache,
        positions=state.positions + 1,
        last_tokens=new_toks[:, None],
        block_tables=state.block_tables,
    )
    return new_state, new_toks


def release_slot(state: EngineState, slot: int) -> EngineState:
    nb_max = state.block_tables.shape[1]
    return EngineState(
        cache=state.cache,
        positions=state.positions.at[slot].set(0),
        last_tokens=state.last_tokens.at[slot, 0].set(0),
        block_tables=state.block_tables.at[slot].set(jnp.zeros((nb_max,), dtype=jnp.int32)),
    )
