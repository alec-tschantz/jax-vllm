import equinox as eqx
from jax import Array
from jax import numpy as jnp

from ..model.common import DecoderOnlyModel, KVCacheConfig
from .generate import decode_step_cb_jit, prefill_step_jit
from .paged import PagedCache, init_paged_cache


class EngineState(eqx.Module):
    cache: PagedCache


def init_state(
    cfg: KVCacheConfig,
    block_size: int,
    num_blocks: int,
    dtype,
) -> EngineState:
    return EngineState(cache=init_paged_cache(cfg, num_blocks, block_size, dtype))


def prefill(
    model: DecoderOnlyModel,
    state: EngineState,
    chunk_ids: Array,
    pos_starts: Array,
    block_tables: Array,
    valid_tokens: Array,
    last_token_idx: Array,
    phys_blocks: Array,
) -> tuple[EngineState, Array]:
    next_toks, new_cache = prefill_step_jit(
        model,
        chunk_ids,
        state.cache,
        block_tables,
        pos_starts,
        valid_tokens,
        last_token_idx,
        phys_blocks,
    )
    return EngineState(cache=new_cache), next_toks


def decode(
    model: DecoderOnlyModel,
    state: EngineState,
    last_tokens: Array,
    positions: Array,
    valid_rows: Array,
    block_tables: Array,
    phys_block: Array,
    slot_in_block: Array,
) -> tuple[EngineState, Array]:
    logits, new_cache = decode_step_cb_jit(
        model,
        last_tokens,
        state.cache,
        positions,
        valid_rows,
        block_tables,
        phys_block,
        slot_in_block,
    )
    new_toks = jnp.argmax(logits[:, 0, :], axis=-1).astype(jnp.int32)
    return EngineState(cache=new_cache), new_toks
