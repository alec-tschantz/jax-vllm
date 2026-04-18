"""JAX-facing engine state and packed kernel wrappers."""
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
    chunk_ids: Array,           # [B, C] int32
    pos_starts: Array,          # [B] int32
    block_tables: Array,        # [B, NB_MAX] int32
    phys_blocks: Array,         # [B] int32
) -> tuple[EngineState, Array]:
    logits, new_cache = prefill_step_jit(
        model, chunk_ids, state.cache, block_tables, pos_starts, phys_blocks
    )
    return EngineState(cache=new_cache), logits


def decode(
    model: DecoderOnlyModel,
    state: EngineState,
    last_tokens: Array,      # [B, 1] int32
    positions: Array,        # [B] int32
    block_tables: Array,     # [B, NB_MAX] int32
    phys_block: Array,       # [B] int32
    slot_in_block: Array,    # [B] int32
) -> tuple[EngineState, Array]:
    logits, new_cache = decode_step_cb_jit(
        model, last_tokens, state.cache, positions, block_tables, phys_block, slot_in_block
    )
    new_toks = jnp.argmax(logits[:, 0, :], axis=-1).astype(jnp.int32)
    return EngineState(cache=new_cache), new_toks
