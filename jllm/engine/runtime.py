from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .paged import BlockHash
from .request import Request

SlotPhase = Literal["prefilling", "decoding"]


@dataclass
class SlotState:
    slot: int
    request: Request
    phase: SlotPhase
    blocks: list[int] = field(default_factory=list)
    hashes: list[BlockHash] = field(default_factory=list)
    next_prefill_pos: int = 0
    filled_blocks: int = 0


@dataclass(frozen=True)
class RuntimeState:
    positions: np.ndarray
    last_tokens: np.ndarray
    block_tables: np.ndarray


def init_runtime_state(max_num_seqs: int, nb_max: int) -> RuntimeState:
    return RuntimeState(
        positions=np.zeros((max_num_seqs,), dtype=np.int32),
        last_tokens=np.zeros((max_num_seqs, 1), dtype=np.int32),
        block_tables=np.zeros((max_num_seqs, nb_max), dtype=np.int32),
    )


def with_cached_prefix(state: RuntimeState, slot: int, block_ids: list[int]) -> RuntimeState:
    positions = state.positions.copy()
    last_tokens = state.last_tokens.copy()
    block_tables = state.block_tables.copy()
    row = block_tables[slot]
    row.fill(0)
    if block_ids:
        row[: len(block_ids)] = np.asarray(block_ids, dtype=np.int32)
    positions[slot] = 0
    last_tokens[slot, 0] = 0
    return RuntimeState(positions=positions, last_tokens=last_tokens, block_tables=block_tables)


def with_block(state: RuntimeState, slot: int, logical_idx: int, block_id: int) -> RuntimeState:
    block_tables = state.block_tables.copy()
    block_tables[slot, logical_idx] = block_id
    return RuntimeState(
        positions=state.positions,
        last_tokens=state.last_tokens,
        block_tables=block_tables,
    )


def with_decode_state(state: RuntimeState, slot: int, position: int, last_token: int) -> RuntimeState:
    positions = state.positions.copy()
    last_tokens = state.last_tokens.copy()
    positions[slot] = np.int32(position)
    last_tokens[slot, 0] = np.int32(last_token)
    return RuntimeState(
        positions=positions,
        last_tokens=last_tokens,
        block_tables=state.block_tables,
    )


def without_slot(state: RuntimeState, slot: int) -> RuntimeState:
    positions = state.positions.copy()
    last_tokens = state.last_tokens.copy()
    block_tables = state.block_tables.copy()
    positions[slot] = 0
    last_tokens[slot, 0] = 0
    block_tables[slot].fill(0)
    return RuntimeState(positions=positions, last_tokens=last_tokens, block_tables=block_tables)
