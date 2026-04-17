"""Threaded driver around the pure EngineState.

No class: just a `Driver` dataclass (data container) + free functions. All
compute goes through pure functions in state.py; this file is plumbing only
(threading, queues, request bookkeeping, block allocation, chunked-prefill
scheduling, prefix-cache lookup).

Slot lifecycle:
    free                                            (no entry in driver.running)
    -> add_request -> waiting queue
    -> _admit                                       touches cached-prefix blocks,
                                                    flips to "prefilling" (or straight
                                                    to "decoding" if prompt fully cached)
    -> N * _prefill_chunk_one (one per step)        each writes 1 new block
    -> last chunk emits first generated token       slot_phase[slot] = "decoding"
    -> _decode (1 token/step, fused over all decoders)
    -> EOS or max_new_tokens                        _release (dec ref, LRU-append freed)

Each step:
    - _admit: pull waiting requests into free slots; compute hashes and touch
      cached prefixes; set up prefill state (or flip to decoding immediately
      for fully-cached prompts).
    - _prefill_chunk_one: advance one prefilling slot by one chunk (round-robin).
    - _decode: fused one-token-per-slot across all decoding slots.

Prefix caching:
    - Block hashes form a Merkle chain over `block_size`-token chunks.
    - Only full blocks are ever hashed/registered. Partial last block is untracked.
    - Registered on block-fill: at the end of an extend_step chunk that writes a
      full block, and after a decode step that crosses a block boundary.
"""
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

import jax.numpy as jnp
import numpy as np

from ..model.qwen2 import Qwen2Model
from .paged import (
    BlockHash,
    BlockManager,
    alloc_new,
    compute_block_hash,
    lookup,
    make_manager,
    register_hash,
    release,
    touch,
)
from .request import Request, SamplingParams, StepEvent
from .state import (
    EngineState,
    decode,
    init_state,
    prefill_chunk,
    release_slot,
    set_slot_after_prefill,
    update_block_table,
    write_block_table_row,
)

_IDLE_SLEEP = 0.001


@dataclass
class SlotPrefillState:
    prompt_ids: list[int]    # full prompt (may include a cached prefix)
    pos_start: int           # absolute position of the next chunk's first token
    filled_blocks: int       # how many logical blocks this slot owns so far


@dataclass
class Driver:
    """Mutable container holding the engine state and threading plumbing."""

    model: Qwen2Model
    max_num_seqs: int
    max_model_len: int
    max_prefill_len: int
    block_size: int
    num_blocks: int
    nb_max: int
    state: EngineState
    block_manager: BlockManager

    slot_blocks: dict[int, list[int]] = field(default_factory=dict)
    slot_hashes: dict[int, list[BlockHash]] = field(default_factory=dict)  # per full block
    slot_phase: dict[int, str] = field(default_factory=dict)  # "prefilling" | "decoding"
    slot_prefill: dict[int, SlotPrefillState] = field(default_factory=dict)
    last_prefill_slot: int = -1  # round-robin cursor
    n_extend_calls: int = 0  # observability for prefix-caching tests / benchmarks

    waiting: "queue.Queue[Request]" = field(default_factory=queue.Queue)
    running: dict[int, Request] = field(default_factory=dict)
    streams: dict[int, "queue.Queue[StepEvent]"] = field(default_factory=dict)

    streams_lock: threading.Lock = field(default_factory=threading.Lock)
    id_lock: threading.Lock = field(default_factory=threading.Lock)
    next_id: int = 0

    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None


def make_driver(
    model: Qwen2Model,
    max_num_seqs: int,
    max_model_len: int,
    max_prefill_len: Optional[int] = None,
    block_size: int = 16,
    num_blocks: Optional[int] = None,
    dtype=jnp.bfloat16,
) -> Driver:
    if max_prefill_len is None:
        max_prefill_len = max_model_len
    if max_prefill_len > max_model_len:
        raise ValueError("max_prefill_len must be <= max_model_len")
    if max_model_len % block_size != 0:
        raise ValueError(f"max_model_len ({max_model_len}) must be a multiple of block_size ({block_size})")
    if max_prefill_len % block_size != 0:
        raise ValueError(f"max_prefill_len ({max_prefill_len}) must be a multiple of block_size ({block_size})")

    nb_max = max_model_len // block_size
    # Worst case: block 0 sentinel + max_num_seqs * nb_max blocks live.
    worst_case = 1 + max_num_seqs * nb_max
    if num_blocks is None:
        num_blocks = worst_case
    if num_blocks < worst_case:
        raise ValueError(
            f"num_blocks={num_blocks} is below worst-case {worst_case} "
            f"(1 sentinel + max_num_seqs={max_num_seqs} * nb_max={nb_max}); "
            f"undersizing requires preemption which isn't implemented."
        )

    return Driver(
        model=model,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        max_prefill_len=max_prefill_len,
        block_size=block_size,
        num_blocks=num_blocks,
        nb_max=nb_max,
        state=init_state(model.cfg, max_num_seqs, max_model_len, block_size, num_blocks, dtype),
        block_manager=make_manager(num_blocks),
    )


def add_request(
    driver: Driver, prompt_ids: list[int], sampling: Optional[SamplingParams] = None
) -> int:
    sp = sampling or SamplingParams()
    if len(prompt_ids) > driver.max_prefill_len:
        raise ValueError(f"prompt too long for max_prefill_len={driver.max_prefill_len}")
    if len(prompt_ids) + sp.max_new_tokens > driver.max_model_len:
        raise ValueError("prompt+max_new exceeds max_model_len")
    with driver.id_lock:
        rid = driver.next_id
        driver.next_id += 1
    req = Request(request_id=rid, prompt_ids=list(prompt_ids), sampling=sp)
    q: "queue.Queue[StepEvent]" = queue.Queue()
    with driver.streams_lock:
        driver.streams[rid] = q
    driver.waiting.put(req)
    return rid


def stream(driver: Driver, request_id: int) -> Iterator[StepEvent]:
    with driver.streams_lock:
        q = driver.streams.get(request_id)
    if q is None:
        raise KeyError(f"unknown or already-drained request_id: {request_id}")
    while True:
        ev = q.get()
        yield ev
        if ev.finished:
            with driver.streams_lock:
                driver.streams.pop(request_id, None)
            return


def has_work(driver: Driver) -> bool:
    return (not driver.waiting.empty()) or bool(driver.running)


def step(driver: Driver) -> list[StepEvent]:
    events = _admit(driver)
    prefill_ev = _prefill_chunk_one(driver)
    if prefill_ev is not None:
        events.append(prefill_ev)
    events.extend(_decode(driver))
    return events


def start(driver: Driver) -> None:
    if driver.thread is not None and driver.thread.is_alive():
        return
    driver.stop_event.clear()
    driver.thread = threading.Thread(target=_driver_loop, args=(driver,), name="jllm-engine", daemon=True)
    driver.thread.start()


def stop(driver: Driver, join: bool = True) -> None:
    driver.stop_event.set()
    if join and driver.thread is not None:
        driver.thread.join()
        driver.thread = None


# ---------- internals ----------


def _driver_loop(driver: Driver) -> None:
    while not driver.stop_event.is_set():
        events = step(driver)
        if events:
            with driver.streams_lock:
                for ev in events:
                    q = driver.streams.get(ev.request_id)
                    if q is not None:
                        q.put(ev)
        elif not has_work(driver):
            time.sleep(_IDLE_SLEEP)


def _free_slots(driver: Driver) -> list[int]:
    return [s for s in range(driver.max_num_seqs) if s not in driver.running]


def _compute_block_hashes(prompt_ids: list[int], block_size: int) -> list[BlockHash]:
    """Merkle chain over `block_size`-groups of full blocks. Partial last block
    (if any) is skipped — only full blocks get cached."""
    nb_full = len(prompt_ids) // block_size
    hashes: list[BlockHash] = []
    parent: BlockHash = b""
    for bi in range(nb_full):
        tokens = prompt_ids[bi * block_size : (bi + 1) * block_size]
        h = compute_block_hash(parent, tokens)
        hashes.append(h)
        parent = h
    return hashes


def _admit(driver: Driver) -> list[StepEvent]:
    """Pull requests into free slots. Look up cached-prefix blocks via content
    hashes; touch them (inc ref_cnt, unlink from free queue). Set up prefill
    state. No compute happens here — prefill chunks advance in _prefill_chunk_one."""
    events: list[StepEvent] = []
    for slot in _free_slots(driver):
        try:
            req = driver.waiting.get_nowait()
        except queue.Empty:
            break

        T = len(req.prompt_ids)
        bs = driver.block_size

        # Compute hashes for all full blocks of the prompt and look up a
        # contiguous cached prefix starting from block 0.
        hashes = _compute_block_hashes(req.prompt_ids, bs)
        cached: list[int] = []
        for h in hashes:
            block_id = lookup(driver.block_manager, h)
            if block_id is None:
                break
            cached.append(block_id)

        # Touch cached blocks (ref_cnt++, unlink from free queue if ref was 0).
        for block_id in cached:
            touch(driver.block_manager, block_id)

        ncached = len(cached)
        pos_start = ncached * bs
        uncached_tokens = T - pos_start

        # Initialise per-slot state.
        driver.running[slot] = req
        driver.slot_blocks[slot] = list(cached)
        driver.slot_hashes[slot] = list(hashes[:ncached])

        # Write cached blocks into the slot's block_tables row. Trailing entries
        # stay 0 (sentinel); prefill chunks will fill them as they run.
        row = np.zeros((driver.nb_max,), dtype=np.int32)
        for i, b in enumerate(cached):
            row[i] = b
        driver.state = write_block_table_row(driver.state, slot, jnp.asarray(row))

        if uncached_tokens == 0:
            # Fully cached prompt. No extend_step needed: Q for the first
            # decode step comes from the last prompt token, whose K/V is
            # already in the cached block. positions[slot] = T-1 so the next
            # decode writes K/V at position T-1 (a no-op overwrite with the
            # same deterministic value) and emits logits for position T.
            driver.slot_phase[slot] = "decoding"
            driver.state = set_slot_after_prefill(
                driver.state, slot,
                prompt_len=T - 1,
                first_token=req.prompt_ids[-1],
                block_table_row=jnp.asarray(row),
            )
            # No StepEvent yet — _decode later this step will emit the first
            # generated token.
        else:
            # Partial/no cache hit: prefill the uncached suffix via extend_step.
            driver.slot_phase[slot] = "prefilling"
            driver.slot_prefill[slot] = SlotPrefillState(
                prompt_ids=list(req.prompt_ids),
                pos_start=pos_start,
                filled_blocks=ncached,
            )
    return events


def _prefill_chunk_one(driver: Driver) -> Optional[StepEvent]:
    """Advance one prefilling slot by one chunk (round-robin). If this chunk was
    the final one for the slot, compute first token + flip to decoding and emit
    a StepEvent. Returns None if no slot is prefilling."""
    prefilling = [s for s in driver.running if driver.slot_phase[s] == "prefilling"]
    if not prefilling:
        return None
    # Round-robin: pick the slot after driver.last_prefill_slot.
    prefilling.sort()
    for candidate in prefilling:
        if candidate > driver.last_prefill_slot:
            slot = candidate
            break
    else:
        slot = prefilling[0]
    driver.last_prefill_slot = slot

    req = driver.running[slot]
    ps = driver.slot_prefill[slot]
    bs = driver.block_size
    prompt_len = len(ps.prompt_ids)

    # Allocate a fresh block for this chunk; wire it into the slot's block_tables.
    phys_block = alloc_new(driver.block_manager)
    driver.slot_blocks[slot].append(phys_block)
    logical_idx = ps.filled_blocks
    driver.state = update_block_table(driver.state, slot, logical_idx, phys_block)

    # Build the padded chunk_ids[1, bs]. Tail padding is harmless: its K/V goes
    # past prompt_len and will be dropped by the decode mask.
    chunk = np.zeros((1, bs), dtype=np.int32)
    tokens_in_chunk = min(bs, prompt_len - ps.pos_start)
    chunk[0, :tokens_in_chunk] = ps.prompt_ids[ps.pos_start : ps.pos_start + tokens_in_chunk]

    driver.state, logits = prefill_chunk(
        driver.state, driver.model,
        jnp.asarray(chunk),
        ps.pos_start,
        driver.state.block_tables[slot],
        phys_block,
    )
    driver.n_extend_calls += 1

    # If this chunk filled a complete block (all bs tokens real), register its
    # hash so a future request with this prefix can hit the cache.
    if tokens_in_chunk == bs:
        parent = driver.slot_hashes[slot][-1] if driver.slot_hashes[slot] else b""
        block_hash = compute_block_hash(parent, ps.prompt_ids[ps.pos_start : ps.pos_start + bs])
        register_hash(driver.block_manager, phys_block, block_hash)
        driver.slot_hashes[slot].append(block_hash)

    # Advance bookkeeping.
    ps.pos_start += bs
    ps.filled_blocks += 1

    is_last_chunk = ps.pos_start >= prompt_len
    if not is_last_chunk:
        return None  # intermediate chunk: no token emitted yet

    # Final chunk: pick logits at the last real token's position within the chunk.
    last_real_in_chunk = tokens_in_chunk - 1
    first_tok = int(jnp.argmax(logits[0, last_real_in_chunk, :]))
    driver.state = set_slot_after_prefill(
        driver.state, slot, prompt_len, first_tok, driver.state.block_tables[slot]
    )
    driver.slot_phase[slot] = "decoding"
    del driver.slot_prefill[slot]

    req.output_ids.append(first_tok)
    finished = _check_finish(req, first_tok)
    ev = StepEvent(req.request_id, first_tok, finished)
    if finished:
        _release(driver, slot)
    return ev


def _decode(driver: Driver) -> list[StepEvent]:
    decoding = [s for s in driver.running if driver.slot_phase[s] == "decoding"]
    if not decoding:
        return []

    # For each decoding slot, `positions[slot]` is the index where the NEW token
    # will be written. Allocate a new physical block if we're crossing into a
    # fresh logical block.
    positions_np = np.asarray(driver.state.positions)
    phys_block_np = np.zeros((driver.max_num_seqs,), dtype=np.int32)
    slot_in_block_np = np.zeros((driver.max_num_seqs,), dtype=np.int32)

    for slot in decoding:
        pos = int(positions_np[slot])
        logical_idx = pos // driver.block_size
        if logical_idx >= len(driver.slot_blocks[slot]):
            new_block = alloc_new(driver.block_manager)
            driver.slot_blocks[slot].append(new_block)
            driver.state = update_block_table(driver.state, slot, logical_idx, new_block)

        phys_block_np[slot] = driver.slot_blocks[slot][logical_idx]
        slot_in_block_np[slot] = pos % driver.block_size

    phys_block = jnp.asarray(phys_block_np)
    slot_in_block = jnp.asarray(slot_in_block_np)
    driver.state, new_toks = decode(driver.state, driver.model, phys_block, slot_in_block)

    new_toks_np = np.asarray(new_toks)
    events: list[StepEvent] = []
    for slot in decoding:
        req = driver.running[slot]
        tok = int(new_toks_np[slot])
        req.output_ids.append(tok)
        finished = _check_finish(req, tok)
        events.append(StepEvent(req.request_id, tok, finished))

        # Did this decode step fill a block? After decode, positions[slot] has
        # been incremented, so new_pos = old_pos + 1. If new_pos % bs == 0, the
        # block at logical index (new_pos//bs - 1) just filled.
        new_pos = int(positions_np[slot]) + 1
        if new_pos % driver.block_size == 0 and new_pos > 0:
            logical_filled = new_pos // driver.block_size - 1
            if logical_filled >= len(driver.slot_hashes[slot]):
                # Full sequence of tokens in this block: positions (new_pos-bs)..new_pos-1.
                full_seq = req.prompt_ids + req.output_ids
                start, end = new_pos - driver.block_size, new_pos
                if end <= len(full_seq):
                    tokens = full_seq[start:end]
                    parent = driver.slot_hashes[slot][-1] if driver.slot_hashes[slot] else b""
                    block_hash = compute_block_hash(parent, tokens)
                    block_id = driver.slot_blocks[slot][logical_filled]
                    register_hash(driver.block_manager, block_id, block_hash)
                    driver.slot_hashes[slot].append(block_hash)

        if finished:
            _release(driver, slot)
    return events


def _release(driver: Driver, slot: int) -> None:
    req = driver.running.pop(slot)
    req.finished = True
    release(driver.block_manager, driver.slot_blocks[slot])
    del driver.slot_blocks[slot]
    driver.slot_hashes.pop(slot, None)
    driver.slot_phase.pop(slot, None)
    driver.slot_prefill.pop(slot, None)
    driver.state = release_slot(driver.state, slot)


def _check_finish(req: Request, tok: int) -> bool:
    if len(req.output_ids) >= req.sampling.max_new_tokens:
        return True
    if req.sampling.eos_id is not None and tok == req.sampling.eos_id:
        return True
    return False
