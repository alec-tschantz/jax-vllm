"""Threaded driver around the JAX KV-cache state.

The host runtime owns slot lifecycle, scheduler policy, and prefix-cache
bookkeeping. The JAX-facing state only carries the paged KV cache.
"""
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

import jax.numpy as jnp
import numpy as np

from ..model.common import DecoderOnlyModel
from .cache import register_full_block, release_blocks, take_cached_prefix
from .paged import BlockManager, alloc_new, make_manager
from .request import EngineStats, Request, SamplingParams, StepEvent
from .runtime import (
    RuntimeState,
    SlotState,
    init_runtime_state,
    with_block,
    with_cached_prefix,
    with_decode_state,
    without_slot,
)
from .state import EngineState, decode, init_state, prefill

_IDLE_SLEEP = 0.001


@dataclass
class Driver:
    """Mutable engine container for scheduler/runtime state."""

    model: DecoderOnlyModel
    max_num_seqs: int
    max_model_len: int
    max_prefill_len: int
    block_size: int
    num_blocks: int
    nb_max: int
    batch_buckets: tuple[int, ...]
    state: EngineState
    runtime: RuntimeState
    block_manager: BlockManager
    stats: EngineStats = field(default_factory=EngineStats)

    last_prefill_slot: int = -1

    waiting: "queue.Queue[Request]" = field(default_factory=queue.Queue)
    running: dict[int, SlotState] = field(default_factory=dict)
    streams: dict[int, "queue.Queue[StepEvent]"] = field(default_factory=dict)

    streams_lock: threading.Lock = field(default_factory=threading.Lock)
    id_lock: threading.Lock = field(default_factory=threading.Lock)
    next_id: int = 0

    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None


def make_driver(
    model: DecoderOnlyModel,
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
        batch_buckets=_batch_buckets(max_num_seqs),
        state=init_state(model.cfg, block_size, num_blocks, dtype),
        runtime=init_runtime_state(max_num_seqs, nb_max),
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
    driver.stats.scheduler_loops += 1
    events = _admit(driver)
    # Prefill before decode so newly-admitted short prompts can join the active
    # decode batch promptly instead of being starved behind existing decoders.
    events.extend(_prefill_batch(driver))
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


def _batch_buckets(max_num_seqs: int) -> tuple[int, ...]:
    buckets: list[int] = []
    size = 1
    while size < max_num_seqs:
        buckets.append(size)
        size *= 2
    buckets.append(max_num_seqs)
    return tuple(dict.fromkeys(buckets))


def _bucket_for(driver: Driver, active_size: int) -> int:
    for bucket in driver.batch_buckets:
        if active_size <= bucket:
            return bucket
    return driver.batch_buckets[-1]


def _free_slots(driver: Driver) -> list[int]:
    return [slot for slot in range(driver.max_num_seqs) if slot not in driver.running]


def _prefilling_slots(driver: Driver) -> list[SlotState]:
    slots = [slot for slot in driver.running.values() if slot.phase == "prefilling"]
    if not slots:
        return []
    slots.sort(key=lambda slot: slot.slot)
    for idx, slot in enumerate(slots):
        if slot.slot > driver.last_prefill_slot:
            return slots[idx:] + slots[:idx]
    return slots


def _decoding_slots(driver: Driver) -> list[SlotState]:
    return sorted(
        (slot for slot in driver.running.values() if slot.phase == "decoding"),
        key=lambda slot: slot.slot,
    )


def _admit(driver: Driver) -> list[StepEvent]:
    events: list[StepEvent] = []
    for slot_id in _free_slots(driver):
        try:
            req = driver.waiting.get_nowait()
        except queue.Empty:
            break

        cached = take_cached_prefix(driver.block_manager, req.prompt_ids, driver.block_size)
        pos_start = len(cached.block_ids) * driver.block_size
        slot = SlotState(
            slot=slot_id,
            request=req,
            phase="prefilling",
            blocks=list(cached.block_ids),
            hashes=list(cached.hashes),
            next_prefill_pos=pos_start,
            filled_blocks=len(cached.block_ids),
        )
        req.slot = slot_id
        driver.running[slot_id] = slot
        driver.runtime = with_cached_prefix(driver.runtime, slot_id, slot.blocks)
        driver.stats.admissions += 1
        driver.stats.cache_hit_blocks += len(slot.blocks)

        if pos_start >= len(req.prompt_ids):
            slot.phase = "decoding"
            driver.runtime = with_decode_state(
                driver.runtime, slot_id, len(req.prompt_ids) - 1, req.prompt_ids[-1]
            )
    return events


def _prefill_batch(driver: Driver) -> list[StepEvent]:
    active_slots = _prefilling_slots(driver)
    if not active_slots:
        return []

    driver.last_prefill_slot = active_slots[-1].slot
    bucket = _bucket_for(driver, len(active_slots))
    bs = driver.block_size

    chunk_ids = np.zeros((bucket, bs), dtype=np.int32)
    pos_starts = np.zeros((bucket,), dtype=np.int32)
    phys_blocks = np.zeros((bucket,), dtype=np.int32)
    block_tables = np.zeros((bucket, driver.nb_max), dtype=np.int32)
    valid_tokens = np.zeros((bucket,), dtype=np.int32)

    for idx, slot in enumerate(active_slots):
        phys_block = alloc_new(driver.block_manager)
        slot.blocks.append(phys_block)
        driver.runtime = with_block(driver.runtime, slot.slot, slot.filled_blocks, phys_block)

        prompt_len = len(slot.request.prompt_ids)
        tokens_in_chunk = min(bs, prompt_len - slot.next_prefill_pos)
        chunk_ids[idx, :tokens_in_chunk] = slot.request.prompt_ids[
            slot.next_prefill_pos : slot.next_prefill_pos + tokens_in_chunk
        ]
        pos_starts[idx] = slot.next_prefill_pos
        phys_blocks[idx] = phys_block
        block_tables[idx] = driver.runtime.block_tables[slot.slot]
        valid_tokens[idx] = tokens_in_chunk

    driver.state, logits = prefill(
        driver.model,
        driver.state,
        jnp.asarray(chunk_ids),
        jnp.asarray(pos_starts),
        jnp.asarray(block_tables),
        jnp.asarray(phys_blocks),
    )
    driver.stats.prefill_batches += 1
    driver.stats.extend_calls += len(active_slots)

    logits_np = np.asarray(logits)
    events: list[StepEvent] = []
    for idx, slot in enumerate(active_slots):
        tokens_in_chunk = int(valid_tokens[idx])
        if tokens_in_chunk == bs:
            parent_hash = slot.hashes[-1] if slot.hashes else b""
            block_hash = register_full_block(
                driver.block_manager,
                int(phys_blocks[idx]),
                parent_hash,
                slot.request.prompt_ids[slot.next_prefill_pos : slot.next_prefill_pos + bs],
            )
            slot.hashes.append(block_hash)

        slot.next_prefill_pos += bs
        slot.filled_blocks += 1
        if slot.next_prefill_pos < len(slot.request.prompt_ids):
            continue

        first_tok = int(np.argmax(logits_np[idx, tokens_in_chunk - 1, :]))
        driver.runtime = with_decode_state(driver.runtime, slot.slot, len(slot.request.prompt_ids), first_tok)
        slot.phase = "decoding"

        slot.request.output_ids.append(first_tok)
        driver.stats.emitted_tokens += 1
        finished = _check_finish(slot.request, first_tok)
        events.append(StepEvent(slot.request.request_id, first_tok, finished))
        if finished:
            _release(driver, slot.slot)
    return events


def _decode(driver: Driver) -> list[StepEvent]:
    active_slots = _decoding_slots(driver)
    if not active_slots:
        return []

    bucket = _bucket_for(driver, len(active_slots))
    bs = driver.block_size

    last_tokens = np.zeros((bucket, 1), dtype=np.int32)
    positions = np.zeros((bucket,), dtype=np.int32)
    phys_blocks = np.zeros((bucket,), dtype=np.int32)
    slot_in_block = np.zeros((bucket,), dtype=np.int32)
    block_tables = np.zeros((bucket, driver.nb_max), dtype=np.int32)

    for idx, slot in enumerate(active_slots):
        position = int(driver.runtime.positions[slot.slot])
        logical_idx = position // bs
        if logical_idx >= len(slot.blocks):
            new_block = alloc_new(driver.block_manager)
            slot.blocks.append(new_block)
            driver.runtime = with_block(driver.runtime, slot.slot, logical_idx, new_block)

        positions[idx] = position
        last_tokens[idx, 0] = driver.runtime.last_tokens[slot.slot, 0]
        phys_blocks[idx] = slot.blocks[logical_idx]
        slot_in_block[idx] = position % bs
        block_tables[idx] = driver.runtime.block_tables[slot.slot]

    driver.state, new_toks = decode(
        driver.model,
        driver.state,
        jnp.asarray(last_tokens),
        jnp.asarray(positions),
        jnp.asarray(block_tables),
        jnp.asarray(phys_blocks),
        jnp.asarray(slot_in_block),
    )
    driver.stats.decode_batches += 1
    driver.stats.decode_calls += len(active_slots)

    new_toks_np = np.asarray(new_toks)
    events: list[StepEvent] = []
    for idx, slot in enumerate(active_slots):
        tok = int(new_toks_np[idx])
        slot.request.output_ids.append(tok)
        driver.runtime = with_decode_state(driver.runtime, slot.slot, int(positions[idx]) + 1, tok)
        driver.stats.emitted_tokens += 1

        new_pos = int(positions[idx]) + 1
        if new_pos % bs == 0 and new_pos > 0:
            logical_filled = new_pos // bs - 1
            if logical_filled >= len(slot.hashes):
                full_seq = slot.request.prompt_ids + slot.request.output_ids
                start = new_pos - bs
                end = new_pos
                if end <= len(full_seq):
                    parent_hash = slot.hashes[-1] if slot.hashes else b""
                    block_hash = register_full_block(
                        driver.block_manager,
                        slot.blocks[logical_filled],
                        parent_hash,
                        full_seq[start:end],
                    )
                    slot.hashes.append(block_hash)

        finished = _check_finish(slot.request, tok)
        events.append(StepEvent(slot.request.request_id, tok, finished))
        if finished:
            _release(driver, slot.slot)
    return events


def _release(driver: Driver, slot_id: int) -> None:
    slot = driver.running.pop(slot_id)
    slot.request.finished = True
    slot.request.slot = None
    release_blocks(driver.block_manager, slot.blocks)
    driver.runtime = without_slot(driver.runtime, slot_id)


def _check_finish(req: Request, tok: int) -> bool:
    if len(req.output_ids) >= req.sampling.max_new_tokens:
        return True
    if req.sampling.eos_id is not None and tok == req.sampling.eos_id:
        return True
    return False
