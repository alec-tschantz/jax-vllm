"""Synchronous greedy helper for tests.

Drives the paged engine in a single thread: add request, step until done.
Replaces the deleted `generate_greedy` reference implementation.
"""
from typing import Optional

import jax.numpy as jnp

from jllm.engine import engine as eng
from jllm.engine.request import SamplingParams


def greedy(
    model,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_id: Optional[int] = None,
    dtype=jnp.float32,
    max_model_len: Optional[int] = None,
    block_size: int = 16,
) -> list[int]:
    """Run prompt through a 1-slot paged engine synchronously; return new tokens."""
    T = len(prompt_ids)
    if max_model_len is None:
        raw = T + max_new_tokens
        max_model_len = ((raw + block_size - 1) // block_size) * block_size
    max_prefill_len = ((T + block_size - 1) // block_size) * block_size
    driver = eng.make_driver(
        model,
        max_num_seqs=1,
        max_model_len=max_model_len,
        max_prefill_len=max_prefill_len,
        block_size=block_size,
        dtype=dtype,
    )
    rid = eng.add_request(driver, list(prompt_ids), SamplingParams(max_new_tokens=max_new_tokens, eos_id=eos_id))
    out: list[int] = []
    while eng.has_work(driver):
        for ev in eng.step(driver):
            out.append(ev.token)
    return out
