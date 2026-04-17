import os

import jax.numpy as jnp
import pytest

from jllm.engine import engine
from jllm.engine.request import SamplingParams
from jllm.model.weights import load_from_path

MODEL_PATH = os.environ.get("PARITY_MODEL", "weights/Qwen2.5-0.5B-Instruct")

PROMPTS = [
    [785, 6722, 315, 9625, 374],              # "The capital of France is"
    [750, 79683, 1445, 982],                  # "def fibonacci(n):"
    [12522, 5193, 264, 882],                  # "Once upon a time"
    [9454, 8420, 11, 1246, 646],              # "Yeah okay, how can"
]
MAX_NEW = 10


def _run_engine_fp32(model, prompts, max_num_seqs, max_model_len, max_new):
    driver = engine.make_driver(
        model, max_num_seqs=max_num_seqs, max_model_len=max_model_len, dtype=jnp.float32
    )
    rids = [engine.add_request(driver, p, SamplingParams(max_new_tokens=max_new)) for p in prompts]
    out: dict[int, list[int]] = {rid: [] for rid in rids}
    while engine.has_work(driver):
        for ev in engine.step(driver):
            out[ev.request_id].append(ev.token)
    return [out[rid] for rid in rids]


@pytest.fixture(scope="module")
def model_fp32():
    return load_from_path(MODEL_PATH, dtype=jnp.float32)


def test_engine_is_deterministic(model_fp32):
    """Running the engine twice with the same requests should produce identical tokens."""
    first = _run_engine_fp32(model_fp32, PROMPTS, max_num_seqs=2, max_model_len=32, max_new=MAX_NEW)
    second = _run_engine_fp32(model_fp32, PROMPTS, max_num_seqs=2, max_model_len=32, max_new=MAX_NEW)
    assert first == second


def test_slot_output_independent_of_batchmate(model_fp32):
    """Request A's output must be independent of what else is in the batch.
    Runs request 0 alongside request 1, then runs request 0 alongside request 2.
    The tokens emitted for request 0 must be identical in both runs."""
    with_peer_1 = _run_engine_fp32(
        model_fp32, [PROMPTS[0], PROMPTS[1]], max_num_seqs=2, max_model_len=32, max_new=MAX_NEW
    )
    with_peer_2 = _run_engine_fp32(
        model_fp32, [PROMPTS[0], PROMPTS[2]], max_num_seqs=2, max_model_len=32, max_new=MAX_NEW
    )
    assert with_peer_1[0] == with_peer_2[0], (
        f"request 0 output changed with different batchmate\n"
        f"  with peer 1: {with_peer_1[0]}\n"
        f"  with peer 2: {with_peer_2[0]}"
    )


def test_engine_matches_solo_fp32(model_fp32):
    """Engine output for each request should match that same request run in a 1-slot engine (solo)."""
    cb_outputs = _run_engine_fp32(
        model_fp32, PROMPTS, max_num_seqs=2, max_model_len=32, max_new=MAX_NEW
    )
    for i, p in enumerate(PROMPTS):
        solo = _run_engine_fp32(model_fp32, [p], max_num_seqs=1, max_model_len=32, max_new=MAX_NEW)[0]
        assert cb_outputs[i] == solo, (
            f"request {i} diverges: cb={cb_outputs[i]} solo={solo}"
        )


def test_prefix_caching_skips_extend_on_second_request(model_fp32):
    """Send the same prompt twice on one driver; the second admit should find
    the prompt's full blocks in the cache and skip all extend_step calls (or
    all but the partial-last-block call). Output tokens must match."""
    from jllm.engine import engine
    from jllm.engine.request import SamplingParams

    # Use a prompt that's exactly 2 full blocks (32 tokens) so it's fully-cacheable.
    block_size = 16
    prompt_32 = (PROMPTS[0] * 10)[:32]  # pad/truncate to exactly 32 tokens

    driver = engine.make_driver(
        model_fp32, max_num_seqs=1, max_model_len=48, max_prefill_len=32,
        block_size=block_size, dtype=jnp.float32,
    )

    # First request: cold cache.
    driver.n_extend_calls = 0
    rid1 = engine.add_request(driver, prompt_32, SamplingParams(max_new_tokens=MAX_NEW))
    out1: list[int] = []
    while engine.has_work(driver):
        for ev in engine.step(driver):
            if ev.request_id == rid1:
                out1.append(ev.token)
    cold_extend_calls = driver.n_extend_calls

    # Second request: same prompt; prefix should be fully cached.
    driver.n_extend_calls = 0
    rid2 = engine.add_request(driver, prompt_32, SamplingParams(max_new_tokens=MAX_NEW))
    out2: list[int] = []
    while engine.has_work(driver):
        for ev in engine.step(driver):
            if ev.request_id == rid2:
                out2.append(ev.token)
    warm_extend_calls = driver.n_extend_calls

    assert out1 == out2, f"outputs diverge: cold={out1} warm={out2}"
    assert cold_extend_calls >= 2, f"cold path should have run >=2 extend_step calls, got {cold_extend_calls}"
    assert warm_extend_calls == 0, (
        f"fully-cached prompt should skip all extend_step calls; got {warm_extend_calls}"
    )


def test_chunked_prefill_runs_one_chunk_per_block(model_fp32):
    """A cold prompt of length T runs exactly ceil(T / block_size) extend_step
    calls. Regression guard: if someone fuses prefill back into one big kernel
    or chunks at the wrong granularity, this fails."""
    from jllm.engine import engine
    from jllm.engine.request import SamplingParams

    block_size = 16
    prompt = [785, 6722, 315, 9625, 374, 750, 79683, 1445, 982, 9454, 8420, 11]  # 12 tokens
    expected_chunks = (len(prompt) + block_size - 1) // block_size  # 1

    driver = engine.make_driver(
        model_fp32, max_num_seqs=1, max_model_len=32, max_prefill_len=16,
        block_size=block_size, dtype=jnp.float32,
    )
    driver.n_extend_calls = 0
    rid = engine.add_request(driver, prompt, SamplingParams(max_new_tokens=2))
    while engine.has_work(driver):
        for _ in engine.step(driver):
            pass
    assert driver.n_extend_calls == expected_chunks, (
        f"expected {expected_chunks} extend_step calls for a {len(prompt)}-token prompt "
        f"with block_size={block_size}; got {driver.n_extend_calls}"
    )

    # Now a longer prompt that needs multiple chunks.
    prompt_long = (prompt * 3)[:32]  # exactly 2 full blocks
    driver2 = engine.make_driver(
        model_fp32, max_num_seqs=1, max_model_len=48, max_prefill_len=32,
        block_size=block_size, dtype=jnp.float32,
    )
    driver2.n_extend_calls = 0
    rid2 = engine.add_request(driver2, prompt_long, SamplingParams(max_new_tokens=2))
    while engine.has_work(driver2):
        for _ in engine.step(driver2):
            pass
    assert driver2.n_extend_calls == 2, (
        f"expected 2 extend_step calls for a 32-token prompt with block_size=16; "
        f"got {driver2.n_extend_calls}"
    )


def test_partial_prefix_cache_hit_skips_only_cached_blocks(model_fp32):
    """Prompt B has the first block of prompt A as prefix, plus new tokens.
    B's admit should hit 1 cached block and extend_step only the suffix
    (1 chunk for the 2nd block). Outputs on identical suffix must match solo."""
    from jllm.engine import engine
    from jllm.engine.request import SamplingParams

    block_size = 16
    # Prompt A: 2 full blocks (32 tokens).
    prompt_a = (PROMPTS[0] * 10)[:32]
    # Prompt B: same first 16 tokens (shares block 0) + 16 different new tokens.
    prompt_b = prompt_a[:16] + (PROMPTS[1] * 10)[:16]

    driver = engine.make_driver(
        model_fp32, max_num_seqs=1, max_model_len=48, max_prefill_len=32,
        block_size=block_size, dtype=jnp.float32,
    )

    # Admit A first; this populates block 0 (first 16 tokens of A) into the cache.
    driver.n_extend_calls = 0
    rid_a = engine.add_request(driver, prompt_a, SamplingParams(max_new_tokens=MAX_NEW))
    while engine.has_work(driver):
        for _ in engine.step(driver):
            pass
    a_extend = driver.n_extend_calls
    assert a_extend == 2, f"cold A should run 2 extend calls, got {a_extend}"

    # Now B: first block matches A's first block → cached; second block is new.
    driver.n_extend_calls = 0
    rid_b = engine.add_request(driver, prompt_b, SamplingParams(max_new_tokens=MAX_NEW))
    while engine.has_work(driver):
        for _ in engine.step(driver):
            pass
    b_extend = driver.n_extend_calls
    assert b_extend == 1, (
        f"B shares 1 cached block with A and has 1 new block; expected 1 extend call, "
        f"got {b_extend}"
    )
