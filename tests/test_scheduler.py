import pytest
import jax.numpy as jnp

from jllm.engine import engine as eng
from jllm.engine.request import SamplingParams

from ._tiny_model import make_tiny_model


def test_prefill_uses_active_batch(monkeypatch):
    model = make_tiny_model()
    driver = eng.make_driver(
        model,
        max_num_seqs=4,
        max_model_len=16,
        max_prefill_len=8,
        block_size=4,
        dtype=jnp.float32,
    )

    seen: list[int] = []
    real_prefill = eng.prefill

    def recording_prefill(
        model,
        state,
        chunk_ids,
        pos_starts,
        block_tables,
        valid_tokens,
        last_token_idx,
        phys_blocks,
    ):
        seen.append(int(chunk_ids.shape[0]))
        return real_prefill(
            model,
            state,
            chunk_ids,
            pos_starts,
            block_tables,
            valid_tokens,
            last_token_idx,
            phys_blocks,
        )

    monkeypatch.setattr(eng, "prefill", recording_prefill)
    eng.add_request(driver, [1, 2, 3, 4], SamplingParams(max_new_tokens=2))
    eng.add_request(driver, [5, 6, 7, 8], SamplingParams(max_new_tokens=2))

    eng.step(driver)

    assert seen == [2]
    assert driver.stats.prefill_batches == 1
    assert driver.stats.extend_calls == 2


@pytest.mark.parametrize(
    ("active_size", "expected_padding"),
    [(1, 0), (2, 0), (3, 1), (4, 0)],
)
def test_prefill_stats_track_active_slots_and_padding(active_size, expected_padding):
    model = make_tiny_model()
    driver = eng.make_driver(
        model,
        max_num_seqs=4,
        max_model_len=16,
        max_prefill_len=8,
        block_size=4,
        dtype=jnp.float32,
    )

    for offset in range(active_size):
        base = 1 + 4 * offset
        eng.add_request(
            driver,
            [base, base + 1, base + 2, base + 3],
            SamplingParams(max_new_tokens=2),
        )

    eng.step(driver)

    assert driver.stats.prefill_batches == 1
    assert driver.stats.prefill_slots_total == active_size
    assert driver.stats.prefill_padding_slots_total == expected_padding


def test_decode_uses_active_batch(monkeypatch):
    model = make_tiny_model()
    driver = eng.make_driver(
        model,
        max_num_seqs=4,
        max_model_len=16,
        max_prefill_len=8,
        block_size=4,
        dtype=jnp.float32,
    )

    seen: list[int] = []
    real_decode = eng.decode

    def recording_decode(
        model,
        state,
        last_tokens,
        positions,
        valid_rows,
        block_tables,
        phys_block,
        slot_in_block,
    ):
        seen.append(int(last_tokens.shape[0]))
        return real_decode(
            model,
            state,
            last_tokens,
            positions,
            valid_rows,
            block_tables,
            phys_block,
            slot_in_block,
        )

    monkeypatch.setattr(eng, "decode", recording_decode)
    eng.add_request(driver, [1, 2, 3, 4], SamplingParams(max_new_tokens=2))
    eng.add_request(driver, [5, 6, 7, 8], SamplingParams(max_new_tokens=2))

    eng.step(driver)
    eng.step(driver)

    assert seen == [2]
    assert driver.stats.decode_batches == 1
    assert driver.stats.decode_calls == 2


@pytest.mark.parametrize(
    ("active_size", "expected_padding"),
    [(1, 0), (2, 0), (3, 1), (4, 0)],
)
def test_decode_stats_track_active_slots_and_padding(active_size, expected_padding):
    model = make_tiny_model()
    driver = eng.make_driver(
        model,
        max_num_seqs=4,
        max_model_len=16,
        max_prefill_len=8,
        block_size=4,
        dtype=jnp.float32,
    )

    for offset in range(active_size):
        base = 1 + 4 * offset
        eng.add_request(
            driver,
            [base, base + 1, base + 2, base + 3],
            SamplingParams(max_new_tokens=2),
        )

    eng.step(driver)
    eng.step(driver)

    assert driver.stats.decode_batches == 1
    assert driver.stats.decode_slots_total == active_size
    assert driver.stats.decode_padding_slots_total == expected_padding


def test_scheduler_prefills_before_decode_when_both_are_ready(monkeypatch):
    model = make_tiny_model()
    driver = eng.make_driver(
        model,
        max_num_seqs=2,
        max_model_len=16,
        max_prefill_len=8,
        block_size=4,
        dtype=jnp.float32,
    )
    cached_prompt = [1, 2, 3, 4]
    cold_prompt = [9, 10, 11, 12]

    eng.add_request(driver, cached_prompt, SamplingParams(max_new_tokens=1))
    while eng.has_work(driver):
        eng.step(driver)

    order: list[str] = []
    real_decode = eng.decode
    real_prefill = eng.prefill

    def recording_decode(
        model,
        state,
        last_tokens,
        positions,
        valid_rows,
        block_tables,
        phys_block,
        slot_in_block,
    ):
        order.append("decode")
        return real_decode(
            model,
            state,
            last_tokens,
            positions,
            valid_rows,
            block_tables,
            phys_block,
            slot_in_block,
        )

    def recording_prefill(
        model,
        state,
        chunk_ids,
        pos_starts,
        block_tables,
        valid_tokens,
        last_token_idx,
        phys_blocks,
    ):
        order.append("prefill")
        return real_prefill(
            model,
            state,
            chunk_ids,
            pos_starts,
            block_tables,
            valid_tokens,
            last_token_idx,
            phys_blocks,
        )

    monkeypatch.setattr(eng, "decode", recording_decode)
    monkeypatch.setattr(eng, "prefill", recording_prefill)

    eng.add_request(driver, cached_prompt, SamplingParams(max_new_tokens=2))
    eng.add_request(driver, cold_prompt, SamplingParams(max_new_tokens=2))

    cache_hits_before = driver.stats.cache_hit_blocks
    extend_calls_before = driver.stats.extend_calls
    eng.step(driver)

    assert order[:2] == ["prefill", "decode"]
    assert driver.stats.cache_hit_blocks - cache_hits_before == 1
    assert driver.stats.extend_calls - extend_calls_before == 1


def test_prefill_uses_live_context_bucket(monkeypatch):
    model = make_tiny_model()
    driver = eng.make_driver(
        model,
        max_num_seqs=2,
        max_model_len=64,
        max_prefill_len=8,
        block_size=4,
        dtype=jnp.float32,
    )

    seen: list[int] = []
    real_prefill = eng.prefill

    def recording_prefill(
        model,
        state,
        chunk_ids,
        pos_starts,
        block_tables,
        valid_tokens,
        last_token_idx,
        phys_blocks,
    ):
        seen.append(int(block_tables.shape[1]))
        return real_prefill(
            model,
            state,
            chunk_ids,
            pos_starts,
            block_tables,
            valid_tokens,
            last_token_idx,
            phys_blocks,
        )

    monkeypatch.setattr(eng, "prefill", recording_prefill)
    eng.add_request(driver, [1, 2, 3, 4], SamplingParams(max_new_tokens=1))

    eng.step(driver)

    assert seen == [1]


def test_decode_uses_live_context_bucket(monkeypatch):
    model = make_tiny_model()
    driver = eng.make_driver(
        model,
        max_num_seqs=2,
        max_model_len=64,
        max_prefill_len=8,
        block_size=4,
        dtype=jnp.float32,
    )

    seen: list[int] = []
    real_decode = eng.decode

    def recording_decode(
        model,
        state,
        last_tokens,
        positions,
        valid_rows,
        block_tables,
        phys_block,
        slot_in_block,
    ):
        seen.append(int(block_tables.shape[1]))
        return real_decode(
            model,
            state,
            last_tokens,
            positions,
            valid_rows,
            block_tables,
            phys_block,
            slot_in_block,
        )

    monkeypatch.setattr(eng, "decode", recording_decode)
    eng.add_request(driver, [1, 2, 3, 4], SamplingParams(max_new_tokens=2))

    eng.step(driver)
    eng.step(driver)

    assert seen == [2]


def test_stream_raises_after_driver_failure():
    model = make_tiny_model()
    driver = eng.make_driver(
        model,
        max_num_seqs=1,
        max_model_len=16,
        max_prefill_len=8,
        block_size=4,
        dtype=jnp.float32,
    )

    request_id = eng.add_request(driver, [1, 2, 3, 4], SamplingParams(max_new_tokens=1))
    driver.thread_error = "RuntimeError: boom"

    with pytest.raises(RuntimeError, match="engine thread failed"):
        next(eng.stream(driver, request_id))


def test_add_request_raises_after_driver_failure():
    model = make_tiny_model()
    driver = eng.make_driver(
        model,
        max_num_seqs=1,
        max_model_len=16,
        max_prefill_len=8,
        block_size=4,
        dtype=jnp.float32,
    )
    driver.thread_error = "RuntimeError: boom"

    with pytest.raises(RuntimeError, match="engine thread failed"):
        eng.add_request(driver, [1, 2, 3, 4], SamplingParams(max_new_tokens=1))
