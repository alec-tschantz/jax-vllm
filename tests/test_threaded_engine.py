import threading

import jax.numpy as jnp
import pytest

from jllm.engine import engine
from jllm.engine.request import SamplingParams

pytestmark = pytest.mark.engine

PROMPTS = [
    [785, 6722, 315, 9625, 374],
    [750, 79683, 1445, 982],
    [12522, 5193, 264, 882],
    [9454, 8420, 11, 1246, 646],
]
MAX_NEW = 6


def test_threaded_requests_produce_correct_outputs(jx_model_fp32):
    driver = engine.make_driver(
        jx_model_fp32, max_num_seqs=2, max_model_len=32, dtype=jnp.float32
    )
    engine.start(driver)
    try:
        solo_outputs: list[list[int]] = []
        for p in PROMPTS:
            rid = engine.add_request(driver, p, SamplingParams(max_new_tokens=MAX_NEW))
            solo_outputs.append([ev.token for ev in engine.stream(driver, rid)])

        results: dict[int, list[int]] = {}
        results_lock = threading.Lock()

        def worker(idx: int, prompt: list[int]) -> None:
            rid = engine.add_request(
                driver, prompt, SamplingParams(max_new_tokens=MAX_NEW)
            )
            out = [ev.token for ev in engine.stream(driver, rid)]
            with results_lock:
                results[idx] = out

        threads = [
            threading.Thread(target=worker, args=(i, p)) for i, p in enumerate(PROMPTS)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert len(results) == len(PROMPTS), "some workers did not finish"
        for i, prompt in enumerate(PROMPTS):
            assert results[i] == solo_outputs[i], (
                f"thread {i} diverges from solo run\n"
                f"  solo:    {solo_outputs[i]}\n"
                f"  threaded: {results[i]}"
            )
    finally:
        engine.stop(driver)
