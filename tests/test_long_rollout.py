"""End-to-end parity: greedy-generate N tokens with HF vs JX in fp32 must match exactly."""

import pytest

from ._greedy import greedy

pytestmark = pytest.mark.parity

N_NEW = 40

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time",
]

@pytest.mark.parametrize("prompt", PROMPTS)
def test_greedy_matches_hf_exactly(prompt, tokenizer, hf_model_fp32, jx_model_fp32):
    torch = pytest.importorskip("torch")
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        hf_out = hf_model_fp32.generate(
            ids,
            max_new_tokens=N_NEW,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.0,
        )[0].tolist()

    prompt_ids = ids[0].tolist()
    new_ids = greedy(jx_model_fp32, prompt_ids, max_new_tokens=N_NEW)
    jx_out = prompt_ids + new_ids

    assert hf_out == jx_out, f"first diff: {_first_diff(hf_out, jx_out)}"


def _first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return (i, x, y)
    return None
