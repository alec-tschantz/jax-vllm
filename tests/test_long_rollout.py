"""End-to-end parity: greedy-generate N tokens with HF vs JX in fp32 must match exactly."""

import os

import jax.numpy as jnp
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jllm.model.weights import load_qwen2

from ._greedy import greedy

MODEL_PATH = os.environ.get("PARITY_MODEL", "weights/Qwen2.5-0.5B-Instruct")
N_NEW = 40

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time",
]


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_PATH)


@pytest.fixture(scope="module")
def hf_model():
    return AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32).eval()


@pytest.fixture(scope="module")
def jx_model():
    return load_qwen2(MODEL_PATH, dtype=jnp.float32)


@pytest.mark.parametrize("prompt", PROMPTS)
def test_greedy_matches_hf_exactly(prompt, tokenizer, hf_model, jx_model):
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        hf_out = hf_model.generate(
            ids,
            max_new_tokens=N_NEW,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.0,
        )[0].tolist()

    prompt_ids = ids[0].tolist()
    new_ids = greedy(jx_model, prompt_ids, max_new_tokens=N_NEW)
    jx_out = prompt_ids + new_ids

    assert hf_out == jx_out, f"first diff: {_first_diff(hf_out, jx_out)}"


def _first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return (i, x, y)
    return None
