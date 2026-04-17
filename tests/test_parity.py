import os

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jllm.model.qwen2 import forward
from jllm.model.weights import load_from_path

MODEL_PATH = os.environ.get("PARITY_MODEL", "weights/Qwen2.5-0.5B-Instruct")
PROMPTS = [
    "The capital of France is",
    "In a shocking discovery, scientists found",
    "def fibonacci(n):",
]


@pytest.fixture(scope="module")
def hf_model():
    return AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).eval()


@pytest.fixture(scope="module")
def jx_model():
    return load_from_path(MODEL_PATH, dtype=jnp.bfloat16)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_PATH)


@pytest.mark.parametrize("prompt", PROMPTS)
def test_last_token_logits_match(prompt, hf_model, jx_model, tokenizer):
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        hf_logits = hf_model(ids).logits[0, -1].to(torch.float32).cpu().numpy()

    jx_ids = jnp.asarray(ids.numpy())
    jx_logits = np.asarray(forward(jx_model, jx_ids)[0, -1].astype(jnp.float32))

    assert int(hf_logits.argmax()) == int(jx_logits.argmax()), "top-1 token mismatch"
    cos = float(hf_logits @ jx_logits) / (
        np.linalg.norm(hf_logits) * np.linalg.norm(jx_logits) + 1e-8
    )
    assert cos > 0.998, f"cos similarity too low: {cos}"
