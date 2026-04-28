import os

import jax.numpy as jnp
import pytest

from jllm.model.weights import load_from_path

from ._weights import require_model_path

PARITY_MODEL = os.environ.get("PARITY_MODEL", "weights/Qwen2.5-0.5B-Instruct")


@pytest.fixture(scope="session")
def parity_model_path() -> str:
    return require_model_path(PARITY_MODEL)


@pytest.fixture(scope="session")
def tokenizer(parity_model_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(parity_model_path)


@pytest.fixture(scope="session")
def jx_model_fp32(parity_model_path: str):
    return load_from_path(parity_model_path, dtype=jnp.float32)


@pytest.fixture(scope="session")
def jx_model_bf16(parity_model_path: str):
    return load_from_path(parity_model_path, dtype=jnp.bfloat16)


@pytest.fixture(scope="session")
def hf_model_fp32(parity_model_path: str):
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        parity_model_path, dtype=torch.float32
    ).eval()


@pytest.fixture(scope="session")
def hf_model_bf16(parity_model_path: str):
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        parity_model_path, dtype=torch.bfloat16
    ).eval()
