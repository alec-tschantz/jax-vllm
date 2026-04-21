import jax.numpy as jnp
import numpy as np
import pytest

from jllm.model.qwen2 import forward

pytestmark = pytest.mark.parity

PROMPTS = [
    "The capital of France is",
    "In a shocking discovery, scientists found",
    "def fibonacci(n):",
]


@pytest.mark.parametrize("prompt", PROMPTS)
def test_last_token_logits_match(prompt, hf_model_bf16, jx_model_bf16, tokenizer):
    torch = pytest.importorskip("torch")
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        hf_logits = hf_model_bf16(ids).logits[0, -1].to(torch.float32).cpu().numpy()

    jx_ids = jnp.asarray(ids.numpy())
    jx_logits = np.asarray(forward(jx_model_bf16, jx_ids)[0, -1].astype(jnp.float32))

    assert int(hf_logits.argmax()) == int(jx_logits.argmax()), "top-1 token mismatch"
    cos = float(hf_logits @ jx_logits) / (
        np.linalg.norm(hf_logits) * np.linalg.norm(jx_logits) + 1e-8
    )
    assert cos > 0.998, f"cos similarity too low: {cos}"
