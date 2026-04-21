import importlib
import sys

import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _restore_common(monkeypatch):
    yield
    monkeypatch.setenv("JLLM_ATTENTION_IMPL", "einsum")
    import jllm.model.common as common

    importlib.reload(common)


def _reload_common(monkeypatch, attention_impl: str):
    monkeypatch.setenv("JLLM_ATTENTION_IMPL", attention_impl)
    import jllm.model.common as common

    return importlib.reload(common)


@pytest.mark.parametrize("attention_impl", ["einsum", "sdpa"])
def test_attention_kernel_causal_matches_explicit_mask(monkeypatch, attention_impl):
    common = _reload_common(monkeypatch, attention_impl)
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.standard_normal((2, 4, 3, 2), dtype=np.float32))
    k = jnp.asarray(rng.standard_normal((2, 2, 5, 2), dtype=np.float32))
    v = jnp.asarray(rng.standard_normal((2, 2, 5, 2), dtype=np.float32))
    q_lens = jnp.asarray([3, 1], dtype=jnp.int32)
    k_lens = jnp.asarray([5, 2], dtype=jnp.int32)

    out_causal = common.attention_kernel_causal(
        q,
        k,
        v,
        head_dim=2,
        num_heads=4,
        num_kv_heads=2,
        q_lens=q_lens,
        k_lens=k_lens,
    )
    mask = common.causal_mask_from_lens(q_lens, k_lens, q.shape[2], k.shape[2])
    out_masked = common.attention_kernel(
        q,
        k,
        v,
        mask,
        head_dim=2,
        num_heads=4,
        num_kv_heads=2,
    )

    np.testing.assert_allclose(
        np.asarray(out_causal),
        np.asarray(out_masked),
        rtol=1e-5,
        atol=1e-5,
    )
