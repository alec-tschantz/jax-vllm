import os

from jllm.config import JllmConfig, apply_jax_env


def test_apply_jax_env_sets_persistent_cache_defaults(monkeypatch):
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", raising=False)
    monkeypatch.delenv("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", raising=False)

    apply_jax_env(JllmConfig(compilation_cache_dir="/tmp/jllm-cache"))

    assert os.environ["JAX_COMPILATION_CACHE_DIR"] == "/tmp/jllm-cache"
    assert os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] == "0"
    assert os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] == "0"
