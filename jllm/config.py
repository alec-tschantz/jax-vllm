"""Runtime configuration loaded from environment variables.

Call `apply_jax_env(JllmConfig.from_env())` at program start, BEFORE any jax
imports, so `JAX_COMPILATION_CACHE_DIR` and friends take effect.

Env vars:
    JLLM_ATTENTION_IMPL          einsum (default) | sdpa | aiter
    JLLM_MODEL_PATH              mirrors --model-path (for Docker deployment)
    JAX_COMPILATION_CACHE_DIR    persistent JIT artifact cache (45-60s → 2-5s
                                 on warm server restart)
    XLA_PYTHON_CLIENT_PREALLOCATE    defaulted to "false" so we don't grab all
                                     GPU memory on startup
    XLA_PYTHON_CLIENT_MEM_FRACTION   defaulted to "0.90"
"""
import os
from dataclasses import dataclass
from typing import Optional

VALID_ATTENTION_IMPLS = ("einsum", "sdpa", "aiter")


@dataclass(frozen=True)
class JllmConfig:
    attention_impl: str = "einsum"
    model_path: Optional[str] = None
    compilation_cache_dir: Optional[str] = None

    def __post_init__(self):
        if self.attention_impl not in VALID_ATTENTION_IMPLS:
            raise ValueError(
                f"JLLM_ATTENTION_IMPL={self.attention_impl!r} not in {VALID_ATTENTION_IMPLS}"
            )

    @classmethod
    def from_env(cls) -> "JllmConfig":
        return cls(
            attention_impl=os.environ.get("JLLM_ATTENTION_IMPL", "einsum"),
            model_path=os.environ.get("JLLM_MODEL_PATH"),
            compilation_cache_dir=os.environ.get("JAX_COMPILATION_CACHE_DIR"),
        )


def apply_jax_env(cfg: JllmConfig) -> None:
    """Write JAX env vars. Must run before `import jax` anywhere in the process.

    Uses setdefault so anything already set by the user/environment wins.
    """
    if cfg.compilation_cache_dir:
        os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", cfg.compilation_cache_dir)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
