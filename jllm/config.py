import os
from dataclasses import dataclass
from typing import Optional

VALID_ATTENTION_IMPLS = ("einsum", "sdpa")


@dataclass(frozen=True)
class JllmConfig:
    attention_impl: str = "einsum"
    model_path: Optional[str] = None
    compilation_cache_dir: Optional[str] = None
    persistent_cache_min_compile_time_secs: Optional[str] = None
    persistent_cache_min_entry_size_bytes: Optional[str] = None

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
            persistent_cache_min_compile_time_secs=os.environ.get(
                "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"
            ),
            persistent_cache_min_entry_size_bytes=os.environ.get(
                "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"
            ),
        )


def apply_jax_env(cfg: JllmConfig) -> None:
    if cfg.compilation_cache_dir:
        os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", cfg.compilation_cache_dir)
    os.environ.setdefault(
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
        cfg.persistent_cache_min_compile_time_secs or "0",
    )
    os.environ.setdefault(
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
        cfg.persistent_cache_min_entry_size_bytes or "0",
    )
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
