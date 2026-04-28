import json
from pathlib import Path
from typing import Callable, Union

from jax import numpy as jnp
from safetensors import safe_open

from .common import Attention, DecoderLayer, Embedding, Linear, RMSNorm, RotaryEmbedding, SwiGLU
from .qwen2 import Qwen2Config, Qwen2Model
from .qwen3 import Qwen3Config, Qwen3Model


def _load_tensors(path: Path) -> dict:
    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors in {path}")
    out = {}
    for shard in shards:
        with safe_open(shard, framework="flax") as f:
            for key in f.keys():
                out[key] = f.get_tensor(key)
    return out


def _arr_fn(tensors: dict, dtype) -> Callable[[str], jnp.ndarray]:
    def arr(name: str) -> jnp.ndarray:
        return tensors[name].astype(dtype)
    return arr


def _linear(arr: Callable[[str], jnp.ndarray], prefix: str, bias: bool) -> Linear:
    return Linear(
        weight=arr(f"{prefix}.weight"),
        bias=arr(f"{prefix}.bias") if bias else None,
    )


def _rms(arr: Callable[[str], jnp.ndarray], prefix: str, eps: float) -> RMSNorm:
    return RMSNorm(weight=arr(f"{prefix}.weight"), eps=eps)


def _mlp(arr: Callable[[str], jnp.ndarray], prefix: str) -> SwiGLU:
    return SwiGLU(
        gate_proj=_linear(arr, f"{prefix}.gate_proj", bias=False),
        up_proj=_linear(arr, f"{prefix}.up_proj", bias=False),
        down_proj=_linear(arr, f"{prefix}.down_proj", bias=False),
    )


def _read_arch(path: Path) -> str:
    cfg_path = path / "config.json" if path.is_dir() else path
    c = json.loads(cfg_path.read_text())
    archs = c.get("architectures") or []
    if not archs:
        raise ValueError(f"no 'architectures' field in {cfg_path}")
    return archs[0]


def load_qwen2(path: "str | Path", dtype=jnp.bfloat16) -> Qwen2Model:
    path = Path(path)
    cfg = Qwen2Config.from_hf(path)
    arr = _arr_fn(_load_tensors(path), dtype)

    layers = []
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        layers.append(
            DecoderLayer(
                self_attn=Attention(
                    q_proj=_linear(arr, f"{p}.self_attn.q_proj", bias=True),
                    k_proj=_linear(arr, f"{p}.self_attn.k_proj", bias=True),
                    v_proj=_linear(arr, f"{p}.self_attn.v_proj", bias=True),
                    o_proj=_linear(arr, f"{p}.self_attn.o_proj", bias=False),
                    q_norm=None,
                    k_norm=None,
                    num_heads=cfg.num_heads,
                    num_kv_heads=cfg.num_kv_heads,
                    head_dim=cfg.head_dim,
                ),
                mlp=_mlp(arr, f"{p}.mlp"),
                input_layernorm=_rms(arr, f"{p}.input_layernorm", cfg.rms_norm_eps),
                post_attention_layernorm=_rms(arr, f"{p}.post_attention_layernorm", cfg.rms_norm_eps),
            )
        )

    embed = Embedding(weight=arr("model.embed_tokens.weight"))
    lm_head = Linear(
        weight=embed.weight if cfg.tie_word_embeddings else arr("lm_head.weight"),
        bias=None,
    )

    return Qwen2Model(
        embed_tokens=embed,
        layers=layers,
        norm=_rms(arr, "model.norm", cfg.rms_norm_eps),
        rotary_emb=RotaryEmbedding(dim=cfg.head_dim, theta=cfg.rope_theta),
        lm_head=lm_head,
        cfg=cfg,
    )


def load_qwen3(path: "str | Path", dtype=jnp.bfloat16) -> Qwen3Model:
    path = Path(path)
    cfg = Qwen3Config.from_hf(path)
    arr = _arr_fn(_load_tensors(path), dtype)

    layers = []
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        layers.append(
            DecoderLayer(
                self_attn=Attention(
                    q_proj=_linear(arr, f"{p}.self_attn.q_proj", bias=False),
                    k_proj=_linear(arr, f"{p}.self_attn.k_proj", bias=False),
                    v_proj=_linear(arr, f"{p}.self_attn.v_proj", bias=False),
                    o_proj=_linear(arr, f"{p}.self_attn.o_proj", bias=False),
                    q_norm=_rms(arr, f"{p}.self_attn.q_norm", cfg.rms_norm_eps),
                    k_norm=_rms(arr, f"{p}.self_attn.k_norm", cfg.rms_norm_eps),
                    num_heads=cfg.num_heads,
                    num_kv_heads=cfg.num_kv_heads,
                    head_dim=cfg.head_dim,
                ),
                mlp=_mlp(arr, f"{p}.mlp"),
                input_layernorm=_rms(arr, f"{p}.input_layernorm", cfg.rms_norm_eps),
                post_attention_layernorm=_rms(arr, f"{p}.post_attention_layernorm", cfg.rms_norm_eps),
            )
        )

    embed = Embedding(weight=arr("model.embed_tokens.weight"))
    lm_head = Linear(
        weight=embed.weight if cfg.tie_word_embeddings else arr("lm_head.weight"),
        bias=None,
    )

    return Qwen3Model(
        embed_tokens=embed,
        layers=layers,
        norm=_rms(arr, "model.norm", cfg.rms_norm_eps),
        rotary_emb=RotaryEmbedding(dim=cfg.head_dim, theta=cfg.rope_theta),
        lm_head=lm_head,
        cfg=cfg,
    )


Model = Union[Qwen2Model, Qwen3Model]


def load_from_path(path: "str | Path", dtype=jnp.bfloat16) -> Model:
    arch = _read_arch(Path(path))
    if arch == "Qwen2ForCausalLM":
        return load_qwen2(path, dtype=dtype)
    if arch == "Qwen3ForCausalLM":
        return load_qwen3(path, dtype=dtype)
    raise ValueError(
        f"unsupported architecture: {arch!r}. "
        f"Supported: Qwen2ForCausalLM, Qwen3ForCausalLM"
    )
