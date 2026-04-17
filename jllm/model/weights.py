from pathlib import Path

from jax import numpy as jnp
from safetensors import safe_open

from .qwen2 import (
    Attention,
    DecoderLayer,
    Dense,
    Embedding,
    Linear,
    Qwen2Config,
    Qwen2Model,
    RMSNorm,
    RotaryEmbedding,
)


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


def load_qwen2(path: str | Path, dtype=jnp.bfloat16) -> Qwen2Model:
    path = Path(path)
    cfg = Qwen2Config.from_hf(path)
    t = _load_tensors(path)

    def arr(name: str) -> jnp.ndarray:
        return t[name].astype(dtype)

    def linear(prefix: str, bias: bool) -> Linear:
        return Linear(
            weight=arr(f"{prefix}.weight"),
            bias=arr(f"{prefix}.bias") if bias else None,
        )

    def rms(prefix: str) -> RMSNorm:
        return RMSNorm(weight=arr(f"{prefix}.weight"), eps=cfg.rms_norm_eps)

    layers = []
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        layers.append(
            DecoderLayer(
                self_attn=Attention(
                    q_proj=linear(f"{p}.self_attn.q_proj", bias=True),
                    k_proj=linear(f"{p}.self_attn.k_proj", bias=True),
                    v_proj=linear(f"{p}.self_attn.v_proj", bias=True),
                    o_proj=linear(f"{p}.self_attn.o_proj", bias=False),
                    num_heads=cfg.num_heads,
                    num_kv_heads=cfg.num_kv_heads,
                    head_dim=cfg.head_dim,
                ),
                mlp=Dense(
                    gate_proj=linear(f"{p}.mlp.gate_proj", bias=False),
                    up_proj=linear(f"{p}.mlp.up_proj", bias=False),
                    down_proj=linear(f"{p}.mlp.down_proj", bias=False),
                ),
                input_layernorm=rms(f"{p}.input_layernorm"),
                post_attention_layernorm=rms(f"{p}.post_attention_layernorm"),
            )
        )

    embed = Embedding(weight=arr("model.embed_tokens.weight"))
    if cfg.tie_word_embeddings:
        lm_head = Linear(weight=embed.weight, bias=None)
    else:
        lm_head = Linear(weight=arr("lm_head.weight"), bias=None)

    return Qwen2Model(
        embed_tokens=embed,
        layers=layers,
        norm=rms("model.norm"),
        rotary_emb=RotaryEmbedding(dim=cfg.head_dim, theta=cfg.rope_theta),
        lm_head=lm_head,
        cfg=cfg,
    )
