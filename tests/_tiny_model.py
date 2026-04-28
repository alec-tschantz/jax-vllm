import numpy as np
import jax.numpy as jnp

from jllm.model.common import (
    Attention,
    DecoderLayer,
    Embedding,
    Linear,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
)
from jllm.model.qwen2 import Qwen2Config, Qwen2Model


def _rand(rng: np.random.Generator, *shape: int) -> jnp.ndarray:
    return jnp.asarray(rng.standard_normal(shape).astype(np.float32) * 0.02)


def make_tiny_model() -> Qwen2Model:
    rng = np.random.default_rng(0)
    kv_hidden = 8
    cfg = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=4,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )

    layers = []
    for _ in range(cfg.num_hidden_layers):
        layers.append(
            DecoderLayer(
                self_attn=Attention(
                    q_proj=Linear(
                        weight=_rand(rng, cfg.hidden_size, cfg.hidden_size),
                        bias=_rand(rng, cfg.hidden_size),
                    ),
                    k_proj=Linear(
                        weight=_rand(rng, kv_hidden, cfg.hidden_size),
                        bias=_rand(rng, kv_hidden),
                    ),
                    v_proj=Linear(
                        weight=_rand(rng, kv_hidden, cfg.hidden_size),
                        bias=_rand(rng, kv_hidden),
                    ),
                    o_proj=Linear(
                        weight=_rand(rng, cfg.hidden_size, cfg.hidden_size), bias=None
                    ),
                    q_norm=None,
                    k_norm=None,
                    num_heads=cfg.num_heads,
                    num_kv_heads=cfg.num_kv_heads,
                    head_dim=cfg.head_dim,
                ),
                mlp=SwiGLU(
                    gate_proj=Linear(
                        weight=_rand(rng, cfg.intermediate_size, cfg.hidden_size),
                        bias=None,
                    ),
                    up_proj=Linear(
                        weight=_rand(rng, cfg.intermediate_size, cfg.hidden_size),
                        bias=None,
                    ),
                    down_proj=Linear(
                        weight=_rand(rng, cfg.hidden_size, cfg.intermediate_size),
                        bias=None,
                    ),
                ),
                input_layernorm=RMSNorm(
                    weight=jnp.ones((cfg.hidden_size,), dtype=jnp.float32),
                    eps=cfg.rms_norm_eps,
                ),
                post_attention_layernorm=RMSNorm(
                    weight=jnp.ones((cfg.hidden_size,), dtype=jnp.float32),
                    eps=cfg.rms_norm_eps,
                ),
            )
        )

    return Qwen2Model(
        embed_tokens=Embedding(weight=_rand(rng, cfg.vocab_size, cfg.hidden_size)),
        layers=layers,
        norm=RMSNorm(
            weight=jnp.ones((cfg.hidden_size,), dtype=jnp.float32), eps=cfg.rms_norm_eps
        ),
        rotary_emb=RotaryEmbedding(dim=cfg.head_dim, theta=cfg.rope_theta),
        lm_head=Linear(weight=_rand(rng, cfg.vocab_size, cfg.hidden_size), bias=None),
        cfg=cfg,
    )
