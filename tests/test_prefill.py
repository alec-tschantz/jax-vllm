import numpy as np
import jax.numpy as jnp

from jllm.engine.state import init_state, prefill
from jllm.model.qwen2 import forward

from ._tiny_model import make_tiny_model


def test_prefill_returns_next_token_from_last_valid_position():
    model = make_tiny_model()
    state = init_state(model.cfg, block_size=4, num_blocks=8, dtype=jnp.float32)

    chunk_ids = jnp.asarray(
        [
            [1, 2, 3, 4],
            [5, 6, 0, 0],
        ],
        dtype=jnp.int32,
    )
    pos_starts = jnp.asarray([0, 0], dtype=jnp.int32)
    block_tables = jnp.asarray([[1], [2]], dtype=jnp.int32)
    valid_tokens = jnp.asarray([4, 2], dtype=jnp.int32)
    last_token_idx = jnp.asarray([3, 1], dtype=jnp.int32)
    phys_blocks = jnp.asarray([1, 2], dtype=jnp.int32)

    _, next_toks = prefill(
        model,
        state,
        chunk_ids,
        pos_starts,
        block_tables,
        valid_tokens,
        last_token_idx,
        phys_blocks,
    )

    ref_full = np.asarray(forward(model, jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)))
    ref_short = np.asarray(forward(model, jnp.asarray([[5, 6]], dtype=jnp.int32)))

    assert next_toks.shape == (2,)
    assert np.asarray(next_toks).tolist() == [
        int(ref_full[0, -1].argmax()),
        int(ref_short[0, -1].argmax()),
    ]
