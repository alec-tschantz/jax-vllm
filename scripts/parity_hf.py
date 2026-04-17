"""HF vs jllm parity checks. Two modes:

- logits: last-token logit diff, argmax match, cos similarity. Multi-prompt,
          configurable dtype + model (works for Qwen2.5-0.5B and -32B).
- layer:  layer-by-layer hidden-state diff. Helps bisect where our model
          diverges from HF's reference.

Requires the `parity` extra (torch + transformers CPU). On gpu-droplet this
means `uv sync --extra parity` — uninstalls the rocm-jax extras! Run parity
checks on a separate host or in a dedicated venv.

  uv run python scripts/parity_hf.py --mode logits
  uv run python scripts/parity_hf.py --mode layer --prompt "The capital of France is Paris. It"
"""
import argparse
import time

import jax.numpy as jnp
import numpy as np

DEFAULT_PROMPTS = [
    "The capital of France is",
    "In a shocking discovery, scientists found",
    "def fibonacci(n):",
]


def _forward_for(model):
    """Dispatch to the right forward() by model class — either arch module exposes
    its own `forward` function (not a method on the eqx.Module)."""
    from jllm.model.qwen2 import Qwen2Model, forward as forward_qwen2
    from jllm.model.qwen3 import Qwen3Model, forward as forward_qwen3
    if isinstance(model, Qwen2Model):
        return forward_qwen2
    if isinstance(model, Qwen3Model):
        return forward_qwen3
    raise ValueError(f"no forward() for model type {type(model).__name__}")


def run_logits(args) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jllm.model.weights import load_from_path

    tok = AutoTokenizer.from_pretrained(args.model_path)

    print(f"loading HF (CPU, {args.dtype})...")
    t0 = time.perf_counter()
    hf_dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    hf = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=hf_dtype).eval()
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    print(f"loading jllm ({args.dtype})...")
    t0 = time.perf_counter()
    jx_dtype = jnp.float32 if args.dtype == "fp32" else jnp.bfloat16
    jx = load_from_path(args.model_path, dtype=jx_dtype)
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    max_abs_overall = 0.0
    mismatches = 0
    prompts = args.prompts or DEFAULT_PROMPTS
    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt").input_ids
        t0 = time.perf_counter()
        with torch.no_grad():
            hf_logits = hf(ids).logits[0, -1].to(torch.float32).cpu().numpy()
        hf_dt = time.perf_counter() - t0

        jx_ids = jnp.asarray(ids.numpy())
        t0 = time.perf_counter()
        forward = _forward_for(jx)
        out = forward(jx, jx_ids)
        jx_logits = np.asarray(out[0, -1].astype(jnp.float32))
        jx_dt = time.perf_counter() - t0

        d = float(np.abs(hf_logits - jx_logits).max())
        mean_d = float(np.abs(hf_logits - jx_logits).mean())
        cos = float(hf_logits @ jx_logits) / (np.linalg.norm(hf_logits) * np.linalg.norm(jx_logits) + 1e-8)
        hf_top = int(hf_logits.argmax())
        jx_top = int(jx_logits.argmax())
        ok = hf_top == jx_top
        if not ok:
            mismatches += 1
        print(
            f"prompt={prompt!r:42s}  HF {hf_dt:5.1f}s  jllm {jx_dt:5.1f}s  "
            f"max|Δ|={d:.4f}  mean|Δ|={mean_d:.5f}  cos={cos:.6f}  "
            f"HF_top={hf_top} jllm_top={jx_top}  {'OK' if ok else 'MISMATCH'}"
        )
        max_abs_overall = max(max_abs_overall, d)

    print(f"\nmax|Δ| overall: {max_abs_overall:.4f}  mismatches: {mismatches}/{len(prompts)}")
    return 0 if mismatches == 0 else 2


def run_layer(args) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jllm.model.common import embed, linear, rms_norm, rope_cos_sin
    from jllm.model.weights import load_from_path
    from jllm.model.qwen2 import Qwen2Model, decoder_layer as dl_qwen2
    from jllm.model.qwen3 import Qwen3Model, decoder_layer as dl_qwen3

    tok = AutoTokenizer.from_pretrained(args.model_path)
    ids_pt = tok(args.prompt, return_tensors="pt").input_ids
    ids_np = ids_pt.numpy()

    # fp32 everywhere — this is a numerical bisect, not a perf test
    hf = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float32).eval()
    with torch.no_grad():
        hf_out = hf(ids_pt, output_hidden_states=True)
    hf_hidden = [h.cpu().numpy() for h in hf_out.hidden_states]
    hf_logits = hf_out.logits.cpu().numpy()

    jx = load_from_path(args.model_path, dtype=jnp.float32)
    dl = dl_qwen2 if isinstance(jx, Qwen2Model) else dl_qwen3
    B, T = ids_np.shape
    position_ids = jnp.broadcast_to(jnp.arange(T), (B, T))
    attention_mask = jnp.ones((B, T), dtype=bool)
    causal = jnp.tril(jnp.ones((T, T), dtype=bool))
    mask = causal[None, None, :, :] & attention_mask[:, None, None, :]
    hidden = embed(jx.embed_tokens, jnp.asarray(ids_np, dtype=jnp.int32))
    cos_arr, sin_arr = rope_cos_sin(jx.rotary_emb, position_ids, hidden.dtype)
    jx_hidden = [np.asarray(hidden)]
    for layer in jx.layers:
        hidden = dl(layer, hidden, cos_arr, sin_arr, mask)
        jx_hidden.append(np.asarray(hidden))
    jx_final = rms_norm(jx.norm, hidden)
    # HF's `output_hidden_states` returns the post-final-norm tensor as the last
    # entry — replace our pre-norm last entry with the normalized version so the
    # layer-by-layer diff is apples-to-apples.
    jx_hidden[-1] = np.asarray(jx_final)
    jx_logits = np.asarray(linear(jx.lm_head, jx_final))

    print(f"prompt: {args.prompt!r}  ({T} tokens)")
    print(f"hidden stages HF: {len(hf_hidden)}, jllm: {len(jx_hidden)}")
    print(f"\n{'layer':>6}  {'max|Δ|':>10}  {'mean|Δ|':>10}  {'cos':>10}")
    for i, (h, j) in enumerate(zip(hf_hidden, jx_hidden)):
        h_last = h[0, -1]
        j_last = j[0, -1]
        d = float(np.abs(h_last - j_last).max())
        m = float(np.abs(h_last - j_last).mean())
        cs = float(h_last @ j_last) / (np.linalg.norm(h_last) * np.linalg.norm(j_last) + 1e-8)
        tag = "embed" if i == 0 else f"L{i-1}"
        print(f"{tag:>6}  {d:>10.6f}  {m:>10.6f}  {cs:>10.6f}")

    last_hf = hf_logits[0, -1]
    last_jx = jx_logits[0, -1]
    d = float(np.abs(last_hf - last_jx).max())
    cs = float(last_hf @ last_jx) / (np.linalg.norm(last_hf) * np.linalg.norm(last_jx) + 1e-8)
    print(f"\nfinal logits: max|Δ|={d:.6f} cos={cs:.6f}")
    print(f"  HF argmax={int(last_hf.argmax())}  jllm argmax={int(last_jx.argmax())}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["logits", "layer"], default="logits")
    p.add_argument("--model-path", default="weights/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--prompt", default="The capital of France is Paris. It",
                   help="Single prompt for --mode layer")
    p.add_argument("--prompts", nargs="+", default=None,
                   help="Override DEFAULT_PROMPTS for --mode logits")
    args = p.parse_args()
    if args.mode == "logits":
        return run_logits(args)
    return run_layer(args)


if __name__ == "__main__":
    import sys
    sys.exit(main())
