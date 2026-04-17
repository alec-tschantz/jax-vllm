# jax-vllm (`jllm`)

A minimal, functional reimplementation of the core ideas behind vLLM in pure JAX:
**continuous batching**, **paged KV cache**, **chunked prefill**, and **content-hash
prefix caching**. Supports **Qwen2 / Qwen2.5** and **Qwen3 / Qwen3.5** (adding a new
decoder-only family is a one-file addition under `jllm/model/`). Runs Qwen2.5-32B on
a single AMD MI300X via the ROCm JAX backend; smaller checkpoints also work on CPU.

## Install

```sh
uv sync --extra dev                    # CPU / general dev
uv sync --extra dev --extra rocm       # AMD MI300X
uv sync --extra dev --extra parity     # adds torch-cpu for HF parity tests
```

## Run

```sh
uv run jllm-serve --model-path weights/Qwen2.5-32B-Instruct --port 8080   # start the server
uv run python cli.py --url http://localhost:8080                          # streaming chat CLI
```

The server dispatches on the HF `architectures` field — `Qwen2ForCausalLM` or
`Qwen3ForCausalLM` — so `--model-path` can point at either family.

First request JIT-compiles the two kernels (`extend_step`, `decode_step_cb`); later
requests reuse the compilation. The server also exposes `POST /v1/chat/completions` (OpenAI-
shaped, applies the chat template, stops on `<|im_end|>`) and `POST /v1/completions`.

### Environment variables

| Var | Default | Effect |
|---|---|---|
| `JAX_COMPILATION_CACHE_DIR` | unset | Persist the JIT compile across server restarts. Setting this drops warm-start first-request latency from ~50 s → ~5 s. Mount as a Docker volume to survive container rebuilds. |
| `JLLM_ATTENTION_IMPL` | `einsum` | Attention backend: `einsum` / `sdpa` / `aiter`. A/B without a rebuild. |
| `JLLM_MODEL_PATH` | unset | Default `--model-path` (useful in Docker). |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` | Auto-set by jllm. Stops XLA from grabbing all GPU memory on startup. |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `0.90` | Auto-set by jllm. |

## Architecture

One extend kernel and one decode kernel handle all work. Both JIT-compile exactly once
per config.

- **Paged KV cache.** `PagedCache` is a pool shaped `[num_blocks, block_size, H_kv, D]`.
  Each slot owns a row of `block_tables[slot, NB_MAX]` mapping logical → physical block.
  Block 0 is a reserved sentinel for idle-slot scatters.
- **Chunked prefill.** `extend_step_jit` advances one slot by `block_size` tokens per
  call: scatter the chunk into one fresh block, gather the prefix via `block_tables`,
  causal attention. Any prompt length runs through N calls at the same JIT shape.
- **Fused decode.** `decode_step_cb_jit` writes one token per slot across all decoding
  slots in a single call. Per-slot `positions[b]` lets sequences at different ages share
  the same GPU step.
- **Prefix caching.** `BlockManager` hashes each full block as
  `sha256(parent_hash || token_ids)` (Merkle chain), keyed into a `hash → block_id`
  map. Admit looks up the prompt's blocks, `touch`es hits (ref count up, unlink from
  free queue). LRU eviction + v1-style duplicate tolerance match vLLM's APC design.
- **Interleaved scheduler.** Each `step()` advances one prefilling slot by one chunk
  (round-robin), then runs one fused decode over the decoding slots.
- **Pure functional state.** `EngineState = (cache, positions, last_tokens, block_tables)`
  is a frozen pytree; every transition returns a new state.

Attention: Qwen2 / Qwen3 GQA (e.g. 40 Q heads, 8 KV heads at Qwen2.5-32B). An explicit
`jnp.repeat` + einsum-softmax-einsum beats `jax.nn.dot_product_attention` by 5–13% at
B ≤ 4 on ROCm; toggle via the `JLLM_ATTENTION_IMPL` env var. Softmax and RMSNorm
accumulate in fp32 — matches HF's fp32 greedy output token-for-token on 0.5B / 0.6B.
Qwen3 additionally applies an RMSNorm to Q and K per-head before RoPE (`q_norm` /
`k_norm`); Qwen2 leaves those fields as `None`, and both archs share the same
`Attention` and `DecoderLayer` dataclasses.

## Tests

```sh
uv run pytest tests/                                                   # requires weights/
PARITY_MODEL=weights/Qwen3-0.6B uv run pytest tests/                   # run against Qwen3
```

22 tests: HF parity (logits + long rollout), paged primitives (scatter/gather, LRU,
hash determinism), engine determinism, slot-output independence of batchmate,
engine-vs-solo parity, threaded concurrent requests. Feature-engagement assertions use
a `driver.n_extend_calls` counter — a same-prompt-twice test requires `warm == 0`
extend_step calls to guard against regressions that silently disable prefix caching.
The `PARITY_MODEL` env var parameterises the HF-comparison tests; runs clean for both
`Qwen2.5-0.5B-Instruct` and `Qwen3-0.6B`.

## Benchmarks vs vLLM

`scripts/bench_vllm.py` runs jllm and vLLM (at `--vllm-url`, default
`http://127.0.0.1:8020`) side by side:

```sh
uv run python scripts/bench_vllm.py --mode sequential                              # greedy parity
uv run python scripts/bench_vllm.py --mode concurrent --concurrency 1 4 16         # throughput
uv run python scripts/bench_vllm.py --mode chat                                    # multi-turn parity
uv run python scripts/bench_vllm.py --mode prefix_cache --prefix-len 256           # cache-hit TTFT
```

Representative on MI300X, Qwen2.5-32B bf16, 40 new tokens:

| concurrency | jllm tok/s | vLLM tok/s | ratio |
|---|---|---|---|
| 1 | 27 | 57 | 47% |
| 4 | 98 | 187 | 52% |

The gap is kernel-level (FlashAttention, fused QKV/RoPE/RMSNorm, graph capture) — not
cache architecture. Where paging pays off: the `prefix_cache` mode shows warm TTFT
dropping to ~0.11 s on cached prefixes vs ~0.35 s uncached.

## Repo layout

```
jllm/
  config.py             # JllmConfig + apply_jax_env (reads JLLM_* / JAX_* env)
  model/common.py       # shared: Attention, DecoderLayer, Linear, RMSNorm, ...
  model/qwen2.py        # Qwen2 / Qwen2.5 config + forward wiring
  model/qwen3.py        # Qwen3 / Qwen3.5 config + forward wiring (+ qk_norm)
  model/weights.py      # safetensors → pytree + arch dispatch (load_from_path)
  engine/paged.py       # PagedCache + BlockManager (hash, ref count, LRU)
  engine/generate.py    # extend_step_jit + decode_step_cb_jit
  engine/state.py       # EngineState + pure transitions
  engine/engine.py      # Driver + interleaved scheduler (threaded)
  engine/request.py     # Request, SamplingParams, StepEvent
  server.py             # FastAPI (`jllm-serve` console script)
cli.py                  # rich streaming chat CLI
scripts/                # sync.sh, download_model.py, bench_vllm.py, parity_hf.py
tests/
```

## Remote GPU workflow

We develop locally and run the server on a remote MI300X host. `scripts/sync.sh`
rsyncs the repo to `gpu-droplet:/root/jax-vllm/` (edit the host alias for your setup);
an SSH tunnel forwards the server port back to the laptop for the CLI.

```sh
scripts/sync.sh                                                 # local → remote
ssh gpu-droplet 'HIP_VISIBLE_DEVICES=0 uv run jllm-serve --port 8080 \
    --model-path weights/Qwen2.5-32B-Instruct'                  # on the remote
ssh -fN -L 8080:localhost:8080 gpu-droplet                      # from laptop
uv run python cli.py --url http://localhost:8080                # talk to it
```

For head-to-head vs vLLM, pin it to another GPU:

```sh
docker run --rm -d --name vllm-qwen32b --network host --ipc host \
  --device /dev/kfd --device /dev/dri --group-add video \
  --env HIP_VISIBLE_DEVICES=1 -v $(pwd)/weights:/weights:ro \
  rocm/vllm-dev:rocm7.2.1_v0.18.0_20260330 \
  vllm serve /weights/Qwen2.5-32B-Instruct --host 0.0.0.0 --port 8020 \
    --gpu-memory-utilization 0.85 --max-model-len 4096 --dtype bfloat16 \
    --served-model-name qwen32b
```
