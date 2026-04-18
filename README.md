# jax-vllm (`jllm`)

A compact JAX implementation of the core serving ideas behind vLLM: continuous
batching, paged KV caching, chunked prefill, and content-addressed prefix
caching. The project is intentionally small, but it is structured as a serious
serving engine rather than a notebook prototype: the hot path is explicit, the
state boundaries are narrow, and the benchmark loop is built into the repo.

`jllm` currently targets decoder-only transformer families with the shared
interface defined in [`jllm/model/common.py`](jllm/model/common.py). The engine
does not depend on Qwen-specific classes; it depends only on the minimal model
surface required for embedding, rotary attention, decoder layers, and LM head
projection.

## Design Overview

The implementation follows five design decisions.

| Choice | Rationale |
|---|---|
| **Paged KV cache** | Keeps allocation stable and allows prompt sharing through physical block reuse. |
| **Content-hash prefix caching** | Reuses full prompt blocks by content, not by request identity. |
| **Chunked prefill** | Makes prefill shape-stable and amortizes compilation across arbitrary prompt lengths. |
| **Packed active batching** | Decode and prefill run on the active slot set rather than always paying for `max_num_seqs`. |
| **Functional runtime state** | Host-side runtime arrays are copy-on-write, which keeps scheduler reasoning simple and testable. |

The engine remains deliberately conservative in scope: no custom kernels, no
speculative decoding, and no architecture-specific scheduling branches in the
engine layer. Performance work is therefore concentrated on scheduling,
batch-shaping, cache reuse, and JAX/XLA behavior.

## Architecture

The codebase is split along three boundaries.

1. **Model interface**. `jllm/model/common.py` defines the shared decoder-only
   primitives and the abstract surface the engine consumes.
2. **Engine runtime**. `jllm/engine/engine.py`, `runtime.py`, and `cache.py`
   own request admission, scheduling, prefix-cache bookkeeping, and streaming.
3. **JAX kernels**. `jllm/engine/generate.py` and `paged.py` own the paged
   cache layout plus the batched prefill/decode kernels.

The resulting separation is intentional:

- The engine knows about slots, block tables, and cache policy.
- The model layer knows about attention, RoPE, layer structure, and weight
  layout.
- The kernel layer knows how to combine a generic decoder model with paged KV
  storage.

## Installation

```sh
uv sync --extra dev
uv sync --extra dev --extra rocm
uv sync --extra dev --extra parity
```

The `parity` extra installs the Hugging Face / PyTorch dependency set used by
the parity tests. It is useful on CPU machines or in a dedicated environment;
on ROCm hosts you may prefer to keep it separate from the serving environment.

## Running the Server

```sh
uv run jllm-serve --model-path weights/Qwen2.5-32B-Instruct --port 8080
uv run python cli.py --url http://localhost:8080
```

The server exposes:

- `POST /generate`
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `GET /health`

`/health` also reports the configured attention backend and engine counters,
which is useful when comparing runs.

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `JLLM_ATTENTION_IMPL` | `einsum` | Attention backend: `einsum` or `sdpa`. |
| `JLLM_MODEL_PATH` | unset | Default model path for `jllm-serve`. |
| `JAX_COMPILATION_CACHE_DIR` | unset | Persist XLA compilations across restarts. |
| `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS` | `0` | Cache even short compiles so repeated experiment lanes warm quickly. |
| `JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES` | `0` | Persist small compiled artifacts instead of only larger entries. |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` | Prevent eager full-device reservation. |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `0.90` | Default memory ceiling for XLA allocation. |

## Benchmarking

`scripts/bench_vllm.py` compares `jllm` and vLLM using the same prompt sets.
It supports both a running HTTP server and an in-process `jllm` engine.

```sh
uv run python scripts/bench_vllm.py --mode sequential
uv run python scripts/bench_vllm.py --mode concurrent --concurrency 1 4 16
uv run python scripts/bench_vllm.py --mode concurrent --workload mixed --concurrency 1 2 4
uv run python scripts/bench_vllm.py --mode chat
uv run python scripts/bench_vllm.py --mode prefix_cache --prefix-len 256
```

Structured output can be written with:

```sh
uv run python scripts/bench_vllm.py \
  --mode concurrent \
  --workload mixed \
  --concurrency 1 4 16 \
  --json-out results/concurrent.json \
  --label mi300x-nightly
```

The JSON payload includes:

- git SHA
- benchmark label
- workload profile and prompt-length summary
- attention backend
- server metadata when available
- engine stats deltas for the timed run
- throughput / latency summaries
- cold vs warm prefix-cache measurements where relevant

## Remote ROCm Workflow

The intended remote workflow is deliberately simple:

1. Sync the repo to the GPU host.
2. SSH into the host.
3. Launch one or more isolated GPU lanes for `jllm` and vLLM.
4. Run benchmark commands against the lane ports you care about.

The sync step is parameterized through environment variables rather than
hard-coded paths.

| Variable | Default | Purpose |
|---|---|---|
| `JLLM_REMOTE` | `gpu-droplet` | SSH host alias used by `scripts/sync.sh` |
| `JLLM_REMOTE_DIR` | `/root/jax-vllm` | Remote target directory for sync |

Example:

```sh
export JLLM_REMOTE=gpu-droplet
export JLLM_REMOTE_DIR=/root/jax-vllm

bash scripts/sync.sh
ssh gpu-droplet
cd /root/jax-vllm
JLLM_GPU=0 JLLM_PORT=8080 bash scripts/run_jllm_lane.sh
VLLM_GPU=1 VLLM_PORT=8020 bash scripts/run_vllm_lane.sh
```

Each lane chooses its own persistent JAX cache directory and log file, so warm
restarts and parallel experiments do not trample one another. The lane helpers
default to:

- `JAX_COMPILATION_CACHE_DIR=/tmp/jllm-jax-cache/gpu<gpu>-port<port>`
- `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0`
- `JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0`

From another shell on the remote host:

```sh
cd /root/jax-vllm
UV_CACHE_DIR=.uv-cache uv run python scripts/bench_vllm.py \
  --jllm-url http://127.0.0.1:8080 \
  --vllm-url http://127.0.0.1:8020 \
  --mode concurrent \
  --workload mixed \
  --concurrency 1 2 4 \
  --json-out results/concurrent.json
```

To add a second lane in parallel:

```sh
cd /root/jax-vllm
JLLM_GPU=2 JLLM_PORT=8182 bash scripts/run_jllm_lane.sh
VLLM_GPU=3 VLLM_PORT=9020 VLLM_MODEL_NAME=qwen32b-lane2 bash scripts/run_vllm_lane.sh
```

For inspection without side effects:

```sh
JLLM_DRY_RUN=1 bash scripts/sync.sh
JLLM_DRY_RUN=1 JLLM_GPU=2 JLLM_PORT=8182 bash scripts/run_jllm_lane.sh
VLLM_DRY_RUN=1 VLLM_GPU=3 VLLM_PORT=9020 bash scripts/run_vllm_lane.sh
```

The local machine in this development environment resolves `gpu-droplet` via
SSH configuration and can connect to it successfully, so the documented loop is
grounded in an actual reachable host rather than a placeholder alias.

## Testing

Fast local checks:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_paged.py tests/test_scheduler.py
```

Full suite:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest tests/
```

Weight-dependent tests skip automatically when the referenced checkpoint is not
present on disk. This keeps local development tight while still allowing full
parity validation on the remote model host.

The test suite currently covers:

- paged cache primitives and LRU behavior
- packed prefill/decode scheduling
- prefill-before-decode scheduling with mixed cached and uncached requests
- threaded request handling
- Hugging Face parity when model weights are available
- benchmark JSON, workload selection, and remote helper script smoke tests

## Repository Layout

```text
jllm/
  config.py
  server.py
  engine/
    engine.py
    runtime.py
    cache.py
    state.py
    generate.py
    paged.py
    request.py
  model/
    common.py
    qwen2.py
    qwen3.py
    weights.py
scripts/
  sync.sh
  bench_vllm.py
tests/
```

## Scope and Limitations

`jllm` is intended as a clean serving engine that is easy to study, extend, and
benchmark. It is not yet a kernel-level competitor to production vLLM builds.
The remaining performance gap is expected to come primarily from fused kernels,
flash attention variants, and graph-capture techniques that are intentionally
out of scope for this repository.

Within that scope, the project is meant to be explicit, extensible, and
measurable.
