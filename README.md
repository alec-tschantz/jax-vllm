# jax-vllm (`jllm`)

`jllm` is a compact JAX serving engine built around the core ideas behind
vLLM: continuous batching, paged KV caching, chunked prefill, and content-
addressed prefix reuse. The codebase is intentionally small, explicit, and easy
to benchmark.

The project currently targets decoder-only transformer families through the
shared interface in `jllm/model/common.py`. The engine depends on a narrow
model surface rather than Qwen-specific serving code.

## Design

The implementation is organized around three boundaries:

1. `jllm/model/`: shared decoder-only primitives plus model-family wiring
2. `jllm/engine/`: request admission, scheduling, runtime metadata, and cache policy
3. `jllm/engine/generate.py` + `jllm/engine/paged.py`: JAX kernels and paged KV layout

The serving path keeps host-side scheduler state separate from device-side KV
state:

- Host runtime state owns slots, positions, block tables, and prefix-cache bookkeeping.
- Device state owns the paged KV tensors and the prefill/decode kernels.

## Installation

```sh
uv sync --extra dev
uv sync --extra dev --extra rocm
uv sync --extra dev --extra parity
```

The `parity` extra installs the Hugging Face / PyTorch stack used by the parity
tests. It is usually best kept separate from the normal serving environment.

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

`/health` reports the configured attention backend, runtime limits, engine
counters, and any background engine-thread failure.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `JLLM_ATTENTION_IMPL` | `einsum` | Attention backend: `einsum` or `sdpa` |
| `JLLM_MODEL_PATH` | unset | Default model path for `jllm-serve` |
| `JAX_COMPILATION_CACHE_DIR` | unset | Persist XLA compilations across restarts |
| `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS` | `0` | Cache even short compiles |
| `JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES` | `0` | Persist small compiled artifacts |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` | Prevent eager full-device reservation |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `0.90` | Default memory ceiling for XLA allocation |

## Testing

The default test command is the fast gate. It excludes weight-backed engine
tests and HF parity tests.

```sh
UV_CACHE_DIR=.uv-cache uv run pytest
```

Engine tests run separately:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest -m engine tests/test_engine.py tests/test_threaded_engine.py --durations=10
```

Parity tests run separately:

```sh
UV_CACHE_DIR=.uv-cache uv run pytest -m parity tests/test_parity.py tests/test_long_rollout.py --durations=10
```

The split is intentional: the heavyweight tests load large JAX and Hugging Face
models, and keeping them out of the default loop makes local development and CI
much more predictable.

## Benchmarking

`scripts/bench_vllm.py` compares `jllm` and vLLM with the same prompt sets.
It supports both a running HTTP server and an in-process `jllm` engine.

```sh
uv run python scripts/bench_vllm.py --mode sequential
uv run python scripts/bench_vllm.py --mode concurrent --workload mixed --concurrency 1 2 4
uv run python scripts/bench_vllm.py --mode chat
uv run python scripts/bench_vllm.py --mode prefix_cache --prefix-len 256
```

Structured output:

```sh
uv run python scripts/bench_vllm.py \
  --mode concurrent \
  --workload mixed \
  --concurrency 1 4 16 \
  --json-out results/concurrent.json \
  --label mi300x-nightly
```

Concurrent runs report both:

- observed completion throughput: based on returned completion tokens
- budget-normalized throughput: based on `num_requests * max_new_tokens`

The JSON payload also records runner metadata plus lane provenance such as host,
port, model, backend, and optional GPU/lane-type labels.

## Remote ROCm Workflow

The remote workflow stays intentionally simple:

1. Sync the repo to the GPU host.
2. SSH into the host.
3. Launch isolated `jllm` and vLLM lanes.
4. Benchmark only healthy stable lanes.

`scripts/sync.sh` is parameterized with environment variables rather than
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

Each `jllm` lane gets its own persistent JAX compilation cache directory:

- `JAX_COMPILATION_CACHE_DIR=/tmp/jllm-jax-cache/gpu<gpu>-port<port>`
- `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0`
- `JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0`

From another shell on the remote host:

```sh
cd /root/jax-vllm
UV_CACHE_DIR=.uv-cache uv run python scripts/bench_vllm.py \
  --jllm-url http://127.0.0.1:8080 \
  --jllm-model-name Qwen2.5-32B-Instruct \
  --jllm-gpu 0 \
  --jllm-lane-type stable \
  --vllm-url http://127.0.0.1:8020 \
  --vllm-model-name qwen32b \
  --vllm-gpu 1 \
  --vllm-lane-type stable \
  --mode concurrent \
  --workload mixed \
  --concurrency 1 2 4 \
  --json-out results/concurrent.json
```
