# jax-vllm 

`jllm` is a compact JAX serving engine for decoder-only language models, built around vLLM-style continuous batching, paged KV caching, chunked prefill, and prefix reuse.

## Install

```sh
uv sync --extra dev
```

Optional extras:

```sh
uv sync --extra dev --extra rocm
uv sync --extra dev --extra torch
```

Use `rocm` on AMD GPU hosts. Use `torch` only in an environment where the Hugging Face/PyTorch CPU stack is acceptable.

Download weights into `weights/`:

```sh
uv run python scripts/download_model.py Qwen/Qwen2.5-32B-Instruct
```

## Usage

Start a server:

```sh
uv run jllm-serve --model-path weights/Qwen2.5-32B-Instruct --port 8080
```

Query it:

```sh
uv run python cli.py --url http://localhost:8080
curl http://localhost:8080/health
```

Basic endpoint examples:

```sh
curl http://localhost:8080/health

curl -X POST http://localhost:8080/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen2.5-32B-Instruct","prompt":"The capital of France is","max_tokens":16}'

curl -X POST http://localhost:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen2.5-32B-Instruct","prompt":"The capital of France is","max_tokens":16}'

curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen2.5-32B-Instruct","messages":[{"role":"user","content":"Say hello."}],"max_tokens":16}'
```


## Test

```bash
uv run pytest
```

Benchmark against vLLM, or run `jllm` alone:

```sh
uv run python scripts/bench_vllm.py --mode sequential
uv run python scripts/bench_vllm.py --mode concurrent --workload mixed --concurrency 1 2 4 --max-new-tokens 64
uv run python scripts/bench_vllm.py --mode prefix_cache --prefix-len 256
uv run python scripts/bench_vllm.py --mode concurrent --jllm-only --jllm-url http://127.0.0.1:8080
```

