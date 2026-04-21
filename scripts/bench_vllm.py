"""Unified jllm-vs-vLLM harness with optional JSON output."""
import argparse
from dataclasses import asdict
import json
import os
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

BALANCED_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time",
    "The 3 hardest things in computer science are",
    "Write a haiku about the ocean:",
    "In a galaxy far, far away,",
]

SHORT_PROMPTS = [
    "2 + 2 =",
    "Name one color.",
    "CPU stands for",
    "Finish: hello",
]

LONG_PROMPTS = [
    (
        "Summarize the following engineering note in two bullets:\n"
        + "We want stable throughput under mixed prompt lengths, predictable memory use, and repeatable benchmark output across runs. " * 12
        + "\nSummary:"
    ),
    (
        "Continue this field report in the same style:\n"
        + "At dusk the observatory logged increased thermal drift, a brief packet-loss burst on the telemetry link, and a slow recovery once the cooling loop stabilized. " * 10
        + "\nContinuation:"
    ),
    (
        "Read this memo and answer with the most important tradeoff:\n"
        + "The system should remain small and legible, but it also needs enough instrumentation to explain scheduler behavior, padding overhead, and compilation sensitivity under real traffic. " * 10
        + "\nTradeoff:"
    ),
]

WORKLOAD_PROMPTS = {
    "balanced": BALANCED_PROMPTS,
    "short": SHORT_PROMPTS,
    "long": LONG_PROMPTS,
    "mixed": SHORT_PROMPTS + BALANCED_PROMPTS + LONG_PROMPTS,
}

CHAT_CONVO = [
    "Hi! In one short sentence, what is the capital of France?",
    "And what language do they speak there?",
    "Name one famous landmark there.",
]


def _post_json(url: str, body: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    data = urllib.request.urlopen(req, timeout=timeout).read()
    return json.loads(data)


def _safe_health(url: str) -> dict:
    try:
        return json.loads(urllib.request.urlopen(f"{url}/health", timeout=5).read().decode())
    except Exception:
        return {}


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _tok_s(total_s: float, n_tokens: int) -> float:
    return n_tokens / total_s if total_s > 0 else 0.0


def _prompt_pool(workload: str) -> list[str]:
    return list(WORKLOAD_PROMPTS[workload])


def _request_prompts(workload: str, num_requests: int) -> list[str]:
    pool = _prompt_pool(workload)
    return (pool * ((num_requests // len(pool)) + 1))[:num_requests]


def _prompt_summary(prompts: list[str]) -> dict:
    if not prompts:
        return {"count": 0, "min_chars": 0, "median_chars": 0, "max_chars": 0, "unique_lengths": 0}
    lengths = sorted(len(prompt) for prompt in prompts)
    return {
        "count": len(prompts),
        "min_chars": lengths[0],
        "median_chars": statistics.median(lengths),
        "max_chars": lengths[-1],
        "unique_lengths": len(set(lengths)),
    }


def _write_json(path: str, payload: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")


def _runner_metadata() -> dict:
    return {
        "host": socket.gethostname(),
        "cwd": str(Path.cwd()),
    }


def _port_for_url(url: str) -> Optional[int]:
    try:
        return urlparse(url).port
    except Exception:
        return None


def _lane_metadata(url: str, model: str, health: dict, lane_type: str, gpu: Optional[int]) -> dict:
    return {
        "url": url,
        "model": model,
        "port": _port_for_url(url),
        "gpu": gpu,
        "lane_type": lane_type,
        "health": health,
    }


def _concurrent_row(
    label: str,
    concurrency: int,
    results: list[dict],
    wall_s: float,
    max_new_tokens: int,
) -> dict:
    observed_tokens = sum(response["n_tokens"] for response in results)
    budget_tokens = len(results) * max_new_tokens
    latencies = sorted(response["total_s"] for response in results)
    p50 = statistics.median(latencies) if latencies else 0.0
    p95 = latencies[int(0.95 * len(latencies)) - 1] if latencies else 0.0
    return {
        "system": label,
        "concurrency": concurrency,
        "num_requests": len(results),
        "wall_s": wall_s,
        "tokens": observed_tokens,
        "tok_per_s": _tok_s(wall_s, observed_tokens),
        "observed_tokens": observed_tokens,
        "observed_tok_per_s": _tok_s(wall_s, observed_tokens),
        "budget_tokens": budget_tokens,
        "budget_tok_per_s": _tok_s(wall_s, budget_tokens),
        "p50_latency_s": p50,
        "p95_latency_s": p95,
        "state": "warm",
    }


def _vllm_completion(url: str, model: str, prompt: str, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    response = _post_json(
        f"{url}/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "stream": False,
        },
    )
    total_s = time.perf_counter() - t0
    return {
        "text": response["choices"][0]["text"],
        "total_s": total_s,
        "n_tokens": response["usage"]["completion_tokens"],
        "tok_per_s": _tok_s(total_s, response["usage"]["completion_tokens"]),
        "state": "warm",
    }


def _vllm_chat(url: str, model: str, messages: list, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    response = _post_json(
        f"{url}/v1/chat/completions",
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        },
    )
    total_s = time.perf_counter() - t0
    return {
        "text": response["choices"][0]["message"]["content"],
        "total_s": total_s,
        "n_tokens": response["usage"]["completion_tokens"],
        "tok_per_s": _tok_s(total_s, response["usage"]["completion_tokens"]),
        "state": "warm",
    }


def _stats_delta(before: dict, after: dict) -> dict:
    delta: dict[str, int] = {}
    for key in sorted(set(before) | set(after)):
        delta[key] = int(after.get(key, 0)) - int(before.get(key, 0))
    return delta


def _safe_stats(jllm) -> dict:
    stats_fn = getattr(jllm, "stats", None)
    if stats_fn is None:
        return {}
    try:
        stats = stats_fn()
    except Exception:
        return {}
    return stats if isinstance(stats, dict) else {}


class HTTPJllm:
    def __init__(self, url: str, model_name: str):
        self.url = url
        self.model_name = model_name
        self.health = _safe_health(url)

    def info(self) -> dict:
        return {
            "backend": "http",
            "url": self.url,
            "model": self.model_name,
            "attention_impl": self.health.get("attention_impl", os.environ.get("JLLM_ATTENTION_IMPL", "einsum")),
            "server_health": self.health,
        }

    def stats(self) -> dict:
        return _safe_health(self.url).get("stats", {})

    def completion(self, prompt: str, max_tokens: int, state: str = "warm") -> dict:
        t0 = time.perf_counter()
        response = _post_json(
            f"{self.url}/v1/completions",
            {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": False,
            },
        )
        total_s = time.perf_counter() - t0
        n_tokens = response["usage"]["completion_tokens"]
        return {
            "text": response["choices"][0]["text"],
            "total_s": total_s,
            "n_tokens": n_tokens,
            "tok_per_s": _tok_s(total_s, n_tokens),
            "state": state,
        }

    def chat(self, messages: list, max_tokens: int, state: str = "warm") -> dict:
        t0 = time.perf_counter()
        response = _post_json(
            f"{self.url}/v1/chat/completions",
            {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": False,
            },
        )
        total_s = time.perf_counter() - t0
        n_tokens = response["usage"]["completion_tokens"]
        return {
            "text": response["choices"][0]["message"]["content"],
            "total_s": total_s,
            "n_tokens": n_tokens,
            "tok_per_s": _tok_s(total_s, n_tokens),
            "state": state,
        }

    def generate_stream(self, prompt: str, max_tokens: int, state: str = "warm") -> dict:
        body = json.dumps({"prompt": prompt, "max_tokens": max_tokens, "stream": True}).encode()
        req = urllib.request.Request(
            f"{self.url}/generate", data=body, headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        ttft = None
        n_tokens = 0
        text = ""
        with urllib.request.urlopen(req, timeout=600) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                if ttft is None:
                    ttft = time.perf_counter() - t0
                text += event["token"]
                n_tokens += 1
                if event.get("finished"):
                    break
        total_s = time.perf_counter() - t0
        return {
            "text": text,
            "total_s": total_s,
            "ttft_s": ttft,
            "n_tokens": n_tokens,
            "tok_per_s": _tok_s(total_s, n_tokens),
            "state": state,
        }


class InProcJllm:
    def __init__(self, args):
        import jax.numpy as jnp
        from transformers import AutoTokenizer

        from jllm.engine import engine as eng
        from jllm.engine.request import SamplingParams
        from jllm.model.weights import load_from_path

        self.args = args
        self.eng = eng
        self.SamplingParams = SamplingParams
        print("loading jllm engine...", flush=True)
        t0 = time.perf_counter()
        dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
        self.tok = AutoTokenizer.from_pretrained(args.model_path)
        self.eos_id = self.tok.convert_tokens_to_ids("<|im_end|>")
        if not isinstance(self.eos_id, int) or self.eos_id < 0:
            self.eos_id = self.tok.eos_token_id
        model = load_from_path(args.model_path, dtype=dtype)
        self.driver = eng.make_driver(
            model,
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
            max_prefill_len=args.max_prefill_len,
            dtype=dtype,
        )
        eng.start(self.driver)
        print(f"  jllm ready ({time.perf_counter()-t0:.1f}s)", flush=True)

    def info(self) -> dict:
        return {
            "backend": "inproc",
            "model_path": self.args.model_path,
            "dtype": self.args.dtype,
            "attention_impl": os.environ.get("JLLM_ATTENTION_IMPL", "einsum"),
            "max_num_seqs": self.args.max_num_seqs,
            "max_model_len": self.args.max_model_len,
            "max_prefill_len": self.args.max_prefill_len,
        }

    def stats(self) -> dict:
        return asdict(self.driver.stats)

    def stop(self):
        self.eng.stop(self.driver)

    def _run(self, ids: list[int], max_tokens: int, eos_id: Optional[int], state: str) -> dict:
        t0 = time.perf_counter()
        request_id = self.eng.add_request(
            self.driver,
            ids,
            self.SamplingParams(max_new_tokens=max_tokens, eos_id=eos_id),
        )
        ttft = None
        toks: list[int] = []
        for event in self.eng.stream(self.driver, request_id):
            if ttft is None:
                ttft = time.perf_counter() - t0
            toks.append(event.token)
        total_s = time.perf_counter() - t0
        return {
            "text": self.tok.decode(toks, skip_special_tokens=True),
            "total_s": total_s,
            "ttft_s": ttft,
            "n_tokens": len(toks),
            "tok_per_s": _tok_s(total_s, len(toks)),
            "state": state,
        }

    def completion(self, prompt: str, max_tokens: int, state: str = "warm") -> dict:
        return self._run(self.tok(prompt).input_ids, max_tokens, eos_id=None, state=state)

    def chat(self, messages: list, max_tokens: int, state: str = "warm") -> dict:
        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return self._run(self.tok(prompt).input_ids, max_tokens, eos_id=self.eos_id, state=state)

    def generate_stream(self, prompt: str, max_tokens: int, state: str = "warm") -> dict:
        return self._run(self.tok(prompt).input_ids, max_tokens, eos_id=None, state=state)


def _warmup(args, jllm, vllm_url: str, vllm_model: str) -> None:
    for _ in range(args.warmups):
        jllm.completion("hello", 4, state="warmup")
        if not args.jllm_only:
            _vllm_completion(vllm_url, vllm_model, "hello", 4)


def _warmup_concurrency_levels(
    args,
    jllm,
    vllm_url: str,
    vllm_model: str,
    workload_prompts: list[str],
) -> list[int]:
    # Warm each concurrency level with the full measurement max_tokens and real
    # workload prompts, so every (batch_bucket, context_bucket) shape the
    # measurement will hit is JIT-compiled before the timer starts. Using
    # max_tokens=4 / "hello" here leaves later shapes uncompiled and their
    # first-hit compile cost lands inside the timed run.
    warmed: list[int] = []
    max_new = args.max_new_tokens
    pool_prompts = list(dict.fromkeys(workload_prompts)) or ["hello"]
    for concurrency in sorted(set(args.concurrency)):
        if concurrency <= 1:
            continue
        prompts = (pool_prompts * ((concurrency // len(pool_prompts)) + 1))[:concurrency]

        def vllm_fn(prompt: str):
            return _vllm_completion(vllm_url, vllm_model, prompt, max_new)

        def jllm_fn(prompt: str):
            return jllm.completion(prompt, max_new, state=f"warmup-c{concurrency}")

        if not args.jllm_only:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                list(pool.map(vllm_fn, prompts))
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(jllm_fn, prompts))
        warmed.append(concurrency)
    return warmed


def _warmup_unique_prompts(
    jllm,
    vllm_url: str,
    vllm_model: str,
    prompts: list[str],
    max_tokens: int = 4,
    skip_vllm: bool = False,
) -> int:
    unique_prompts = list(dict.fromkeys(prompts))
    for prompt in unique_prompts:
        jllm.completion(prompt, max_tokens, state="warmup-prompts")
        if not skip_vllm:
            _vllm_completion(vllm_url, vllm_model, prompt, max_tokens)
    return len(unique_prompts)


def run_sequential(args, jllm, vllm_url: str, vllm_model: str) -> tuple[int, dict]:
    print("warming up...", flush=True)
    _warmup(args, jllm, vllm_url, vllm_model)
    print("  warmed", flush=True)

    prompts: list[dict] = []
    for prompt in _prompt_pool(args.workload):
        v_result = (
            None if args.jllm_only
            else _vllm_completion(vllm_url, vllm_model, prompt, args.max_new_tokens)
        )
        j_result = jllm.completion(prompt, args.max_new_tokens)
        if v_result is not None:
            match_chars = 0
            for left, right in zip(v_result["text"], j_result["text"]):
                if left == right:
                    match_chars += 1
                else:
                    break
            exact_match = j_result["text"] == v_result["text"]
            prompts.append({
                "prompt": prompt,
                "vllm": v_result,
                "jllm": j_result,
                "match_chars": match_chars,
                "exact_match": exact_match,
            })
            print(
                f"prompt={prompt!r:50s}  "
                f"vLLM {v_result['total_s']:.2f}s ({v_result['tok_per_s']:.1f} tok/s)  "
                f"jllm {j_result['total_s']:.2f}s ({j_result['tok_per_s']:.1f} tok/s)  "
                f"match={match_chars}/{min(len(v_result['text']), len(j_result['text']))}",
                flush=True,
            )
        else:
            prompts.append({"prompt": prompt, "jllm": j_result})
            print(
                f"prompt={prompt!r:50s}  "
                f"jllm {j_result['total_s']:.2f}s ({j_result['tok_per_s']:.1f} tok/s)",
                flush=True,
            )

    total_jllm_s = sum(row["jllm"]["total_s"] for row in prompts)
    total_jllm_toks = sum(row["jllm"]["n_tokens"] for row in prompts)
    print()
    if not args.jllm_only:
        total_vllm_s = sum(row["vllm"]["total_s"] for row in prompts)
        total_vllm_toks = sum(row["vllm"]["n_tokens"] for row in prompts)
        exact_matches = sum(1 for row in prompts if row.get("exact_match"))
        print(f"TOTAL vLLM: {total_vllm_s:.2f}s  {total_vllm_toks} tok  {_tok_s(total_vllm_s, total_vllm_toks):.1f} tok/s")
    else:
        total_vllm_s = 0.0
        total_vllm_toks = 0
        exact_matches = 0
    print(f"TOTAL jllm: {total_jllm_s:.2f}s  {total_jllm_toks} tok  {_tok_s(total_jllm_s, total_jllm_toks):.1f} tok/s")
    if not args.jllm_only:
        print(f"exact text matches: {exact_matches}/{len(prompts)}")

    result = {
        "mode": "sequential",
        "workload": args.workload,
        "warmups": args.warmups,
        "prompt_summary": _prompt_summary([row["prompt"] for row in prompts]),
        "prompts": prompts,
        "summary": {
            "vllm_total_s": total_vllm_s,
            "jllm_total_s": total_jllm_s,
            "vllm_total_tokens": total_vllm_toks,
            "jllm_total_tokens": total_jllm_toks,
            "vllm_tok_per_s": _tok_s(total_vllm_s, total_vllm_toks),
            "jllm_tok_per_s": _tok_s(total_jllm_s, total_jllm_toks),
            "exact_matches": exact_matches,
        },
    }
    if args.jllm_only:
        rc = 0
    else:
        rc = 0 if exact_matches >= len(prompts) * 0.5 else 2
    return rc, result


def run_concurrent(args, jllm, vllm_url: str, vllm_model: str) -> tuple[int, dict]:
    prompts = _request_prompts(args.workload, args.num_requests)
    print("warming up...", flush=True)
    _warmup(args, jllm, vllm_url, vllm_model)
    warmed_prompts = _warmup_unique_prompts(
        jllm, vllm_url, vllm_model, prompts, skip_vllm=args.jllm_only
    )
    warmed_concurrency = _warmup_concurrency_levels(args, jllm, vllm_url, vllm_model, prompts)
    print("  warmed", flush=True)
    if warmed_concurrency:
        print(f"  concurrency buckets warmed: {warmed_concurrency}", flush=True)
    print(f"  unique prompts warmed: {warmed_prompts}", flush=True)
    print(f"  workload={args.workload} prompt_chars={_prompt_summary(prompts)}", flush=True)

    print(
        f"{'system':<6} {'C':>4} {'N':>4} {'wall_s':>8} {'obs_tok':>8} "
        f"{'obs_tok/s':>10} {'budget_tok/s':>12} {'p50_lat':>8} {'p95_lat':>8}"
    )
    rows: list[dict] = []
    for concurrency in args.concurrency:
        def vllm_fn(prompt: str):
            return _vllm_completion(vllm_url, vllm_model, prompt, args.max_new_tokens)

        def jllm_fn(prompt: str):
            return jllm.completion(prompt, args.max_new_tokens)

        lanes = [("jllm", jllm_fn)] if args.jllm_only else [("vllm", vllm_fn), ("jllm", jllm_fn)]
        for label, fn in lanes:
            t0 = time.perf_counter()
            results = []
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for response in pool.map(fn, prompts):
                    results.append(response)
            wall_s = time.perf_counter() - t0
            row = _concurrent_row(label, concurrency, results, wall_s, args.max_new_tokens)
            rows.append(row)
            print(
                f"{label:<6} {concurrency:>4} {len(results):>4} {wall_s:>8.2f} "
                f"{row['observed_tokens']:>8} {row['observed_tok_per_s']:>10.1f} "
                f"{row['budget_tok_per_s']:>12.1f} {row['p50_latency_s']:>8.2f} "
                f"{row['p95_latency_s']:>8.2f}",
                flush=True,
            )
    return 0, {
        "mode": "concurrent",
        "workload": args.workload,
        "warmups": args.warmups,
        "concurrency_warmups": warmed_concurrency,
        "prompt_warmups": warmed_prompts,
        "prompt_summary": _prompt_summary(prompts),
        "rows": rows,
    }


def run_chat(args, jllm, vllm_url: str, vllm_model: str) -> tuple[int, dict]:
    def run_side(chat_fn) -> list[dict]:
        messages = []
        outputs = []
        for user_text in CHAT_CONVO:
            messages.append({"role": "user", "content": user_text})
            response = chat_fn(messages, args.max_new_tokens)
            messages.append({"role": "assistant", "content": response["text"]})
            outputs.append(response)
        return outputs

    print("=== jllm ===", flush=True)
    jllm_turns = run_side(jllm.chat)
    for idx, response in enumerate(jllm_turns):
        print(f"[t{idx}] ({response['total_s']:.1f}s)\n{response['text']}\n")

    if args.jllm_only:
        turns = [{"turn": idx, "jllm": t} for idx, t in enumerate(jllm_turns)]
        return 0, {
            "mode": "chat",
            "turns": turns,
            "summary": {"exact_matches": 0, "num_turns": len(turns)},
        }

    print("=== vllm ===", flush=True)
    vllm_turns = run_side(lambda messages, n: _vllm_chat(vllm_url, vllm_model, messages, n))
    for idx, response in enumerate(vllm_turns):
        print(f"[t{idx}] ({response['total_s']:.1f}s)\n{response['text']}\n")

    exact_matches = 0
    turns = []
    print("=== diff ===")
    for idx, (j_turn, v_turn) in enumerate(zip(jllm_turns, vllm_turns)):
        same = j_turn["text"].strip() == v_turn["text"].strip()
        if same:
            exact_matches += 1
        turns.append({"turn": idx, "jllm": j_turn, "vllm": v_turn, "exact_match": same})
        print(f"[t{idx}] {'MATCH' if same else 'DIFFER'}")
        if not same:
            print(f"  jllm: {j_turn['text']!r}")
            print(f"  vllm: {v_turn['text']!r}")

    return (0 if exact_matches == len(turns) else 2), {
        "mode": "chat",
        "turns": turns,
        "summary": {"exact_matches": exact_matches, "num_turns": len(turns)},
    }


def run_prefix_cache(args, jllm, vllm_url: str, vllm_model: str) -> tuple[int, dict]:
    prefix = (
        "In machine learning research, "
        "the scaling laws paper by Kaplan et al. establishes that "
        "loss decreases as a power law in compute, parameters, and data. " * 20
    )
    prefix = prefix[: args.prefix_len * 4]
    suffixes = [f" Q{i}: What's the main point?" for i in range(args.num_requests)]

    print(f"prefix_cache: prefix ~{args.prefix_len} tokens, {args.num_requests} requests", flush=True)

    jllm_rows = []
    print("=== jllm (warms its cache on req0) ===")
    for idx, suffix in enumerate(suffixes):
        state = "cold" if idx == 0 else "warm"
        result = jllm.generate_stream(prefix + suffix, args.max_new_tokens, state=state)
        jllm_rows.append(result)
        print(
            f"  jllm req{idx:02d}  total={result['total_s']:.2f}s  "
            f"ttft={result.get('ttft_s', 0):.2f}s  tok/s={result['tok_per_s']:.1f}",
            flush=True,
        )

    vllm_rows = []
    if not args.jllm_only:
        print("\n=== vllm (configure --enable-prefix-caching for a fair comparison) ===")
        for idx, suffix in enumerate(suffixes):
            result = _vllm_completion(vllm_url, vllm_model, prefix + suffix, args.max_new_tokens)
            result["state"] = "cold" if idx == 0 else "warm"
            vllm_rows.append(result)
            print(f"  vllm req{idx:02d}  total={result['total_s']:.2f}s  tok/s={result['tok_per_s']:.1f}", flush=True)

    warm_ttfts = [row["ttft_s"] for row in jllm_rows[1:] if row.get("ttft_s") is not None]
    summary = {
        "jllm_cold_ttft_s": jllm_rows[0].get("ttft_s"),
        "jllm_warm_ttft_median_s": statistics.median(warm_ttfts) if warm_ttfts else None,
    }
    if summary["jllm_cold_ttft_s"] is not None:
        print(f"\njllm cold TTFT: {summary['jllm_cold_ttft_s']:.3f}s")
    if summary["jllm_warm_ttft_median_s"] is not None:
        print(f"jllm warm TTFT median: {summary['jllm_warm_ttft_median_s']:.3f}s")

    return 0, {"mode": "prefix_cache", "jllm": jllm_rows, "vllm": vllm_rows, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sequential", "concurrent", "chat", "prefix_cache"], default="sequential")
    parser.add_argument("--workload", choices=sorted(WORKLOAD_PROMPTS), default="balanced",
                        help="Prompt workload profile used by sequential/concurrent modes.")
    parser.add_argument("--jllm-url", default=None,
                        help="Hit a running jllm server at this URL. If omitted, spin up in-process.")
    parser.add_argument("--jllm-model-name", default="qwen32b",
                        help="Model name to put in OpenAI-shaped requests (HTTP mode only)")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8020")
    parser.add_argument("--vllm-model-name", default="qwen32b")
    parser.add_argument(
        "--jllm-only",
        action="store_true",
        help="Skip all vLLM calls (warmup, measurement, metadata). Use when vLLM is not running.",
    )
    parser.add_argument("--model-path", default="weights/Qwen2.5-32B-Instruct")
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-prefill-len", type=int, default=512)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--prefix-len", type=int, default=256,
                        help="prefix_cache mode: approximate tokens of shared prefix")
    parser.add_argument("--json-out", default=None,
                        help="Optional path to write structured benchmark results as JSON.")
    parser.add_argument("--label", default=None, help="Optional label stored in the JSON metadata.")
    parser.add_argument("--warmups", type=int, default=1, help="Number of warmup requests to run before timed modes.")
    parser.add_argument("--jllm-gpu", type=int, default=None, help="Optional GPU id for the jllm lane metadata.")
    parser.add_argument("--vllm-gpu", type=int, default=None, help="Optional GPU id for the vLLM lane metadata.")
    parser.add_argument("--jllm-lane-type", choices=["stable", "experimental"], default="stable")
    parser.add_argument("--vllm-lane-type", choices=["stable", "experimental"], default="stable")
    args = parser.parse_args()

    if args.jllm_url is not None:
        jllm = HTTPJllm(args.jllm_url, args.jllm_model_name)
        stop_fn = lambda: None
    else:
        jllm = InProcJllm(args)
        stop_fn = jllm.stop

    stats_before = _safe_stats(jllm)
    try:
        if args.mode == "sequential":
            rc, result = run_sequential(args, jllm, args.vllm_url, args.vllm_model_name)
        elif args.mode == "concurrent":
            rc, result = run_concurrent(args, jllm, args.vllm_url, args.vllm_model_name)
        elif args.mode == "chat":
            rc, result = run_chat(args, jllm, args.vllm_url, args.vllm_model_name)
        elif args.mode == "prefix_cache":
            rc, result = run_prefix_cache(args, jllm, args.vllm_url, args.vllm_model_name)
        else:
            raise AssertionError(f"unknown mode {args.mode}")
        stats_after = _safe_stats(jllm)
    finally:
        stop_fn()

    vllm_health = {} if args.jllm_only else _safe_health(args.vllm_url)
    jllm_info = jllm.info()
    if args.jllm_url is not None:
        jllm_info.update(
            _lane_metadata(
                args.jllm_url,
                args.jllm_model_name,
                jllm_info.get("server_health", {}),
                args.jllm_lane_type,
                args.jllm_gpu,
            )
        )
    else:
        jllm_info.update({"gpu": args.jllm_gpu, "lane_type": "inproc"})

    payload = {
        "metadata": {
            "label": args.label,
            "git_sha": _git_sha(),
            "mode": args.mode,
            "workload": args.workload,
            "warmups": args.warmups,
            "started_at_unix_s": time.time(),
            "runner": _runner_metadata(),
            "token_metrics": {
                "observed": "returned completion_tokens",
                "budgeted": "num_requests * max_new_tokens",
            },
            "jllm": jllm_info,
            "jllm_stats_before": stats_before,
            "jllm_stats_after": stats_after,
            "jllm_stats_delta": _stats_delta(stats_before, stats_after),
            "vllm": _lane_metadata(
                args.vllm_url,
                args.vllm_model_name,
                vllm_health,
                args.vllm_lane_type,
                args.vllm_gpu,
            ),
        },
        "result": result,
    }
    if args.json_out:
        _write_json(args.json_out, payload)
        print(f"wrote JSON results to {args.json_out}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
