"""Unified jllm-vs-vLLM harness with optional JSON output."""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

DEFAULT_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time",
    "The 3 hardest things in computer science are",
    "Write a haiku about the ocean:",
    "In a galaxy far, far away,",
]

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


def _write_json(path: str, payload: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")


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
        _vllm_completion(vllm_url, vllm_model, "hello", 4)


def run_sequential(args, jllm, vllm_url: str, vllm_model: str) -> tuple[int, dict]:
    print("warming up...", flush=True)
    _warmup(args, jllm, vllm_url, vllm_model)
    print("  warmed", flush=True)

    prompts: list[dict] = []
    for prompt in DEFAULT_PROMPTS:
        v_result = _vllm_completion(vllm_url, vllm_model, prompt, args.max_new_tokens)
        j_result = jllm.completion(prompt, args.max_new_tokens)
        match_chars = 0
        for left, right in zip(v_result["text"], j_result["text"]):
            if left == right:
                match_chars += 1
            else:
                break
        prompts.append(
            {
                "prompt": prompt,
                "vllm": v_result,
                "jllm": j_result,
                "match_chars": match_chars,
                "exact_match": j_result["text"] == v_result["text"],
            }
        )
        print(
            f"prompt={prompt!r:50s}  "
            f"vLLM {v_result['total_s']:.2f}s ({v_result['tok_per_s']:.1f} tok/s)  "
            f"jllm {j_result['total_s']:.2f}s ({j_result['tok_per_s']:.1f} tok/s)  "
            f"match={match_chars}/{min(len(v_result['text']), len(j_result['text']))}",
            flush=True,
        )

    total_vllm_s = sum(row["vllm"]["total_s"] for row in prompts)
    total_jllm_s = sum(row["jllm"]["total_s"] for row in prompts)
    total_vllm_toks = sum(row["vllm"]["n_tokens"] for row in prompts)
    total_jllm_toks = sum(row["jllm"]["n_tokens"] for row in prompts)
    exact_matches = sum(1 for row in prompts if row["exact_match"])
    print()
    print(f"TOTAL vLLM: {total_vllm_s:.2f}s  {total_vllm_toks} tok  {_tok_s(total_vllm_s, total_vllm_toks):.1f} tok/s")
    print(f"TOTAL jllm: {total_jllm_s:.2f}s  {total_jllm_toks} tok  {_tok_s(total_jllm_s, total_jllm_toks):.1f} tok/s")
    print(f"exact text matches: {exact_matches}/{len(prompts)}")

    result = {
        "mode": "sequential",
        "warmups": args.warmups,
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
    return (0 if exact_matches >= len(prompts) * 0.5 else 2), result


def run_concurrent(args, jllm, vllm_url: str, vllm_model: str) -> tuple[int, dict]:
    prompts = (DEFAULT_PROMPTS * ((args.num_requests // len(DEFAULT_PROMPTS)) + 1))[: args.num_requests]
    print("warming up...", flush=True)
    _warmup(args, jllm, vllm_url, vllm_model)
    print("  warmed", flush=True)

    print(f"{'system':<6} {'C':>4} {'N':>4} {'wall_s':>8} {'tokens':>8} {'tok/s':>9} {'p50_lat':>8} {'p95_lat':>8}")
    rows: list[dict] = []
    for concurrency in args.concurrency:
        def vllm_fn(prompt: str):
            return _vllm_completion(vllm_url, vllm_model, prompt, args.max_new_tokens)

        def jllm_fn(prompt: str):
            return jllm.completion(prompt, args.max_new_tokens)

        for label, fn in [("vllm", vllm_fn), ("jllm", jllm_fn)]:
            t0 = time.perf_counter()
            results = []
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for response in pool.map(fn, prompts):
                    results.append(response)
            wall_s = time.perf_counter() - t0
            tokens = sum(response["n_tokens"] for response in results)
            latencies = sorted(response["total_s"] for response in results)
            p50 = statistics.median(latencies)
            p95 = latencies[int(0.95 * len(latencies)) - 1] if latencies else 0.0
            row = {
                "system": label,
                "concurrency": concurrency,
                "num_requests": len(results),
                "wall_s": wall_s,
                "tokens": tokens,
                "tok_per_s": _tok_s(wall_s, tokens),
                "p50_latency_s": p50,
                "p95_latency_s": p95,
                "state": "warm",
            }
            rows.append(row)
            print(
                f"{label:<6} {concurrency:>4} {len(results):>4} {wall_s:>8.2f} {tokens:>8} "
                f"{row['tok_per_s']:>9.1f} {p50:>8.2f} {p95:>8.2f}",
                flush=True,
            )
    return 0, {"mode": "concurrent", "warmups": args.warmups, "rows": rows}


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

    print("\n=== vllm (configure --enable-prefix-caching for a fair comparison) ===")
    vllm_rows = []
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
    parser.add_argument("--jllm-url", default=None,
                        help="Hit a running jllm server at this URL. If omitted, spin up in-process.")
    parser.add_argument("--jllm-model-name", default="qwen32b",
                        help="Model name to put in OpenAI-shaped requests (HTTP mode only)")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8020")
    parser.add_argument("--vllm-model-name", default="qwen32b")
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
    args = parser.parse_args()

    if args.jllm_url is not None:
        jllm = HTTPJllm(args.jllm_url, args.jllm_model_name)
        stop_fn = lambda: None
    else:
        jllm = InProcJllm(args)
        stop_fn = jllm.stop

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
    finally:
        stop_fn()

    payload = {
        "metadata": {
            "label": args.label,
            "git_sha": _git_sha(),
            "mode": args.mode,
            "warmups": args.warmups,
            "started_at_unix_s": time.time(),
            "jllm": jllm.info(),
            "vllm": {
                "url": args.vllm_url,
                "model": args.vllm_model_name,
                "health": _safe_health(args.vllm_url),
            },
        },
        "result": result,
    }
    if args.json_out:
        _write_json(args.json_out, payload)
        print(f"wrote JSON results to {args.json_out}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
