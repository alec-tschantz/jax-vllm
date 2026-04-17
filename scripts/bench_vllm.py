"""Unified jllm-vs-vLLM harness. One script, four modes.

Transports:
- If --jllm-url is set, hit jllm over HTTP (server must be running).
- Otherwise, spin up a jllm engine in-process (bench owns the model + GPU0).

Modes:
- sequential     (default)   single-prompt /v1/completions parity, greedy
- concurrent                  ThreadPool fan-out, aggregate throughput + p50/p95
- chat                        multi-turn /v1/chat/completions parity
- prefix_cache                N requests sharing a long system prompt; measure
                              TTFT/throughput cold vs warm

Only `sequential` and `chat` exercise text-match; the others report timings.

Typical use (from gpu-droplet, jllm on :8080, vLLM on :8020):
  uv run python scripts/bench_vllm.py --mode sequential
  uv run python scripts/bench_vllm.py --mode concurrent --concurrency 1 4 16
  uv run python scripts/bench_vllm.py --mode chat
  uv run python scripts/bench_vllm.py --mode prefix_cache --num-requests 8 --prefix-len 256
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
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


# ---------- HTTP clients ----------


def _post_json(url: str, body: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    data = urllib.request.urlopen(req, timeout=timeout).read()
    return json.loads(data)


def _vllm_completion(url: str, model: str, prompt: str, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    r = _post_json(f"{url}/v1/completions", {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "stream": False,
    })
    dt = time.perf_counter() - t0
    return {
        "text": r["choices"][0]["text"],
        "total_s": dt,
        "n_tokens": r["usage"]["completion_tokens"],
    }


def _vllm_chat(url: str, model: str, messages: list, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    r = _post_json(f"{url}/v1/chat/completions", {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    })
    dt = time.perf_counter() - t0
    return {
        "text": r["choices"][0]["message"]["content"],
        "total_s": dt,
        "n_tokens": r["usage"]["completion_tokens"],
    }


# ---------- jllm backends: HTTP or in-process ----------


class HTTPJllm:
    """Talk to a running jllm server."""
    def __init__(self, url: str, model_name: str):
        self.url = url
        self.model_name = model_name

    def completion(self, prompt: str, max_tokens: int) -> dict:
        t0 = time.perf_counter()
        r = _post_json(f"{self.url}/v1/completions", {
            "model": self.model_name, "prompt": prompt,
            "max_tokens": max_tokens, "temperature": 0.0, "stream": False,
        })
        dt = time.perf_counter() - t0
        return {
            "text": r["choices"][0]["text"],
            "total_s": dt,
            "n_tokens": r["usage"]["completion_tokens"],
        }

    def chat(self, messages: list, max_tokens: int) -> dict:
        t0 = time.perf_counter()
        r = _post_json(f"{self.url}/v1/chat/completions", {
            "model": self.model_name, "messages": messages,
            "max_tokens": max_tokens, "temperature": 0.0, "stream": False,
        })
        dt = time.perf_counter() - t0
        return {
            "text": r["choices"][0]["message"]["content"],
            "total_s": dt,
            "n_tokens": r["usage"]["completion_tokens"],
        }

    def generate_stream(self, prompt: str, max_tokens: int) -> dict:
        """Stream /generate to capture TTFT."""
        body = json.dumps({"prompt": prompt, "max_tokens": max_tokens, "stream": True}).encode()
        req = urllib.request.Request(
            f"{self.url}/generate", data=body, headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        ttft = None
        n = 0
        text = ""
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if line.startswith("data: "):
                    ev = json.loads(line[6:])
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text += ev["token"]
                    n += 1
                    if ev.get("finished"):
                        break
        return {"text": text, "total_s": time.perf_counter() - t0, "ttft_s": ttft, "n_tokens": n}


class InProcJllm:
    """Spin up a jllm driver + tokenizer in-process."""
    def __init__(self, args):
        import jax.numpy as jnp
        from transformers import AutoTokenizer
        from jllm.engine import engine as eng
        from jllm.engine.request import SamplingParams
        from jllm.model.weights import load_qwen2

        self.eng = eng
        self.SamplingParams = SamplingParams
        print("loading jllm engine...", flush=True)
        t0 = time.perf_counter()
        dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
        self.tok = AutoTokenizer.from_pretrained(args.model_path)
        self.eos_id = self.tok.convert_tokens_to_ids("<|im_end|>")
        if not isinstance(self.eos_id, int) or self.eos_id < 0:
            self.eos_id = self.tok.eos_token_id
        model = load_qwen2(args.model_path, dtype=dtype)
        self.driver = eng.make_driver(
            model,
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
            max_prefill_len=args.max_prefill_len,
            dtype=dtype,
        )
        eng.start(self.driver)
        print(f"  jllm ready ({time.perf_counter()-t0:.1f}s)", flush=True)

    def stop(self):
        self.eng.stop(self.driver)

    def _run(self, ids: list, max_tokens: int, eos_id: Optional[int] = None) -> dict:
        t0 = time.perf_counter()
        rid = self.eng.add_request(
            self.driver, ids,
            self.SamplingParams(max_new_tokens=max_tokens, eos_id=eos_id),
        )
        ttft = None
        toks: list = []
        for ev in self.eng.stream(self.driver, rid):
            if ttft is None:
                ttft = time.perf_counter() - t0
            toks.append(ev.token)
        return {
            "text": self.tok.decode(toks, skip_special_tokens=True),
            "total_s": time.perf_counter() - t0,
            "ttft_s": ttft,
            "n_tokens": len(toks),
        }

    def completion(self, prompt: str, max_tokens: int) -> dict:
        return self._run(self.tok(prompt).input_ids, max_tokens, eos_id=None)

    def chat(self, messages: list, max_tokens: int) -> dict:
        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return self._run(self.tok(prompt).input_ids, max_tokens, eos_id=self.eos_id)

    def generate_stream(self, prompt: str, max_tokens: int) -> dict:
        return self._run(self.tok(prompt).input_ids, max_tokens, eos_id=None)


# ---------- modes ----------


def run_sequential(args, jllm, vllm_url: str, vllm_model: str) -> int:
    """Single-turn /v1/completions parity. Greedy. Report tok/s and text match."""
    print("warming up...", flush=True)
    jllm.completion("hello", 4)
    _vllm_completion(vllm_url, vllm_model, "hello", 4)
    print("  warmed", flush=True)

    results = []
    for prompt in DEFAULT_PROMPTS:
        v = _vllm_completion(vllm_url, vllm_model, prompt, args.max_new_tokens)
        j = jllm.completion(prompt, args.max_new_tokens)
        match = 0
        for a, b in zip(v["text"], j["text"]):
            if a == b:
                match += 1
            else:
                break
        results.append({"prompt": prompt, "vllm": v, "jllm": j, "match_chars": match})
        print(
            f"prompt={prompt!r:50s}  "
            f"vLLM {v['total_s']:.2f}s ({(v['n_tokens'] or 1)/v['total_s']:.1f} tok/s)  "
            f"jllm {j['total_s']:.2f}s ({(j['n_tokens'] or 1)/j['total_s']:.1f} tok/s)  "
            f"match={match}/{min(len(v['text']), len(j['text']))}",
            flush=True,
        )

    vtot = sum(r["vllm"]["total_s"] for r in results)
    jtot = sum(r["jllm"]["total_s"] for r in results)
    vtoks = sum(r["vllm"]["n_tokens"] or 0 for r in results)
    jtoks = sum(r["jllm"]["n_tokens"] or 0 for r in results)
    print()
    print(f"TOTAL vLLM: {vtot:.2f}s  {vtoks} tok  {vtoks/vtot:.1f} tok/s")
    print(f"TOTAL jllm: {jtot:.2f}s  {jtoks} tok  {jtoks/jtot:.1f} tok/s")
    full = sum(1 for r in results if r["jllm"]["text"] == r["vllm"]["text"])
    print(f"exact text matches: {full}/{len(results)}")
    return 0 if full >= len(results) * 0.5 else 2


def run_concurrent(args, jllm, vllm_url: str, vllm_model: str) -> int:
    prompts = (DEFAULT_PROMPTS * ((args.num_requests // len(DEFAULT_PROMPTS)) + 1))[: args.num_requests]
    print("warming up at C=1...", flush=True)
    jllm.completion("hello", 4)
    _vllm_completion(vllm_url, vllm_model, "hello", 4)

    print(f"{'system':<6} {'C':>4} {'N':>4} {'wall_s':>8} {'tokens':>8} {'tok/s':>9} {'p50_lat':>8} {'p95_lat':>8}")
    for c in args.concurrency:
        def vllm_fn(p, c=c):
            return _vllm_completion(vllm_url, vllm_model, p, args.max_new_tokens)
        def jllm_fn(p):
            return jllm.completion(p, args.max_new_tokens)
        for label, fn in [("vllm", vllm_fn), ("jllm", jllm_fn)]:
            t0 = time.perf_counter()
            out = []
            with ThreadPoolExecutor(max_workers=c) as pool:
                for r in pool.map(fn, prompts):
                    out.append(r)
            wall = time.perf_counter() - t0
            tokens = sum(r["n_tokens"] or 0 for r in out)
            lats = sorted(r["total_s"] for r in out)
            p50 = statistics.median(lats)
            p95 = lats[int(0.95 * len(lats)) - 1] if lats else 0.0
            print(
                f"{label:<6} {c:>4} {len(out):>4} {wall:>8.2f} {tokens:>8} "
                f"{tokens/wall:>9.1f} {p50:>8.2f} {p95:>8.2f}",
                flush=True,
            )
    return 0


def run_chat(args, jllm, vllm_url: str, vllm_model: str) -> int:
    def run_side(side: str, chat_fn) -> list:
        msgs = []
        outs = []
        for user in CHAT_CONVO:
            msgs.append({"role": "user", "content": user})
            r = chat_fn(msgs, args.max_new_tokens)
            msgs.append({"role": "assistant", "content": r["text"]})
            outs.append(r)
        return outs

    print("=== jllm ===", flush=True)
    jllm_out = run_side("jllm", jllm.chat)
    for i, r in enumerate(jllm_out):
        print(f"[t{i}] ({r['total_s']:.1f}s)\n{r['text']}\n")

    print("=== vllm ===", flush=True)
    vllm_out = run_side("vllm", lambda m, n: _vllm_chat(vllm_url, vllm_model, m, n))
    for i, r in enumerate(vllm_out):
        print(f"[t{i}] ({r['total_s']:.1f}s)\n{r['text']}\n")

    print("=== diff ===")
    exact = 0
    for i, (j, v) in enumerate(zip(jllm_out, vllm_out)):
        same = j["text"].strip() == v["text"].strip()
        if same:
            exact += 1
        print(f"[t{i}] {'MATCH' if same else 'DIFFER'}")
        if not same:
            print(f"  jllm: {j['text']!r}")
            print(f"  vllm: {v['text']!r}")
    return 0 if exact == len(jllm_out) else 2


def run_prefix_cache(args, jllm, vllm_url: str, vllm_model: str) -> int:
    """N requests share a long system prompt + unique user turn. Report TTFT
    and tok/s cold vs warm. Also fire the same workload at vLLM for comparison
    (vLLM needs --enable-prefix-caching to be running for this to be fair)."""
    # Build a deterministic long prefix (repeated text — the tokenizer will
    # produce the same tokens each time).
    prefix = ("In machine learning research, "
              "the scaling laws paper by Kaplan et al. establishes that "
              "loss decreases as a power law in compute, parameters, and data. " * 20)
    # Trim so the tokenized prefix length ~= args.prefix_len.
    # Cheap heuristic: 4 chars per token. Take args.prefix_len * 4 chars.
    prefix = prefix[: args.prefix_len * 4]
    suffixes = [f" Q{i}: What's the main point?" for i in range(args.num_requests)]

    print(f"prefix_cache: prefix ~{args.prefix_len} tokens, {args.num_requests} requests", flush=True)

    def bench(label: str, fn) -> list:
        rows = []
        for i, suf in enumerate(suffixes):
            prompt = prefix + suf
            r = fn(prompt, args.max_new_tokens)
            rows.append(r)
            print(
                f"  {label} req{i:02d}  total={r['total_s']:.2f}s  "
                f"ttft={r.get('ttft_s', 0):.2f}s  tok/s={(r['n_tokens'] or 1)/r['total_s']:.1f}",
                flush=True,
            )
        return rows

    print("=== jllm (warms its cache on req0) ===")
    jllm_rows = bench("jllm", jllm.generate_stream)
    cold_ttft = jllm_rows[0].get("ttft_s", 0)
    warm_ttfts = [r.get("ttft_s", 0) for r in jllm_rows[1:]]
    print(f"\njllm cold TTFT: {cold_ttft:.3f}s")
    if warm_ttfts:
        print(f"jllm warm TTFT median: {statistics.median(warm_ttfts):.3f}s  "
              f"(min {min(warm_ttfts):.3f}s, max {max(warm_ttfts):.3f}s)")

    # vLLM parallel comparison — useful if vLLM has --enable-prefix-caching on.
    print("\n=== vllm (configure --enable-prefix-caching for a fair comparison) ===")
    vllm_rows = []
    for i, suf in enumerate(suffixes):
        prompt = prefix + suf
        r = _vllm_completion(vllm_url, vllm_model, prompt, args.max_new_tokens)
        vllm_rows.append(r)
        print(f"  vllm req{i:02d}  total={r['total_s']:.2f}s  tok/s={(r['n_tokens'] or 1)/r['total_s']:.1f}", flush=True)

    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["sequential", "concurrent", "chat", "prefix_cache"], default="sequential")
    p.add_argument("--jllm-url", default=None,
                   help="Hit a running jllm server at this URL. If omitted, spin up in-process.")
    p.add_argument("--jllm-model-name", default="qwen32b",
                   help="Model name to put in OpenAI-shaped requests (HTTP mode only)")
    p.add_argument("--vllm-url", default="http://127.0.0.1:8020")
    p.add_argument("--vllm-model-name", default="qwen32b")
    # in-process jllm options (ignored when --jllm-url is set)
    p.add_argument("--model-path", default="weights/Qwen2.5-32B-Instruct")
    p.add_argument("--max-num-seqs", type=int, default=4)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-prefill-len", type=int, default=512)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    # mode-specific options
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--num-requests", type=int, default=32)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--prefix-len", type=int, default=256,
                   help="prefix_cache mode: approximate tokens of shared prefix")
    args = p.parse_args()

    jllm: "HTTPJllm | InProcJllm"
    if args.jllm_url is not None:
        jllm = HTTPJllm(args.jllm_url, args.jllm_model_name)
        stop_fn = lambda: None
    else:
        jllm = InProcJllm(args)
        stop_fn = jllm.stop

    try:
        if args.mode == "sequential":
            rc = run_sequential(args, jllm, args.vllm_url, args.vllm_model_name)
        elif args.mode == "concurrent":
            rc = run_concurrent(args, jllm, args.vllm_url, args.vllm_model_name)
        elif args.mode == "chat":
            rc = run_chat(args, jllm, args.vllm_url, args.vllm_model_name)
        elif args.mode == "prefix_cache":
            rc = run_prefix_cache(args, jllm, args.vllm_url, args.vllm_model_name)
        else:
            raise AssertionError(f"unknown mode {args.mode}")
    finally:
        stop_fn()
    return rc


if __name__ == "__main__":
    sys.exit(main())
