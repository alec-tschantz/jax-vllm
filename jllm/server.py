"""FastAPI server exposing the jllm engine over HTTP.

Run locally, or on a remote GPU host with a port-forwarded tunnel
(e.g. `ssh -L 8080:localhost:8080 <host>`).

Endpoints:
- POST /generate            raw-prompt completion (streams SSE)
- POST /v1/completions      OpenAI-compatible completion (raw prompt, no template)
- POST /v1/chat/completions OpenAI-compatible chat (applies tokenizer chat template)
- GET  /health
"""
# Apply JAX env vars (JAX_COMPILATION_CACHE_DIR, XLA_PYTHON_CLIENT_*) before
# any jax import anywhere in the process. See jllm.config for the knob list.
from jllm.config import JllmConfig, apply_jax_env
_INITIAL_CFG = JllmConfig.from_env()
apply_jax_env(_INITIAL_CFG)

import argparse
import json
import time
import uuid
from dataclasses import asdict

import jax.numpy as jnp
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

from jllm.engine import engine
from jllm.engine.request import SamplingParams
from jllm.model.weights import load_from_path

app = FastAPI(title="jllm", version="0.0.1")
STATE: dict = {}


class GenRequest(BaseModel):
    prompt: str
    max_tokens: int = 64
    stream: bool = True


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = 128
    temperature: float = 0.0
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int = 64
    temperature: float = 0.0
    stream: bool = False


@app.get("/health")
def health():
    d = STATE["driver"]
    return {
        "status": "error" if d.thread_error is not None else "ok",
        "model": STATE["model_path"],
        "attention_impl": _INITIAL_CFG.attention_impl,
        "active_slots": len(d.running),
        "waiting": d.waiting.qsize(),
        "max_num_seqs": d.max_num_seqs,
        "max_model_len": d.max_model_len,
        "max_prefill_len": d.max_prefill_len,
        "thread_error": d.thread_error,
        "stats": asdict(d.stats),
    }


def _stream_generate(ids: list[int], max_tokens: int, eos_id: int | None = None):
    tok = STATE["tok"]
    driver = STATE["driver"]
    rid = engine.add_request(
        driver, ids, SamplingParams(max_new_tokens=max_tokens, eos_id=eos_id)
    )
    t0 = time.perf_counter()
    ttft = None
    n = 0
    for ev in engine.stream(driver, rid):
        n += 1
        if ttft is None:
            ttft = time.perf_counter() - t0
        piece = tok.decode([ev.token], skip_special_tokens=True)
        yield {
            "token": piece,
            "token_id": ev.token,
            "n": n,
            "elapsed_s": time.perf_counter() - t0,
            "ttft_s": ttft,
            "finished": ev.finished,
        }


def _collect(ids: list[int], max_tokens: int, eos_id: int | None = None) -> tuple[list[int], float]:
    driver = STATE["driver"]
    rid = engine.add_request(
        driver, ids, SamplingParams(max_new_tokens=max_tokens, eos_id=eos_id)
    )
    tokens: list[int] = []
    t0 = time.perf_counter()
    for ev in engine.stream(driver, rid):
        tokens.append(ev.token)
    return tokens, time.perf_counter() - t0


@app.post("/generate")
def generate(req: GenRequest):
    tok = STATE["tok"]
    ids = tok(req.prompt).input_ids

    if req.stream:
        def gen():
            for data in _stream_generate(ids, req.max_tokens):
                yield f"data: {json.dumps(data)}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    tokens, total = _collect(ids, req.max_tokens)
    return JSONResponse({
        "text": tok.decode(tokens, skip_special_tokens=True),
        "n_tokens": len(tokens),
        "total_s": total,
        "tok_per_s": len(tokens) / total if total > 0 else 0.0,
    })


@app.post("/v1/completions")
def v1_completions(req: CompletionRequest):
    tok = STATE["tok"]
    ids = tok(req.prompt).input_ids
    prompt_tokens = len(ids)

    if req.stream:
        def gen():
            cid = f"cmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            for data in _stream_generate(ids, req.max_tokens):
                chunk = {
                    "id": cid,
                    "object": "text_completion",
                    "created": created,
                    "model": req.model or STATE["model_name"],
                    "choices": [{
                        "index": 0,
                        "text": data["token"],
                        "finish_reason": "stop" if data["finished"] else None,
                    }],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    tokens, _ = _collect(ids, req.max_tokens)
    return JSONResponse({
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model or STATE["model_name"],
        "choices": [{
            "index": 0,
            "text": tok.decode(tokens, skip_special_tokens=True),
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(tokens),
            "total_tokens": prompt_tokens + len(tokens),
        },
    })


@app.post("/v1/chat/completions")
def v1_chat_completions(req: ChatRequest):
    tok = STATE["tok"]
    messages = [m.model_dump() for m in req.messages]
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = tok(prompt).input_ids
    prompt_tokens = len(ids)
    eos_id = STATE["chat_eos_id"]

    if req.stream:
        def gen():
            cid = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            for data in _stream_generate(ids, req.max_tokens, eos_id=eos_id):
                chunk = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model or STATE["model_name"],
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": data["token"]},
                        "finish_reason": "stop" if data["finished"] else None,
                    }],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    tokens, _ = _collect(ids, req.max_tokens, eos_id=eos_id)
    text = tok.decode(tokens, skip_special_tokens=True)
    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or STATE["model_name"],
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(tokens),
            "total_tokens": prompt_tokens + len(tokens),
        },
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=_INITIAL_CFG.model_path or "weights/Qwen2.5-32B-Instruct",
                   help="Path to HF model directory. Env: JLLM_MODEL_PATH.")
    p.add_argument("--model-name", default=None,
                   help="Name exposed on /v1 endpoints (default: last path component)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--max-num-seqs", type=int, default=4)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-prefill-len", type=int, default=512)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    args = p.parse_args()

    print(f"config: attention_impl={_INITIAL_CFG.attention_impl} "
          f"jit_cache={_INITIAL_CFG.compilation_cache_dir or '<disabled>'}", flush=True)
    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    print(f"loading {args.model_path} ({args.dtype})...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = load_from_path(args.model_path, dtype=dtype)
    driver = engine.make_driver(
        model,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        max_prefill_len=args.max_prefill_len,
        dtype=dtype,
    )
    engine.start(driver)

    STATE["tok"] = tok
    STATE["driver"] = driver
    STATE["model_path"] = args.model_path
    STATE["model_name"] = args.model_name or args.model_path.rstrip("/").split("/")[-1]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    STATE["chat_eos_id"] = im_end if isinstance(im_end, int) and im_end >= 0 else tok.eos_token_id
    print(f"serving on http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
