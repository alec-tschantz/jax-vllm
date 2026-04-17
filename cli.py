"""Rich-based chat CLI that talks to the jllm HTTP server over /v1/chat/completions.

Maintains a conversation history so multi-turn chat works (each turn POSTs the
full messages list). The server applies the tokenizer's chat template and stops
at <|im_end|>.

Run locally after port-forwarding:

    ssh -L 8080:localhost:8080 gpu-droplet
    # in another terminal on your laptop:
    uv run python cli.py --url http://localhost:8080

Commands at the prompt:
    /reset   clear conversation history
    /quit    exit
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text


def _post_chat_stream(url: str, model: str, messages: list, max_tokens: int):
    """Stream /v1/chat/completions. Yields {'delta': str, 'finished': bool}."""
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                return
            ev = json.loads(payload)
            choice = ev["choices"][0]
            delta = choice.get("delta", {}).get("content", "")
            finished = choice.get("finish_reason") is not None
            yield {"delta": delta, "finished": finished}


def _panel_for(transcript: str, current_prompt: str, current_reply: str, stats: str, done: bool) -> Group:
    title = "[green]done[/]" if done else "[yellow]streaming[/]"
    panels = []
    if transcript:
        panels.append(Panel(Text(transcript, style="dim"), title="history", border_style="blue"))
    panels.append(Panel(Text(current_prompt, style="cyan"), title="you", border_style="cyan"))
    panels.append(Panel(
        Text(current_reply or " ", style="white"),
        title=title,
        subtitle=stats,
        border_style="green" if done else "yellow",
    ))
    return Group(*panels)


def _format_transcript(messages: list) -> str:
    """Render the prior conversation compactly above the live reply panel."""
    lines = []
    for m in messages:
        role = m["role"]
        tag = {"user": "you", "assistant": "bot", "system": "sys"}.get(role, role)
        content = m["content"].strip().replace("\n", " ")
        if len(content) > 120:
            content = content[:117] + "..."
        lines.append(f"[{tag}] {content}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8080")
    p.add_argument("--model", default=None,
                   help="Model name in the /v1/chat/completions payload. If omitted, "
                        "derived from the server's /health response.")
    p.add_argument("--max-tokens", type=int, default=1536)
    p.add_argument("--system", default=None,
                   help="Optional system prompt prepended to every conversation")
    args = p.parse_args()

    console = Console()

    try:
        with urllib.request.urlopen(f"{args.url}/health", timeout=5) as r:
            health = json.loads(r.read().decode())
        if args.model is None:
            # server's "model" field is typically "weights/<ModelName>"
            raw = health.get("model", "model")
            args.model = raw.rstrip("/").split("/")[-1] or "model"
        console.print(
            Panel.fit(
                Text.assemble(
                    ("jllm chat\n", "bold cyan"),
                    (f"model: {health.get('model', '?')}\n", "dim"),
                    (f"max_num_seqs={health.get('max_num_seqs')}  "
                     f"max_model_len={health.get('max_model_len')}  "
                     f"max_prefill_len={health.get('max_prefill_len')}\n", "dim"),
                    ("commands: /reset  /quit", "dim"),
                ),
                border_style="cyan",
            )
        )
    except Exception as e:
        console.print(f"[red]cannot reach server at {args.url}: {e}[/]")
        sys.exit(1)

    messages: list = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    console.print(Rule(style="dim"))
    while True:
        try:
            user_in = console.input("[bold green]>[/] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]bye[/]")
            break
        if not user_in.strip():
            continue
        if user_in.strip() in ("/quit", "/exit"):
            console.print("[dim]bye[/]")
            break
        if user_in.strip() == "/reset":
            messages = []
            if args.system:
                messages.append({"role": "system", "content": args.system})
            console.print("[dim]history cleared[/]")
            console.print(Rule(style="dim"))
            continue

        messages.append({"role": "user", "content": user_in})
        # Show prior turns (exclude the just-appended user turn, it's in the "you" panel).
        transcript = _format_transcript(messages[:-1])
        reply = ""
        t0 = time.perf_counter()
        n = 0
        ttft = None

        try:
            with Live(
                _panel_for(transcript, user_in, reply, "[dim]…[/]", False),
                console=console,
                refresh_per_second=20,
            ) as live:
                for ev in _post_chat_stream(args.url, args.model, messages, args.max_tokens):
                    piece = ev["delta"]
                    reply += piece
                    n += 1 if piece else 0
                    if ttft is None and piece:
                        ttft = time.perf_counter() - t0
                    elapsed = time.perf_counter() - t0
                    tok_s = n / elapsed if elapsed > 0 else 0.0
                    stats = (f"[dim]{n} tok · {tok_s:.1f} tok/s · "
                             f"ttft {ttft or 0:.2f}s · {elapsed:.1f}s[/]")
                    live.update(_panel_for(transcript, user_in, reply, stats, ev["finished"]))
                    if ev["finished"]:
                        break
        except urllib.error.URLError as e:
            console.print(f"[red]request failed: {e}[/]")
            messages.pop()  # roll back the user turn we just appended
            console.print(Rule(style="dim"))
            continue

        messages.append({"role": "assistant", "content": reply})
        console.print(Rule(style="dim"))


if __name__ == "__main__":
    main()
