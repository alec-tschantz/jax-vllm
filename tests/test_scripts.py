import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run_script(script: str, env: dict[str, str]) -> str:
    result = subprocess.run(
        ["bash", str(SCRIPTS_DIR / script)],
        cwd=REPO_ROOT,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def test_bench_json_output(tmp_path, monkeypatch):
    bench = _load_module(SCRIPTS_DIR / "bench_vllm.py", "bench_vllm_test")

    class DummyJllm:
        def __init__(self):
            self.stats_calls = 0

        def info(self):
            return {"backend": "http", "attention_impl": "einsum"}

        def stats(self):
            self.stats_calls += 1
            return {"decode_batches": self.stats_calls}

    monkeypatch.setattr(bench, "HTTPJllm", lambda url, model_name: DummyJllm())
    monkeypatch.setattr(bench, "_safe_health", lambda url: {"status": "ok"})
    monkeypatch.setattr(bench, "_git_sha", lambda: "deadbeef")
    monkeypatch.setattr(bench, "_runner_metadata", lambda: {"host": "gpu-test", "cwd": "/srv/jax-vllm"})
    monkeypatch.setattr(
        bench,
        "run_sequential",
        lambda args, jllm, vllm_url, vllm_model: (0, {"mode": "sequential", "summary": {"jllm_tok_per_s": 1.0}}),
    )
    out_path = tmp_path / "bench.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_vllm.py",
            "--jllm-url",
            "http://127.0.0.1:8080",
            "--json-out",
            str(out_path),
            "--label",
            "smoke",
        ],
    )

    rc = bench.main()
    payload = json.loads(out_path.read_text())

    assert rc == 0
    assert payload["metadata"]["label"] == "smoke"
    assert payload["metadata"]["git_sha"] == "deadbeef"
    assert payload["metadata"]["runner"] == {"host": "gpu-test", "cwd": "/srv/jax-vllm"}
    assert payload["metadata"]["workload"] == "balanced"
    assert payload["metadata"]["jllm_stats_delta"]["decode_batches"] == 1
    assert payload["metadata"]["token_metrics"]["budgeted"] == "num_requests * max_new_tokens"
    assert payload["result"]["summary"]["jllm_tok_per_s"] == 1.0


def test_mixed_workload_prompts_cover_multiple_lengths():
    bench = _load_module(SCRIPTS_DIR / "bench_vllm.py", "bench_vllm_workload_test")

    prompts = bench._request_prompts("mixed", 12)
    lengths = [len(prompt) for prompt in prompts]
    summary = bench._prompt_summary(prompts)

    assert len(prompts) == 12
    assert min(lengths) < 32
    assert max(lengths) > 300
    assert summary["min_chars"] == min(lengths)
    assert summary["max_chars"] == max(lengths)
    assert summary["unique_lengths"] >= 3


def test_concurrency_warmup_skips_bucket_one(monkeypatch):
    bench = _load_module(SCRIPTS_DIR / "bench_vllm.py", "bench_vllm_warmup_test")
    calls: list[tuple[str, str]] = []

    class DummyJllm:
        def completion(self, prompt: str, max_tokens: int, state: str = "warm"):
            calls.append(("jllm", state))
            return {"text": "", "total_s": 0.01, "n_tokens": max_tokens, "tok_per_s": 100.0, "state": state}

    args = SimpleNamespace(concurrency=[1, 4, 4], warmups=1, max_new_tokens=8, jllm_only=False)

    monkeypatch.setattr(
        bench,
        "_vllm_completion",
        lambda url, model, prompt, max_tokens: calls.append(("vllm", "warmup")) or {
            "text": "",
            "total_s": 0.01,
            "n_tokens": max_tokens,
            "tok_per_s": 100.0,
            "state": "warm",
        },
    )

    warmed = bench._warmup_concurrency_levels(
        args, DummyJllm(), "http://127.0.0.1:8020", "qwen32b", ["hello"]
    )

    assert warmed == [4]
    assert calls.count(("vllm", "warmup")) == 4
    assert calls.count(("jllm", "warmup-c4")) == 4


def test_unique_prompt_warmup_deduplicates_prompts(monkeypatch):
    bench = _load_module(SCRIPTS_DIR / "bench_vllm.py", "bench_vllm_prompt_warmup_test")
    calls: list[tuple[str, str]] = []

    class DummyJllm:
        def completion(self, prompt: str, max_tokens: int, state: str = "warm"):
            calls.append(("jllm", prompt))
            return {"text": "", "total_s": 0.01, "n_tokens": max_tokens, "tok_per_s": 100.0, "state": state}

    monkeypatch.setattr(
        bench,
        "_vllm_completion",
        lambda url, model, prompt, max_tokens: calls.append(("vllm", prompt)) or {
            "text": "",
            "total_s": 0.01,
            "n_tokens": max_tokens,
            "tok_per_s": 100.0,
            "state": "warm",
        },
    )

    warmed = bench._warmup_unique_prompts(
        DummyJllm(),
        "http://127.0.0.1:8020",
        "qwen32b",
        ["alpha", "beta", "alpha", "gamma", "beta"],
    )

    assert warmed == 3
    assert calls == [
        ("jllm", "alpha"),
        ("vllm", "alpha"),
        ("jllm", "beta"),
        ("vllm", "beta"),
        ("jllm", "gamma"),
        ("vllm", "gamma"),
    ]


def test_stats_delta_uses_zero_for_missing_keys():
    bench = _load_module(SCRIPTS_DIR / "bench_vllm.py", "bench_vllm_stats_test")

    assert bench._stats_delta({"prefill_batches": 2, "decode_batches": 1}, {"decode_batches": 4}) == {
        "decode_batches": 3,
        "prefill_batches": -2,
    }


def test_concurrent_row_tracks_observed_and_budget_tokens():
    bench = _load_module(SCRIPTS_DIR / "bench_vllm.py", "bench_vllm_concurrent_row_test")

    row = bench._concurrent_row(
        "jllm",
        concurrency=4,
        results=[
            {"n_tokens": 2, "total_s": 0.4},
            {"n_tokens": 3, "total_s": 0.6},
        ],
        wall_s=1.0,
        max_new_tokens=4,
    )

    assert row["tokens"] == 5
    assert row["tok_per_s"] == 5.0
    assert row["observed_tokens"] == 5
    assert row["observed_tok_per_s"] == 5.0
    assert row["budget_tokens"] == 8
    assert row["budget_tok_per_s"] == 8.0


def test_sync_script_dry_run():
    out = _run_script(
        "sync.sh",
        {
            "JLLM_DRY_RUN": "1",
            "JLLM_REMOTE": "gpu-test",
            "JLLM_REMOTE_DIR": "/srv/jax-vllm",
        },
    )
    assert "rsync" in out
    assert "gpu-test:/srv/jax-vllm/" in out


def test_remote_jllm_script_dry_run():
    out = _run_script(
        "run_jllm_lane.sh",
        {
            "JLLM_DRY_RUN": "1",
            "JLLM_GPU": "2",
            "JLLM_PORT": "8182",
            "JLLM_JAX_CACHE_ROOT": "/tmp/jax-cache",
        },
    )
    assert "HIP_VISIBLE_DEVICES=2" in out
    assert "JAX_COMPILATION_CACHE_DIR=/tmp/jax-cache/gpu2-port8182" in out
    assert "--port 8182" in out
    assert "/tmp/jllm-8182.log" in out

def test_remote_vllm_script_dry_run():
    out = _run_script(
        "run_vllm_lane.sh",
        {
            "VLLM_DRY_RUN": "1",
            "VLLM_GPU": "3",
            "VLLM_PORT": "9020",
            "VLLM_MODEL_NAME": "qwen32b-lane",
        },
    )
    assert "docker run" in out
    assert "HIP_VISIBLE_DEVICES=3" in out
    assert "vllm serve /weights/Qwen2.5-32B-Instruct" in out
    assert "--port 9020" in out
    assert "--served-model-name qwen32b-lane" in out
    assert "--network host" in out
    assert "/dev/kfd" in out
