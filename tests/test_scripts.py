import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

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
        def info(self):
            return {"backend": "http", "attention_impl": "einsum"}

    monkeypatch.setattr(bench, "HTTPJllm", lambda url, model_name: DummyJllm())
    monkeypatch.setattr(bench, "_safe_health", lambda url: {"status": "ok"})
    monkeypatch.setattr(bench, "_git_sha", lambda: "deadbeef")
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
    assert payload["result"]["summary"]["jllm_tok_per_s"] == 1.0


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
        "sync.sh",
        {
            "JLLM_DRY_RUN": "1",
            "JLLM_REMOTE": "gpu-test",
            "JLLM_REMOTE_DIR": "/srv/jax-vllm",
        },
    )
    assert "gpu-test:/srv/jax-vllm/" in out
