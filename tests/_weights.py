from pathlib import Path

import pytest


def require_model_path(path: str) -> str:
    if not Path(path).exists():
        pytest.skip(f"missing model weights at {path}")
    return path
