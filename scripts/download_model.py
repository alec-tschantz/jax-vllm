import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def main(repo_id: str) -> None:
    local = Path("weights") / repo_id.split("/")[-1]
    local.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id,
        local_dir=str(local),
        allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt"],
    )
    print(path)


if __name__ == "__main__":
    main(sys.argv[1])
