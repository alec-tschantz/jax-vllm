#!/usr/bin/env bash
set -euo pipefail
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)/"
REMOTE="gpu-droplet:/root/jax-vllm/"
rsync -az --delete \
  --exclude=.venv \
  --exclude=weights \
  --exclude=__pycache__ \
  --exclude=.pytest_cache \
  --exclude='.*cache' \
  --exclude=uv.lock \
  --exclude=.git \
  "$LOCAL_DIR" "$REMOTE"
echo "synced $LOCAL_DIR -> $REMOTE"
