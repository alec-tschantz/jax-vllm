#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)/"
REMOTE="${1:-${JLLM_REMOTE:-gpu-droplet}}"
REMOTE_DIR="${2:-${JLLM_REMOTE_DIR:-/root/jax-vllm}}"
REMOTE_PATH="${REMOTE}:${REMOTE_DIR%/}/"

cmd=(
  rsync -az --delete
  --exclude=.venv
  --exclude=.uv-cache
  --exclude=weights
  --exclude=__pycache__
  --exclude=.pytest_cache
  --exclude='.*cache'
  --exclude=uv.lock
  --exclude=.git
  "$LOCAL_DIR" "$REMOTE_PATH"
)

if [[ "${JLLM_DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}"
echo "synced $LOCAL_DIR -> $REMOTE_PATH"
