#!/usr/bin/env bash
set -euo pipefail

GPU="${VLLM_GPU:-1}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8020}"
MODEL_PATH="${VLLM_MODEL_PATH:-/weights/Qwen2.5-32B-Instruct}"
MODEL_NAME="${VLLM_MODEL_NAME:-qwen32b}"
DTYPE="${VLLM_DTYPE:-bfloat16}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
UV_CACHE_DIR_VALUE="${UV_CACHE_DIR:-.uv-cache}"
LOG_PATH="${VLLM_LOG_PATH:-/tmp/vllm-${PORT}.log}"
if command -v vllm >/dev/null 2>&1; then
  VLLM_BIN="${VLLM_BIN:-$(command -v vllm)}"
else
  VLLM_BIN="${VLLM_BIN:-/usr/local/bin/vllm}"
fi

cmd=(
  env
  "HIP_VISIBLE_DEVICES=${GPU}"
  "UV_CACHE_DIR=${UV_CACHE_DIR_VALUE}"
  "$VLLM_BIN" serve "$MODEL_PATH"
  --host "$HOST"
  --port "$PORT"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --dtype "$DTYPE"
  --served-model-name "$MODEL_NAME"
)

if [[ "${VLLM_DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '> %q 2>&1 &\n' "$LOG_PATH"
  exit 0
fi

mkdir -p "$(dirname "$LOG_PATH")"
nohup "${cmd[@]}" >"$LOG_PATH" 2>&1 &
echo "started vllm lane gpu=${GPU} port=${PORT} log=${LOG_PATH} pid=$!"
