#!/usr/bin/env bash
# Launch a vLLM serving lane inside the rocm/vllm-dev image. vLLM is not
# installed on the host; the container is the only place it lives on this
# droplet. Weights are mounted read-only at /weights inside the container.
#
# Stop: `docker stop "$VLLM_CONTAINER"` (default name: jllm-vllm-lane-$PORT).
set -euo pipefail

GPU="${VLLM_GPU:-1}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8020}"
WEIGHTS_DIR="${VLLM_WEIGHTS_DIR:-/root/jax-vllm/weights}"
MODEL_SUBDIR="${VLLM_MODEL_SUBDIR:-Qwen2.5-32B-Instruct}"
MODEL_PATH_IN_CONTAINER="/weights/${MODEL_SUBDIR}"
SERVED_NAME="${VLLM_MODEL_NAME:-$MODEL_SUBDIR}"
DTYPE="${VLLM_DTYPE:-bfloat16}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
IMAGE="${VLLM_IMAGE:-rocm/vllm-dev:rocm7.2.1_v0.18.0_20260330}"
CONTAINER_NAME="${VLLM_CONTAINER:-jllm-vllm-lane-${PORT}}"
LOG_PATH="${VLLM_LOG_PATH:-/tmp/vllm-${PORT}.log}"

cmd=(
  docker run --rm -d
  --name "$CONTAINER_NAME"
  --network host
  --ipc host
  --device /dev/kfd
  --device /dev/dri
  --group-add video
  --security-opt seccomp=unconfined
  -e "HIP_VISIBLE_DEVICES=${GPU}"
  -v "${WEIGHTS_DIR}:/weights:ro"
  "$IMAGE"
  vllm serve "$MODEL_PATH_IN_CONTAINER"
  --host "$HOST"
  --port "$PORT"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --dtype "$DTYPE"
  --served-model-name "$SERVED_NAME"
)

if [[ "${VLLM_DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$(dirname "$LOG_PATH")"
# Remove a stale container if one is left over from a previous run.
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

cid="$("${cmd[@]}")"
docker logs -f "$CONTAINER_NAME" >"$LOG_PATH" 2>&1 &
echo "started vllm lane gpu=${GPU} port=${PORT} container=${CONTAINER_NAME} log=${LOG_PATH} cid=${cid:0:12}"
