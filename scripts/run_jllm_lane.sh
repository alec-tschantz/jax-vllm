#!/usr/bin/env bash
set -euo pipefail

GPU="${JLLM_GPU:-0}"
HOST="${JLLM_HOST:-0.0.0.0}"
PORT="${JLLM_PORT:-8080}"
MODEL_PATH="${JLLM_MODEL_PATH:-weights/Qwen2.5-32B-Instruct}"
MODEL_NAME="${JLLM_MODEL_NAME:-$(basename "$MODEL_PATH")}"
MAX_NUM_SEQS="${JLLM_MAX_NUM_SEQS:-4}"
MAX_MODEL_LEN="${JLLM_MAX_MODEL_LEN:-4096}"
MAX_PREFILL_LEN="${JLLM_MAX_PREFILL_LEN:-512}"
DTYPE="${JLLM_DTYPE:-bf16}"
ATTENTION_IMPL="${JLLM_ATTENTION_IMPL:-einsum}"
UV_CACHE_DIR_VALUE="${UV_CACHE_DIR:-.uv-cache}"
CACHE_ROOT="${JLLM_JAX_CACHE_ROOT:-/tmp/jllm-jax-cache}"
CACHE_DIR="${JLLM_JAX_CACHE_DIR:-${CACHE_ROOT}/gpu${GPU}-port${PORT}}"
PERSIST_MIN_COMPILE="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}"
PERSIST_MIN_ENTRY="${JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES:-0}"
# Disable XLA command-buffer scheduling and hipBLASLt. Both have been the
# observed source of intermittent `rocblas_gemm_ex failed` crashes when the
# engine is re-entered after a warm pytest rerun on MI300X. The overrides
# land as env vars so a caller can unset them by exporting an empty value.
XLA_FLAGS_DEFAULT="--xla_gpu_enable_command_buffer="
XLA_FLAGS_VALUE="${XLA_FLAGS:-$XLA_FLAGS_DEFAULT}"
ROCBLAS_USE_HIPBLASLT_VALUE="${ROCBLAS_USE_HIPBLASLT:-0}"
LOG_PATH="${JLLM_LOG_PATH:-/tmp/jllm-${PORT}.log}"

cmd=(
  env
  "HIP_VISIBLE_DEVICES=${GPU}"
  "UV_CACHE_DIR=${UV_CACHE_DIR_VALUE}"
  "JLLM_ATTENTION_IMPL=${ATTENTION_IMPL}"
  "JAX_COMPILATION_CACHE_DIR=${CACHE_DIR}"
  "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=${PERSIST_MIN_COMPILE}"
  "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=${PERSIST_MIN_ENTRY}"
  "XLA_FLAGS=${XLA_FLAGS_VALUE}"
  "ROCBLAS_USE_HIPBLASLT=${ROCBLAS_USE_HIPBLASLT_VALUE}"
  uv run jllm-serve
  --host "$HOST"
  --port "$PORT"
  --model-path "$MODEL_PATH"
  --model-name "$MODEL_NAME"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-model-len "$MAX_MODEL_LEN"
  --max-prefill-len "$MAX_PREFILL_LEN"
  --dtype "$DTYPE"
)

if [[ "${JLLM_DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '> %q 2>&1 &\n' "$LOG_PATH"
  exit 0
fi

mkdir -p "$(dirname "$LOG_PATH")" "$CACHE_DIR"
nohup "${cmd[@]}" >"$LOG_PATH" 2>&1 &
echo "started jllm lane gpu=${GPU} port=${PORT} cache=${CACHE_DIR} log=${LOG_PATH} pid=$!"
