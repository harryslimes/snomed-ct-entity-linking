#!/usr/bin/env bash
set -euo pipefail

# Keep the model loaded for fast iteration. Run in a dedicated terminal.
#
# Example:
#   scripts/llm_postprocess/serve_sglang_fp8.sh Qwen/Qwen3-30B-A3B-Instruct-2507-FP8

MODEL_PATH="${1:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"

export PYTHONNOUSERSITE=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

exec /usr/bin/python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --quantization fp8 \
  --context-length 32768 \
  --max-total-tokens 32768 \
  --allow-auto-truncate \
  --mem-fraction-static 0.55 \
  --max-running-requests 64 \
  --disable-cuda-graph \
  --attention-backend triton \
  --fp8-gemm-backend triton \
  --moe-runner-backend triton \
  --speculative-moe-runner-backend triton

