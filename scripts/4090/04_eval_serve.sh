#!/usr/bin/env bash
#
# 4x RTX 4090 (24GB) safe plan - Stage 3a: plain vLLM server on GPU3 for eval.
# (Eval does not need TRL weight sync, so plain `vllm serve` is fine.)
#
# Usage:
#   scripts/4090/04_eval_serve.sh [MODEL_PATH] [PORT]
#
# Defaults:
#   MODEL_PATH=grpo_repo_repair_model_merged
#   PORT=8001
#
set -euo pipefail
cd "$(dirname "$0")/../.."

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"

MODEL="${1:-grpo_repo_repair_model_merged}"
PORT="${2:-8001}"

echo "=== Eval vLLM start: model=${MODEL}, GPU=3, port=${PORT} ==="

CUDA_VISIBLE_DEVICES=3 uv run vllm serve "$MODEL" \
  --port "$PORT" \
  --max-model-len 16384 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --reasoning-parser qwen3 \
  --tool-call-parser hermes
