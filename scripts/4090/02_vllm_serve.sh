#!/usr/bin/env bash
#
# 4x RTX 4090 (24GB) safe plan - Stage 2a: TRL async vLLM server on GPU0.
# Must be the TRL fork command (`trl vllm-serve-async`) so LoRA weight sync works.
#
# Usage:
#   scripts/4090/02_vllm_serve.sh [MODEL_PATH]
#
# Defaults:
#   MODEL_PATH=outputs/crrl_8b_sft_v1_merged
#
set -euo pipefail
cd "$(dirname "$0")/../.."

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"

MODEL="${1:-outputs/crrl_8b_sft_v1_merged}"

echo "=== vLLM (async) start: model=${MODEL}, GPU=0 ==="

CUDA_VISIBLE_DEVICES=0 uv run trl vllm-serve-async \
  --model "$MODEL" \
  --max-model-len 9216 \
  --disable-log-stats \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 4 \
  --enable-auto-tool-choice \
  --reasoning-parser qwen3 \
  --tool-call-parser hermes
