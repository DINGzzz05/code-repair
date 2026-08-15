#!/usr/bin/env bash
#
# 4x RTX 4090 (24GB) safe plan - Stage 1: SFT (Qwen3-8B LoRA) on GPU0-1.
#
# Usage:
#   scripts/4090/01_sft_8b.sh [RUN_NAME]
#
# Defaults:
#   RUN_NAME=crrl_8b_sft_v1  ->  merged model saved to outputs/crrl_8b_sft_v1_merged
#
# If you change RUN_NAME, update src/conf/model/small_qwen_sft.yaml model_name
# accordingly before starting GRPO.
#
set -euo pipefail
cd "$(dirname "$0")/../.."

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"

RUN_NAME="${1:-crrl_8b_sft_v1}"

echo "=== SFT start: RUN_NAME=${RUN_NAME}, GPUs=0,1 ==="

CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch \
  --num_processes=2 --num_machines=1 \
  --mixed_precision=bf16 --dynamo_backend=no \
  --module src.train_sft -- \
    model=small_qwen model.attn_implementation=flash_attention_2 \
    model.lora=true model.r=8 model.lora_alpha=16 \
    sft.kl_lambda=0.0 \
    sft.max_length=8192 sft.packing=false \
    sft.per_device_train_batch_size=1 sft.gradient_accumulation_steps=16 \
    sft.num_train_epochs=1 \
    sft.run_name="${RUN_NAME}" \
    run.wandb_project=SWE-Gym-SFT

echo "=== SFT done. Merged model: outputs/${RUN_NAME}_merged ==="
