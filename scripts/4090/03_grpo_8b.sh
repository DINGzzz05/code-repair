#!/usr/bin/env bash
#
# 4x RTX 4090 (24GB) safe plan - Stage 2b: GRPO/GSPO (Qwen3-8B LoRA) on GPU1-2.
# GPU0 must be serving the same model via scripts/4090/02_vllm_serve.sh first.
#
# Usage:
#   scripts/4090/03_grpo_8b.sh [RUN_NAME] [SMOKE]
#
#   RUN_NAME  default crrl_8b_grpo_v1
#   SMOKE     set to 1 to cap at grpo.max_steps=2 (pipeline check, ~10-25 min)
#
# Safe-plan key settings vs the stock medium script:
#   - 8B model (16GB bf16 weights fit in 24GB)
#   - grpo.beta=0.0            -> no reference model copy (+16GB saved)
#   - num_generations=4        -> group size 4 = 2 GPUs x per_device 2
#   - max_completion_length=7168 / vLLM max-model-len 9216
#   - max-num-seqs 4 on the vLLM card (see 02_vllm_serve.sh)
#   - num_train_epochs=1       -> ~225 rollout batches, roughly 20-30h
#
set -euo pipefail
cd "$(dirname "$0")/../.."

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"

RUN_NAME="${1:-crrl_8b_grpo_v1}"
SMOKE="${2:-0}"

EXTRA_ARGS=()
if [[ "${SMOKE}" == "1" ]]; then
  EXTRA_ARGS+=(grpo.max_steps=2)
  echo "=== SMOKE mode: capping at 2 optimizer steps ==="
fi

echo "=== GRPO start: RUN_NAME=${RUN_NAME}, GPUs=1,2 ==="

CUDA_VISIBLE_DEVICES=1,2 uv run accelerate launch \
  --main_process_port 43001 \
  --num_processes 2 \
  --config_file scripts/deepspeed/zero2.yaml \
  --module src.train_grpo -- \
    run=repo_repair \
    run.dataset_name="SWE-Gym/SWE-Gym" \
    run.difficulty=all \
    run.wandb_project=SWE-Gym-GRPO \
    run.push_to_hub=false \
    model=small_qwen_sft \
    agent.time_limit=60 \
    grpo=multi_turn_gspo \
    grpo.beta=0.0 \
    grpo.max_prompt_length=1024 \
    grpo.max_completion_length=7168 \
    grpo.num_generations=4 \
    grpo.generation_batch_size=8 \
    grpo.per_device_train_batch_size=2 \
    grpo.gradient_accumulation_steps=8 \
    grpo.num_train_epochs=1 \
    grpo.optim=paged_adamw_8bit \
    grpo.run_name="${RUN_NAME}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "=== GRPO done. Merged model: grpo_repo_repair_model_merged ==="
