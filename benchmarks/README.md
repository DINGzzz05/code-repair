# Benchmark Suite

## Setup

Build the benchmark container (includes vLLM and all deps):

```bash
cd benchmarks/
apptainer build benchmark_container.sif benchmark_container.def
```

## Run Flow

### Inference (on the GPU cluster)

Submit a single SLURM job that launches vLLM (optional) and runs the SWE-bench evaluation to produce only `preds.jsonl`.

1) Ensure `CRRL_WORKDIR` is set (used by Apptainer bindings and caches):
```bash
export CRRL_WORKDIR="/proj/<project>/users/<user>"
```

2) Submit the job (starts its own vLLM server):
```bash
# Nano agent
sbatch benchmarks/swe_bench_nano_infer_job.sh \
  --base-model Qwen/Qwen3-8B
# With LoRA (adapter name auto-derived from lora path basename)
sbatch benchmarks/swe_bench_nano_infer_job.sh \
  --base-model Qwen/Qwen3-8B \
  --lora-path /path/to/nano_lora

# Mini-SWE-Agent (mini-swe-agent)
sbatch benchmarks/swe_bench_mini_infer_job.sh \
  --base-model Qwen/Qwen3-8B
# With LoRA
sbatch benchmarks/swe_bench_mini_infer_job.sh \
  --base-model Qwen/Qwen3-8B \
  --lora-path /path/to/adapter
```

Outputs are organized per model and scaffold under:

```
benchmarks/swe_bench/results/<scaffold>-<model_tag>/shard_<array_id>/preds.jsonl
```

Examples:
- Nano, base model: `benchmarks/swe_bench/results/nano-agent-Qwen__Qwen3-8B/shard_0/preds.jsonl`
- Nano, with LoRA (basename "nano_lora"): `benchmarks/swe_bench/results/nano-agent-Qwen__Qwen3-8B__lora__nano_lora/shard_0/preds.jsonl`
- Mini, base model: `benchmarks/swe_bench/results/mini-swe-agent-Qwen__Qwen3-8B/shard_0/preds.jsonl`
- Mini, with LoRA (basename "adapter"): `benchmarks/swe_bench/results/mini-swe-agent-Qwen__Qwen3-8B__lora__adapter/shard_0/preds.jsonl`

Notes:
- `<model_tag>` is sanitized to be filesystem-safe. For base-only runs it derives from `<BASE_MODEL>`; for LoRA runs it derives from `<BASE_MODEL>__lora__<adapter_basename>`.
- The agent model name is set automatically: if a LoRA is provided, it uses the LoRA adapter name (basename of the LoRA path); otherwise it uses the base model name.
- The job scripts support SLURM arrays. Each task writes to its own `shard_<array_id>` and auto-selects a dataset slice (default shard size 50).

### Eval (on the CPU server)

Then on a CPU server with Docker and the SWE-bench harness installed, run evaluation:
```bash
# Install harness once on CPU machine:
pip install swebench

# Evaluate a preds.jsonl file:
benchmarks/swe_bench/run_harness_eval.sh \
  --subset verified --split test \
  --preds /PATH/TO/preds.jsonl \
  --run-id nano_test
```

### Generating pass/fail labels for SFT curation

Curated SFT rollouts (`curate_sft_data.py`) have no labels yet. Real pass/fail
labels are produced by running the SWE-bench harness on the generated patches
and merging the `resolved` results back as a `passed` column:

```bash
# 1) Export predictions from the curated dataset (any machine):
uv run python src/merge_sft_pass_fail.py \
  --dataset-path data/ASSERT-KTH-Nano-SFT-SWE-Gym-gemini-2.5-flash-v1.0 \
  --preds-out data/ASSERT-KTH-Nano-SFT-SWE-Gym-gemini-2.5-flash-v1.0/preds.jsonl

# 2) Run the harness on the training machine (Docker/Apptainer, SWE-Gym images pre-pulled):
benchmarks/swe_bench/run_harness_eval.sh \
  --subset swegym --split train \
  --preds data/ASSERT-KTH-Nano-SFT-SWE-Gym-gemini-2.5-flash-v1.0/preds.jsonl \
  --run-id sft_pass_fail_v1

# 3) Merge the results back into the dataset (any machine):
uv run python src/merge_sft_pass_fail.py \
  --dataset-path data/ASSERT-KTH-Nano-SFT-SWE-Gym-gemini-2.5-flash-v1.0 \
  --instance-results evaluation_results/sft_pass_fail_v1/instance_results.jsonl \
  --output-path data/ASSERT-KTH-Nano-SFT-SWE-Gym-gemini-2.5-flash-v1.0_with_passed
```

The labeled dataset can then be loaded for SFT training with
`only_passed=true` (see `src/data/swe_gym.py`), which keeps only rows where
`passed` is true. Rows without a harness result default to `passed=false`.
After filtering, at most 4 passing rollouts per instance are kept, preferring
the shortest trajectories (configurable via `max_passed_per_instance`).

## Results

Outputs are saved under:
- `benchmarks/tau_bench/results/`
- `benchmarks/swe_bench/results/<scaffold>-<model_tag>/shard_<array_id>/`
Evaluation logs and reports (from the harness) will be written under the harness working directory (e.g., `evaluation_results/`). See the harness docs for details.
