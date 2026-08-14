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
Each remaining trajectory also carries a static weight
`w = (num_total_rollouts / num_passed_rollouts)^0.5` (stored as `sample_weight`)
and the dataset is resampled proportionally to `w`, so trajectories from
instances with a low pass rate contribute more to SFT training. The exponent
(`sample_weight_exponent`, default 0.5) controls the strength: 1.0 gives the
full 8:1 spread for a 1/8 pass rate, 0.5 softens it to about 2.8:1, and `<= 0`
disables weighting.

### RL difficulty curriculum (easy -> hard annealing)

The GRPO stage can anneal instance difficulty from easy to hard within a single
run. Difficulty is measured with the same pass-rate logic as SFT: run `N`
rollouts per instance on the RL (repo-repair) partition, label them with the
harness, and write a per-instance difficulty map:

```bash
# 1) Measure: N rollouts per instance on the RL partition
uv run python src/measure_swe_gym_difficulty.py \
  --dataset-name SWE-Gym/SWE-Gym --num-rollouts 8 --backend apptainer

# 2) Label with the harness and write difficulty.jsonl
uv run python src/merge_sft_pass_fail.py \
  --dataset-path data/swe_gym_difficulty_measurement \
  --preds-out data/swe_gym_difficulty_measurement/preds.jsonl
benchmarks/swe_bench/run_harness_eval.sh \
  --subset swegym --split train \
  --preds data/swe_gym_difficulty_measurement/preds.jsonl \
  --run-id swe_gym_difficulty_v1
uv run python src/merge_sft_pass_fail.py \
  --dataset-path data/swe_gym_difficulty_measurement \
  --instance-results evaluation_results/swe_gym_difficulty_v1/instance_results.jsonl \
  --output-path data/swe_gym_difficulty_measurement_with_passed \
  --difficulty-out data/swe_gym_difficulty.jsonl

# 3) Turn absolute pass counts into relative difficulty bins (distribution-driven)
uv run python src/bin_difficulty.py \
  --difficulty-in data/swe_gym_difficulty.jsonl \
  --difficulty-out data/swe_gym_difficulty_relative.jsonl \
  --num-bins 5 --min-bin-size 0.05
```

Then train GRPO with the annealing curriculum:

```bash
uv run python src/train_grpo.py \
  run=repo_repair \
  run.difficulty=curriculum \
  run.difficulty_path=data/swe_gym_difficulty_relative.jsonl \
  grpo.curriculum.enabled=true \
  grpo.run_name=swe_gym_curriculum_v1
```

Difficulty is defined relative to the measured distribution: `bin_difficulty.py`
aggregates distinct pass-count levels into at most `--num-bins` equal-ish bins
(merging undersized bins), so even when most instances have only 0-2 passing
rollouts the curriculum still has meaningful easy/medium/hard tiers. The
annealer starts at the easiest bin present and slides to the hardest bin,
sampling batches from a softmax window around the current center (radius
`curriculum.window`, sharpness `curriculum.tau`) while keeping a minimum
`easy_mix_floor` share of easiest-bin data to prevent forgetting.
`run.difficulty=easy|hard` (with an optional `run.difficulty_threshold`,
default median) selects a static subset instead of annealing.

### Fine-grained live difficulty loop

Optionally, difficulty labels refresh *during* RL training instead of using a
stale offline measurement. Every completed rollout updates the live difficulty
state with a cheap proxy signal (unified diff similarity vs. the oracle patch,
compared against a calibrated threshold); every `rebin_every_steps` steps the
relative bins are recomputed from the live pass-rate distribution, so the
annealing curriculum follows the model being trained.

```bash
uv run python src/train_grpo.py run=repo_repair \
  run.difficulty=curriculum \
  run.difficulty_path=data/swe_gym_difficulty.jsonl \
  grpo.curriculum.enabled=true \
  grpo.live_difficulty.enabled=true \
  grpo.live_difficulty.calibration_dataset=data/swe_gym_difficulty_measurement_with_passed \
  grpo.live_difficulty.rebin_every_steps=200 \
  grpo.run_name=swe_gym_live_curriculum_v1
```

The pseudo-pass threshold is auto-calibrated from the labeled measurement
dataset (maximizing F1 between diff similarity and real `passed` labels).

Optional async harness calibration (phase 2): point `live_difficulty.harness_dir`
at a directory and run the worker on the training machine. Every
`export_every_steps` steps the training process exports the best pending diff
per instance to `harness_dir/inbox`; the worker runs the harness and drops
`instance_results.jsonl` into `harness_dir/outbox`, which the training process
folds back as real pass/fail evidence:

```bash
uv run python src/harness_worker.py \
  --inbox data/live_harness/inbox \
  --outbox data/live_harness/outbox
```

## Results

Outputs are saved under:
- `benchmarks/tau_bench/results/`
- `benchmarks/swe_bench/results/<scaffold>-<model_tag>/shard_<array_id>/`
Evaluation logs and reports (from the harness) will be written under the harness working directory (e.g., `evaluation_results/`). See the harness docs for details.
