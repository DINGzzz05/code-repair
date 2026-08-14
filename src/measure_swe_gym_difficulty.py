"""
Measure per-instance difficulty on the RL (repo-repair) partition by running
``num_rollouts`` agent rollouts per instance and saving the raw results for the
harness to label with real pass/fail.

Usage:
  1) Run this script (on the machine with the agent environment):
     uv run python src/measure_swe_gym_difficulty.py \
       --dataset-name SWE-Gym/SWE-Gym --num-rollouts 8 --backend apptainer

  2) Export predictions and run the SWE-bench harness:
     uv run python src/merge_sft_pass_fail.py \
       --dataset-path data/swe_gym_difficulty_measurement \
       --preds-out data/swe_gym_difficulty_measurement/preds.jsonl
     benchmarks/swe_bench/run_harness_eval.sh \
       --subset swegym --split train \
       --preds data/swe_gym_difficulty_measurement/preds.jsonl \
       --run-id swe_gym_difficulty_v1

  3) Merge labels back and write the difficulty map:
     uv run python src/merge_sft_pass_fail.py \
       --dataset-path data/swe_gym_difficulty_measurement \
       --instance-results evaluation_results/swe_gym_difficulty_v1/instance_results.jsonl \
       --output-path data/swe_gym_difficulty_measurement_with_passed \
       --difficulty-out data/swe_gym_difficulty.jsonl

The produced difficulty.jsonl maps instance_id -> (n_passed, n_total) and is
consumed by ``get_swe_gym_repo_repair_dataset(..., difficulty_path=...)``.
"""

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from datasets import Dataset, DatasetInfo

from src.data.swe_gym import SWE_GYM_HOLDOUT_RATIO, get_swe_gym_repo_repair_dataset
from src.agents.nano_agent import NanoConfig, _process_one

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

for noisy in ("httpx", "LiteLLM", "transformers.tokenization_utils_base"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)


def process_one(
    problem_data: dict[str, Any], config: NanoConfig, dataset_name: str
) -> dict[str, Any]:
    """Run one agent rollout and attach SWE-Gym metadata."""
    result = _process_one(problem_data, config, dataset_name)
    return {
        "instance_id": problem_data["instance_id"],
        "problem_statement": problem_data["problem_statement"],
        "repo": problem_data["repo"],
        "base_commit": problem_data["base_commit"],
        "oracle_diff": problem_data["patch"],
        "oracle_test_diff": problem_data["test_patch"],
        "generated_diff": result["generated_diff"],
        "messages": result["prompt"] + result["completion"],
        "tools": result["tools"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure per-instance pass rates on the RL partition for difficulty labeling"
    )
    parser.add_argument("--dataset-name", default="SWE-Gym/SWE-Gym")
    parser.add_argument("--num-rollouts", type=int, default=8, help="Rollouts per instance")
    parser.add_argument("--model", default="gemini/gemini-2.5-flash")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--backend", default="apptainer", choices=["local", "apptainer", "docker"])
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-problems", type=int, default=None, help="For testing")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output-dir", default="data/swe_gym_difficulty_measurement")
    args = parser.parse_args()

    logger.info(
        f"Loading RL (repo-repair) partition of {args.dataset_name} "
        f"(holdout_ratio={SWE_GYM_HOLDOUT_RATIO})..."
    )
    dataset = get_swe_gym_repo_repair_dataset(
        dataset_name=args.dataset_name, holdout_ratio=SWE_GYM_HOLDOUT_RATIO
    )
    if args.max_problems:
        dataset = dataset.select(range(min(args.max_problems, len(dataset))))

    tasks = [
        dict(problem_data)
        for problem_data in dataset
        for _ in range(args.num_rollouts)
    ]
    logger.info(
        f"Measured partition: {len(dataset)} instances -> {len(tasks)} rollouts "
        f"({args.num_rollouts} per instance)"
    )

    agent_config = NanoConfig(model=args.model, api_base=args.api_base, backend=args.backend)
    all_solutions = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(process_one, task, agent_config, args.dataset_name)
            for task in tasks
        ]
        for future in as_completed(futures):
            completed += 1
            try:
                result = future.result(timeout=args.timeout)
                all_solutions.append(result)
                logger.info(
                    f"[{completed}/{len(tasks)}] Rollout done for {result['instance_id']}"
                )
            except Exception as exc:
                logger.error(f"[{completed}/{len(tasks)}] Rollout failed: {exc}")

    if not all_solutions:
        logger.error("No rollouts produced; aborting")
        return

    info = DatasetInfo(
        description=(
            "Raw rollouts on the SWE-Gym repo-repair (RL) partition, produced by "
            f"measure_swe_gym_difficulty.py ({args.num_rollouts} rollouts per instance). "
            "Label these with real pass/fail via merge_sft_pass_fail.py, then write the "
            "per-instance difficulty map with --difficulty-out."
        )
    )
    rollout_dataset = Dataset.from_list(all_solutions, info=info)
    rollout_dataset.save_to_disk(args.output_dir)
    logger.info(f"Saved {len(all_solutions)} labeled-ready rollouts to {args.output_dir}")
    logger.info(
        "Next steps:\n"
        f"  1) uv run python src/merge_sft_pass_fail.py --dataset-path {args.output_dir} "
        f"--preds-out {args.output_dir}/preds.jsonl\n"
        "  2) benchmarks/swe_bench/run_harness_eval.sh --subset swegym --split train "
        f"--preds {args.output_dir}/preds.jsonl --run-id swe_gym_difficulty_v1\n"
        "  3) uv run python src/merge_sft_pass_fail.py --dataset-path "
        f"{args.output_dir} --instance-results "
        "evaluation_results/swe_gym_difficulty_v1/instance_results.jsonl "
        f"--output-path {args.output_dir}_with_passed "
        "--difficulty-out data/swe_gym_difficulty.jsonl"
    )


if __name__ == "__main__":
    main()
