"""
Background harness worker for the fine-grained live difficulty loop.

Polls ``inbox`` for preds_*.jsonl files exported by the training process, runs
the SWE-bench harness on each file (in parallel with training), copies
instance_results.jsonl into ``outbox``, and removes the processed file.
The training process folds the outbox results back into the live difficulty
state on its periodic update.

Usage (on the training machine with Docker/Apptainer and swebench):
    uv run python src/harness_worker.py \
      --inbox data/live_harness/inbox \
      --outbox data/live_harness/outbox
"""

import argparse
import re
import logging
import shutil
import subprocess
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_harness(
    preds_path: Path,
    run_id: str,
    dataset_name: str,
    split: str,
    max_workers: int,
    timeout: int,
) -> Path:
    """Run the SWE-bench harness on one preds file and return its results path."""
    cmd = [
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--predictions_path",
        str(preds_path),
        "--max_workers",
        str(max_workers),
        "--run_id",
        run_id,
        "--cache_level",
        "instance",
        "--timeout",
        str(timeout),
    ]
    logger.info(f"Running harness: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    results_path = Path("evaluation_results") / run_id / "instance_results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"Harness output not found: {results_path}")
    return results_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Background SWE-bench harness worker for live difficulty calibration"
    )
    parser.add_argument("--inbox", required=True, help="Directory with preds_*.jsonl files")
    parser.add_argument("--outbox", required=True, help="Directory for instance_results.jsonl")
    parser.add_argument("--dataset-name", default="SWE-Gym/SWE-Gym")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--run-id-prefix", default="live_diff")
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    inbox = Path(args.inbox)
    outbox = Path(args.outbox)
    failed_dir = inbox / "failed"
    inbox.mkdir(parents=True, exist_ok=True)
    outbox.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Harness worker watching {inbox} -> {outbox} "
        f"(dataset {args.dataset_name}, split {args.split})"
    )

    while True:
        pending = sorted(inbox.glob("preds_*.jsonl"))
        for preds_path in pending:
            run_id = f"{args.run_id_prefix}_{preds_path.stem}"
            try:
                src = run_harness(
                    preds_path,
                    run_id,
                    args.dataset_name,
                    args.split,
                    args.max_workers,
                    args.timeout,
                )
                dest = outbox / f"{run_id}.instance_results.jsonl"
                shutil.copy2(src, dest)
                preds_path.unlink(missing_ok=True)
                logger.info(f"Harness finished: {len(dest.read_text(encoding='utf-8').splitlines())} results -> {dest}")
            except Exception as exc:
                logger.error(f"Harness failed for {preds_path.name}: {exc}")
                match = re.search(r"\.attempt(\d+)\.jsonl$", preds_path.name)
                attempt = int(match.group(1)) if match else 0
                if attempt < 3:
                    retry_name = preds_path.name.replace(
                        ".jsonl",
                        f".attempt{attempt + 1}.jsonl",
                        1,
                    )
                    preds_path.rename(preds_path.with_name(retry_name))
                    logger.info(f"Will retry {retry_name} on the next poll (attempt {attempt + 1}/3)")
                else:
                    preds_path.rename(failed_dir / preds_path.name)
                    logger.error(
                        f"Harness failed 3 times for {preds_path.name}; "
                        f"moved to {failed_dir} for manual review"
                    )
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
