import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, load_dataset, load_from_disk
from huggingface_hub import whoami

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Silence noisy loggers
for noisy in ("httpx", "datasets", "filelock"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)


def load_curated_dataset(dataset_path: Optional[str], dataset_name: Optional[str]) -> Dataset:
    """Load the curated SFT dataset from a local disk path or a HuggingFace dataset name."""
    if dataset_path and dataset_name:
        raise ValueError("Provide only one of --dataset-path or --dataset-name")
    if dataset_path:
        dataset = load_from_disk(dataset_path)
        logger.info(f"Loaded curated dataset from disk: {dataset_path} ({len(dataset)} rows)")
        return dataset
    if dataset_name:
        dataset = load_dataset(dataset_name, split="train")
        logger.info(f"Loaded curated dataset from Hub: {dataset_name} ({len(dataset)} rows)")
        return dataset
    raise ValueError("One of --dataset-path or --dataset-name is required")


def export_predictions(dataset: Dataset, preds_path: Path, model_name: str) -> None:
    """Write preds.jsonl in the format expected by the SWE-bench harness."""
    required = {"instance_id", "generated_diff"}
    missing_cols = required - set(dataset.column_names)
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required columns for harness evaluation: {sorted(missing_cols)}"
        )

    preds_path.parent.mkdir(parents=True, exist_ok=True)
    empty_diff_count = 0
    with preds_path.open("w", encoding="utf-8") as f:
        for example in dataset:
            if not str(example["generated_diff"] or "").strip():
                empty_diff_count += 1
            record = {
                "instance_id": example["instance_id"],
                "model_name_or_path": model_name,
                "model_patch": example["generated_diff"] or "",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Exported {len(dataset)} predictions to {preds_path}")
    if empty_diff_count:
        logger.warning(
            f"{empty_diff_count}/{len(dataset)} rows have an empty generated_diff; "
            "the harness will mark them as unresolved."
        )


def load_harness_results(
    instance_results_path: Optional[Path], results_path: Optional[Path]
) -> dict[str, bool]:
    """Load pass/fail labels from harness output (instance_results.jsonl or results.json)."""
    if instance_results_path:
        if not instance_results_path.exists():
            raise FileNotFoundError(f"instance results file not found: {instance_results_path}")
        resolved_map: dict[str, bool] = {}
        with instance_results_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                instance_id = record.get("instance_id")
                if instance_id is None:
                    continue
                # Missing/unknown resolved means not passing.
                resolved_map[str(instance_id)] = bool(record.get("resolved", False))
        logger.info(f"Loaded {len(resolved_map)} instance results from {instance_results_path}")
        return resolved_map

    if results_path:
        if not results_path.exists():
            raise FileNotFoundError(f"results file not found: {results_path}")
        with results_path.open("r", encoding="utf-8") as f:
            results = json.load(f)
        resolved_ids = results.get("resolved_ids", results.get("RESOLVED_IDS", []))
        unresolved_ids = results.get("unresolved_ids", results.get("UNRESOLVED_IDS", []))
        resolved_map = {str(i): True for i in resolved_ids}
        for i in unresolved_ids:
            resolved_map[str(i)] = False
        logger.info(
            f"Loaded summary results from {results_path}: "
            f"{len(resolved_ids)} resolved, {len(unresolved_ids)} unresolved"
        )
        return resolved_map

    raise ValueError(
        "No harness results provided. Pass --instance-results or --results, "
        "or run with only --preds-out to export predictions first."
    )


def merge_passed(
    dataset: Dataset,
    resolved_map: dict[str, bool],
    output_path: Path,
    source_desc: str,
) -> Dataset:
    """Add a boolean ``passed`` column and save the labeled dataset."""
    def add_passed(example: dict[str, Any]) -> dict[str, Any]:
        return {"passed": resolved_map.get(str(example["instance_id"]), False)}

    labeled = dataset.map(add_passed)

    matched = sum(
        1 for instance_id in dataset["instance_id"] if str(instance_id) in resolved_map
    )
    passed = sum(1 for value in labeled["passed"] if value)
    failed = len(labeled) - passed

    base_desc = getattr(labeled.info, "description", None) or ""
    pass_fail_desc = (
        "\n\n## Pass/Fail Labels\n"
        f"- `passed`: real pass/fail labels produced by the SWE-bench harness "
        f"(source: {source_desc})\n"
        f"- {passed}/{len(labeled)} rollouts resolved the problem; "
        f"{len(labeled) - matched} rows had no harness result and default to `False`"
    )
    if base_desc:
        labeled.info.description = base_desc.rstrip() + pass_fail_desc
    else:
        labeled.info.description = pass_fail_desc.lstrip("\n")

    if output_path.exists():
        raise FileExistsError(
            f"Output path already exists: {output_path}. "
            "Choose a new output path or remove the existing directory first."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.save_to_disk(str(output_path))

    logger.info(f"Labeled dataset saved to {output_path}")
    logger.info(
        f"Pass/fail stats: {len(labeled)} total | {passed} passed | "
        f"{failed} failed | {len(labeled) - matched} missing results"
    )
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate real pass/fail labels for curated SFT rollouts.\n\n"
            "Stage 1 (any machine):\n"
            "  --dataset-path data/<ds> --preds-out data/<ds>/preds.jsonl\n\n"
            "Stage 2 (on the training machine with Docker/Apptainer):\n"
            "  benchmarks/swe_bench/run_harness_eval.sh --subset swegym --split train "
            "--preds data/<ds>/preds.jsonl --run-id sft_pass_fail_v1\n\n"
            "Stage 3 (merge labels back):\n"
            "  --dataset-path data/<ds> "
            "--instance-results evaluation_results/<run_id>/instance_results.jsonl "
            "--output-path data/<ds>_with_passed"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset-path", help="Local path to the curated dataset (load_from_disk)")
    parser.add_argument("--dataset-name", help="HuggingFace dataset name of the curated dataset")
    parser.add_argument("--preds-out", type=Path, help="Where to write preds.jsonl for the harness")
    parser.add_argument(
        "--model-name",
        default="nano-agent",
        help="Model label written into preds.jsonl (default: nano-agent)",
    )
    parser.add_argument(
        "--instance-results",
        type=Path,
        help="instance_results.jsonl produced by the SWE-bench harness",
    )
    parser.add_argument(
        "--results",
        type=Path,
        help="results.json produced by the SWE-bench harness (fallback summary)",
    )
    parser.add_argument("--output-path", type=Path, help="Where to save the labeled dataset")
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the labeled dataset to the HuggingFace Hub",
    )
    parser.add_argument(
        "--hub-dataset-name",
        help="Hub dataset name for --push-to-hub (default: ASSERT-KTH/<output dir name>)",
    )
    args = parser.parse_args()

    if not args.preds_out and not args.instance_results and not args.results:
        parser.error("Provide --preds-out and/or harness results (--instance-results/--results)")

    dataset = load_curated_dataset(args.dataset_path, args.dataset_name)

    source_desc = "unknown"
    if args.preds_out:
        export_predictions(dataset, args.preds_out, args.model_name)
        source_desc = str(args.preds_out)

    if args.instance_results or args.results:
        if not args.output_path:
            parser.error("--output-path is required when merging harness results")
        resolved_map = load_harness_results(args.instance_results, args.results)
        merge_passed(dataset, resolved_map, args.output_path, source_desc)

        if args.push_to_hub:
            try:
                whoami()
            except Exception:
                raise ValueError(
                    "Not logged in to HuggingFace. Please run 'huggingface-cli login' first."
                )
            hub_name = args.hub_dataset_name or f"ASSERT-KTH/{args.output_path.name}"
            labeled = load_from_disk(str(args.output_path))
            labeled.push_to_hub(hub_name)
            logger.info(f"Pushed labeled dataset to https://huggingface.co/datasets/{hub_name}")
    else:
        logger.info(
            "Predictions exported. Run the SWE-bench harness on the training machine, "
            "then re-run this script with --instance-results/--results and --output-path "
            "to merge pass/fail labels."
        )


if __name__ == "__main__":
    main()
