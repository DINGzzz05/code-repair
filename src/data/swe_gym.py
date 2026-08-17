import logging
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from datasets import load_dataset, load_from_disk, Dataset

logger = logging.getLogger(__name__)


# Single source of truth for the holdout ratio.
# SFT curation uses the holdout partition; GRPO repo-repair uses its exact
# complement. Always split with this ratio on both sides, otherwise the two
# partitions overlap and data leaks between SFT and RL training.
SWE_GYM_HOLDOUT_RATIO = 0.25
DIFFICULTY_MODES = ("all", "easy", "hard", "curriculum")

PASS_FIELDS = ("passed", "pass", "resolved", "success", "status")
PASS_STRING_VALUES = {"pass", "passed", "true", "success", "resolved", "1"}

# Metadata columns that must never reach the SFT prompt/training path.
SFT_DROP_COLUMNS = {
    "oracle_diff",
    "oracle_test_diff",
    "patch",
    "test_patch",
    "repo",
    "base_commit",
    "problem_statement",
    "generated_diff",
    "prompt",
}


def _get_swe_gym_split(
    dataset_name: str,
    holdout_partition: bool,
    holdout_ratio: float = SWE_GYM_HOLDOUT_RATIO,
) -> Dataset:
    """
    Internal function to load and split the SWE-Gym dataset.
    
    Args:
        dataset_name: HuggingFace dataset name for SWE-bench
        holdout_partition: If True, return holdout partition; if False, return repo repair partition
        holdout_ratio: Ratio of data to allocate to holdout partition.
            Must be identical for the holdout and repo-repair partitions, which are
            exact complements of each other.
        
    Returns:
        The requested partition of the dataset
    """
    logger.info(f"Loading SWE-bench dataset: {dataset_name}")
    
    # Load the SWE-bench dataset
    swe_ds = load_dataset(dataset_name)
    swe_ds = swe_ds.get("train") or swe_ds.get("test")
    
    # Create deterministic split based on instance_id hash
    def should_be_holdout(example):
        # Use MD5 hash of instance_id for deterministic splitting
        hash_val = int(hashlib.md5(example['instance_id'].encode()).hexdigest(), 16)
        # Convert to [0, 1] range and compare with holdout_ratio
        return (hash_val / (16**32)) < holdout_ratio
    
    # Filter based on partition type
    if holdout_partition:
        swe_ds = swe_ds.filter(should_be_holdout)
        logger.info(f"Creating holdout dataset with {len(swe_ds)} examples")
    else:
        swe_ds = swe_ds.filter(lambda x: not should_be_holdout(x))
        logger.info(f"Creating repository repair dataset with {len(swe_ds)} examples")

    _log_partition_balance(
        swe_ds, "holdout" if holdout_partition else "repo_repair"
    )
    
    # Add a dummy "prompt" key for compatibility with trl
    swe_ds = swe_ds.map(lambda x: {"prompt": "Dummy"})
    
    return swe_ds


def _log_partition_balance(dataset: Dataset, partition_name: str) -> None:
    """Log size and repo distribution of a partition to surface split imbalance."""
    repos = Counter(str(example.get("repo", "unknown")) for example in dataset)
    top = repos.most_common(10)
    logger.info(
        f"{partition_name} partition balance: {len(dataset)} instances; "
        f"top repos {top[:5]}"
    )


def load_difficulty_map(difficulty_path: Optional[str]) -> dict[str, dict[str, Any]]:
    """
    Load per-instance difficulty labels from a difficulty.jsonl file produced by
    ``merge_sft_pass_fail.py --difficulty-out`` (absolute pass counts) or
    ``src/bin_difficulty.py`` (relative bins, ranks and tiers).

    Returns:
        Mapping of instance_id -> label dict
    """
    if not difficulty_path:
        return {}
    path = Path(difficulty_path)
    if not path.exists():
        raise FileNotFoundError(f"Difficulty file not found: {difficulty_path}")
    mapping: dict[str, tuple[int, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            instance_id = str(record["instance_id"])
            mapping[instance_id] = {
                "n_passed": int(record.get("n_passed", 0)),
                "n_total": int(record.get("n_total", 0)),
                "difficulty_bin": record.get("difficulty_bin"),
                "difficulty_rank": record.get("difficulty_rank"),
                "difficulty_tier": record.get("difficulty_tier"),
            }
    logger.info(
        f"Loaded difficulty labels for {len(mapping)} instances from {difficulty_path}"
    )
    return mapping


def add_difficulty_columns(
    dataset: Dataset, difficulty_map: dict[str, dict[str, Any]]
) -> Dataset:
    """
    Attach pass-rate difficulty columns to an RL dataset:
    ``n_passed``, ``n_total``, ``difficulty_score`` (n_passed / n_total) and
    ``difficulty_bin`` (relative bin when the map came from bin_difficulty.py,
    falling back to absolute n_passed), plus ``difficulty_rank`` and
    ``difficulty_tier`` when available. Instances missing from the map get -1
    markers and are excluded by ``easy``/``hard``/``curriculum`` modes.
    """
    def add(example: dict[str, Any]) -> dict[str, Any]:
        info = difficulty_map.get(str(example["instance_id"]))
        if info is None:
            return {
                "n_passed": -1,
                "n_total": -1,
                "difficulty_score": -1.0,
                "difficulty_bin": -1,
                "difficulty_rank": -1.0,
                "difficulty_tier": "unknown",
            }
        n_passed = info["n_passed"]
        n_total = info["n_total"]
        return {
            "n_passed": n_passed,
            "n_total": n_total,
            "difficulty_score": (n_passed / n_total) if n_total > 0 else -1.0,
            "difficulty_bin": (
                info["difficulty_bin"] if info["difficulty_bin"] is not None else n_passed
            ),
            "difficulty_rank": (
                info["difficulty_rank"]
                if info["difficulty_rank"] is not None
                else -1.0
            ),
            "difficulty_tier": (
                info["difficulty_tier"] if info["difficulty_tier"] else "unknown"
            ),
        }

    return dataset.map(add)


def _filter_by_difficulty(
    dataset: Dataset, difficulty: str, threshold: Optional[int]
) -> Dataset:
    """Keep easy/hard instances based on their measured pass counts."""
    dataset = dataset.filter(lambda x: int(x["n_passed"]) >= 0)
    measured = sorted(int(x) for x in dataset["n_passed"])
    if not measured:
        logger.warning("No measured difficulty labels available; returning empty dataset")
        return dataset
    split = threshold if threshold is not None else measured[len(measured) // 2]
    if difficulty == "easy":
        kept = dataset.filter(lambda x: int(x["n_passed"]) >= split)
    else:  # hard
        kept = dataset.filter(lambda x: int(x["n_passed"]) < split)
    logger.info(
        f"Difficulty filter '{difficulty}' (threshold {split}): "
        f"{len(dataset)} -> {len(kept)} instances"
    )
    return kept


# mirroring the other data methods though not strictly doing much
def get_swe_gym_repo_repair_dataset(
    dataset_name: str,
    holdout_ratio: float = SWE_GYM_HOLDOUT_RATIO,
    difficulty: str = "all",
    difficulty_path: Optional[str] = None,
    difficulty_threshold: Optional[int] = None,
    **kwargs  # absorbs additional arguments required by the other get functions
) -> Dataset:
    """
    Load the SWE-bench dataset and convert it to a repository repair dataset.
    This function returns the exact complement of the holdout partition
    (for the same ``holdout_ratio``) for RL/GRPO training.
    
    Args:
        dataset_name: HuggingFace dataset name for SWE-bench
        holdout_ratio: Ratio of data to allocate to the holdout partition.
            Keep it equal to ``SWE_GYM_HOLDOUT_RATIO`` so that this partition
            never overlaps with the SFT holdout partition.
        
    Returns:
        The processed dataset (repo repair partition)
    """
    dataset = _get_swe_gym_split(
        dataset_name, holdout_partition=False, holdout_ratio=holdout_ratio
    )

    if difficulty not in DIFFICULTY_MODES:
        raise ValueError(
            f"Unknown difficulty '{difficulty}' (expected one of {DIFFICULTY_MODES})"
        )
    if difficulty != "all" and not difficulty_path:
        raise ValueError(
            "difficulty_path is required when difficulty != 'all'. "
            "Run measure_swe_gym_difficulty.py + merge_sft_pass_fail.py --difficulty-out first."
        )
    if difficulty == "all":
        return dataset

    dataset = add_difficulty_columns(dataset, load_difficulty_map(difficulty_path))
    if difficulty in ("easy", "hard"):
        dataset = _filter_by_difficulty(dataset, difficulty, difficulty_threshold)
    else:  # curriculum mode keeps all measured instances (bins are used by the annealer)
        before = len(dataset)
        dataset = dataset.filter(lambda x: int(x["n_passed"]) >= 0)
        logger.info(
            f"Difficulty curriculum mode: kept {len(dataset)}/{before} "
            "instances with measured pass counts"
        )
    return dataset

def get_swe_gym_holdout_dataset(
    dataset_name: str,
    holdout_ratio: float = SWE_GYM_HOLDOUT_RATIO,
    **kwargs  # absorbs additional arguments required by the other get functions
) -> Dataset:
    """
    Load the SWE-bench dataset for SFT data holdout via rejection sampling.
    This function returns the holdout partition of the data, ensuring no overlap 
    with the repository repair dataset. Used by curate_sft_data.py to generate
    high-quality SFT examples through multiple rollouts and filtering.
    
    Args:
        dataset_name: HuggingFace dataset name for SWE-bench
        holdout_ratio: Ratio of data to allocate to holdout partition.
            Keep it equal to ``SWE_GYM_HOLDOUT_RATIO`` so that this partition
            never overlaps with the GRPO repo-repair partition.
        
    Returns:
        The processed dataset (holdout partition for rejection sampling)
    """
    return _get_swe_gym_split(dataset_name, holdout_partition=True, holdout_ratio=holdout_ratio)


def _trajectory_length(example: dict[str, Any]) -> int:
    """
    Length of a rollout trajectory, measured as total characters across all
    message contents. Used to prefer shorter trajectories when capping the
    number of passing rollouts per instance.
    """
    total = 0
    for message in example.get("messages", []) or []:
        content = message.get("content", "") if isinstance(message, dict) else message
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += len(str(part.get("text", part.get("content", ""))))
                else:
                    total += len(str(part))
        else:
            total += len(str(content))
    return total


def _cap_passed_rollouts_per_instance(
    dataset: Dataset, max_per_instance: int
) -> tuple[Dataset, dict[str, int]]:
    """
    Keep at most ``max_per_instance`` passing rollouts per instance, preferring
    the shortest trajectories. Returns the capped dataset (in original order)
    and a mapping of instance_id -> number of passing rollouts before capping.
    """
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for index, example in enumerate(dataset):
        groups[str(example["instance_id"])].append((index, _trajectory_length(example)))

    passed_counts = {instance_id: len(items) for instance_id, items in groups.items()}
    kept_indices: list[int] = []
    capped_instances = 0
    removed = 0
    for items in groups.values():
        if len(items) > max_per_instance:
            # Shortest trajectories first; original order breaks ties.
            items.sort(key=lambda item: (item[1], item[0]))
            kept_indices.extend(index for index, _ in items[:max_per_instance])
            capped_instances += 1
            removed += len(items) - max_per_instance
        else:
            kept_indices.extend(index for index, _ in items)

    kept_indices.sort()
    capped = dataset.select(kept_indices)
    logger.info(
        f"Capped to {max_per_instance} passing rollouts per instance: "
        f"{capped_instances} instances had more, {removed} rollouts removed"
    )
    return capped, passed_counts


def _apply_static_weights(dataset: Dataset, weight_exponent: float) -> Dataset:
    """
    Assign each passing trajectory a static weight
    ``w = (num_total_rollouts / num_passed_rollouts) ** weight_exponent`` and
    resample the dataset proportionally to ``w`` via stochastic rounding (fixed
    seed for reproducibility). Instances with a low pass rate get a higher
    weight, so their (rare) passing trajectories contribute more to SFT
    training. An exponent of 1.0 reproduces the plain inverse pass rate (8:1
    for 1/8); smaller exponents soften the spread (0.5 turns it into ~2.8:1);
    a non-positive exponent disables weighting.
    """
    if weight_exponent <= 0:
        logger.info("Static sample weights disabled (weight_exponent <= 0)")
        return dataset

    required = {"num_total_rollouts", "num_passed_rollouts"}
    if not required.issubset(set(dataset.column_names)):
        logger.warning(
            "Static sample weights requested but the dataset lacks "
            f"{sorted(required)} columns; skipping weighting."
        )
        return dataset

    def add_weight(example: dict[str, Any]) -> dict[str, Any]:
        n_total = int(example["num_total_rollouts"])
        n_passed = int(example["num_passed_rollouts"])
        raw = (n_total / n_passed) if n_total > 0 and n_passed > 0 else 1.0
        weight = raw ** weight_exponent
        return {"sample_weight": float(weight)}

    dataset = dataset.map(add_weight)

    rng = random.Random(42)
    indices: list[int] = []
    for index, weight in enumerate(dataset["sample_weight"]):
        count = int(weight)
        if rng.random() < weight - count:
            count += 1
        indices.extend([index] * count)

    logger.info(
        f"Static pass-rate weights applied: {len(dataset)} -> {len(indices)} rows "
        "(low pass rate -> higher sample weight)"
    )
    return dataset.select(indices)


def _split_by_instance(
    dataset: Dataset, ratio: float, seed: int
) -> tuple[Dataset, Dataset]:
    """Split by instance (not by rollout) so no instance spans train and eval."""
    instance_ids = sorted({str(x) for x in dataset["instance_id"]})
    rng = random.Random(seed)
    rng.shuffle(instance_ids)
    n_eval = max(1, int(round(len(instance_ids) * ratio)))
    eval_ids = set(instance_ids[:n_eval])
    eval_part = dataset.filter(lambda x: str(x["instance_id"]) in eval_ids)
    train_part = dataset.filter(lambda x: str(x["instance_id"]) not in eval_ids)
    return train_part, eval_part


def _drop_sft_metadata(dataset: Dataset) -> Dataset:
    """Remove columns that must never reach the SFT prompt/training path."""
    columns = [c for c in dataset.column_names if c in SFT_DROP_COLUMNS]
    if columns:
        dataset = dataset.remove_columns(columns)
    return dataset


def get_swe_gym_formatted_sft_dataset(
    dataset_name: str,
    only_passed: bool = True,
    pass_field: Optional[str] = None,
    max_passed_per_instance: Optional[int] = 4,
    apply_sample_weights: bool = True,
    sample_weight_exponent: float = 0.5,
    validation_ratio: float = 0.0,
    validation_seed: int = 42,
    drop_metadata_columns: bool = True,
    **kwargs
) -> Dataset:
    """
    Load and format a curated SFT dataset for training.
    This function loads an already-curated dataset (created by curate_sft_data.py)
    and formats it for SFT training. Teacher trajectories are kept only when
    their pass/fail result indicates a successful fix.
    
    Args:
        dataset_name: HuggingFace dataset name for curated SFT data
        only_passed: If True, keep only passing trajectories. If False, return
            the full dataset without pass/fail filtering.
        pass_field: Optional name of the pass/fail column. If omitted, the
            loader auto-detects one of ``passed``, ``pass``, ``resolved``,
            ``success``, or ``status``.
        max_passed_per_instance: After pass/fail filtering, keep at most this
            many passing rollouts per instance, preferring the shortest
            trajectories (``None`` or a non-positive value disables the cap).
        apply_sample_weights: If True, weight trajectories by the inverse pass
            rate of their instance and resample the dataset proportionally to
            ``w``, so trajectories from instances with a low pass rate are
            up-weighted during SFT.
        sample_weight_exponent: Exponent applied to the inverse pass rate,
            ``w = (num_total_rollouts / num_passed_rollouts) ** exponent``.
            1.0 is the full-strength inverse pass rate; lower values (default
            0.5) soften the weight spread; ``<= 0`` disables weighting.
        validation_ratio: If > 0, hold out this fraction of *instances* as an
            eval split (never mixed with train) and return ``(train, eval)``.
        validation_seed: Seed for the instance-level eval split.
        drop_metadata_columns: Remove oracle/ground-truth and repo metadata
            columns from the returned dataset(s) to prevent prompt leakage.
        
    Returns:
        The formatted dataset ready for SFT training, or a ``(train, eval)``
        tuple when ``validation_ratio > 0``.
    """
    logger.info(f"Loading curated SFT dataset: {dataset_name}")

    # Load the curated dataset. Support a local save_to_disk directory
    # (e.g. produced by merge_sft_pass_fail.py --output-path) in addition to
    # a HuggingFace Hub dataset id.
    if Path(dataset_name).exists():
        logger.info(f"Local dataset directory detected, loading from disk: {dataset_name}")
        dataset = load_from_disk(dataset_name)
    else:
        dataset = load_dataset(dataset_name, split="train")
    
    logger.info(f"Preparing dataset with {len(dataset)} examples...")
    original_size = len(dataset)
    
    if not only_passed:
        logger.info("Pass/fail filtering disabled; keeping all examples")
        return dataset

    # Record total rollouts per instance before any filtering.
    total_counts: dict[str, int] = defaultdict(int)
    for instance_id in dataset["instance_id"]:
        total_counts[str(instance_id)] += 1

    fields = [pass_field] if pass_field else list(PASS_FIELDS)
    available_fields = [field for field in fields if field in dataset.column_names]
    if not available_fields:
        raise ValueError(
            "Pass/fail filtering is enabled, but the dataset has none of the "
            f"supported pass/fail columns: {', '.join(fields)}."
        )

    pass_field = available_fields[0]

    def is_pass_example(example: dict[str, Any]) -> bool:
        value = example[pass_field]
        if isinstance(value, str):
            return value.strip().lower() in PASS_STRING_VALUES
        return bool(value)

    dataset = dataset.filter(is_pass_example)
    logger.info(f"Filtered dataset from {original_size} to {len(dataset)} examples")

    if max_passed_per_instance and max_passed_per_instance > 0:
        dataset, passed_counts = _cap_passed_rollouts_per_instance(
            dataset, max_passed_per_instance
        )
    else:
        # Still record per-instance pass counts when the cap is disabled.
        passed_counts = defaultdict(int)
        for instance_id in dataset["instance_id"]:
            passed_counts[str(instance_id)] += 1

    dataset = dataset.map(
        lambda example: {
            "num_total_rollouts": total_counts[str(example["instance_id"])],
            "num_passed_rollouts": passed_counts[str(example["instance_id"])],
        }
    )

    train_part = dataset
    eval_part = None
    if validation_ratio and validation_ratio > 0:
        train_part, eval_part = _split_by_instance(
            dataset, validation_ratio, validation_seed
        )

    if apply_sample_weights:
        train_part = _apply_static_weights(
            train_part, weight_exponent=sample_weight_exponent
        )

    if drop_metadata_columns:
        train_part = _drop_sft_metadata(train_part)
        if eval_part is not None:
            eval_part = _drop_sft_metadata(eval_part)

    if eval_part is not None:
        logger.info(
            f"SFT validation split (instance-level, ratio={validation_ratio}): "
            f"{len(train_part)} train / {len(eval_part)} eval rollouts"
        )
        return train_part, eval_part
    return train_part

if __name__ == "__main__":
    ds = load_dataset("SWE-Gym/SWE-Gym-Lite")
    print(ds)
    
    # Test the split functions
    holdout_ds = get_swe_gym_holdout_dataset(dataset_name="SWE-Gym/SWE-Gym-Lite")
    repair_ds = get_swe_gym_repo_repair_dataset(dataset_name="SWE-Gym/SWE-Gym-Lite")
    
    print(f"Holdout dataset size: {len(holdout_ds)}")
    print(f"Repo repair dataset size: {len(repair_ds)}")
    print(f"Total: {len(holdout_ds) + len(repair_ds)}")
    
    # Verify no overlap
    holdout_ids = set(holdout_ds['instance_id'])
    repair_ids = set(repair_ds['instance_id'])
    overlap = holdout_ids.intersection(repair_ids)
    print(f"Overlap between partitions: {len(overlap)} items")
    assert len(overlap) == 0, "Holdout and repo-repair partitions must be disjoint"
