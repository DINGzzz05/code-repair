import logging
import hashlib
from collections import defaultdict
from typing import Any, Optional

from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)


# Single source of truth for the holdout ratio.
# SFT curation uses the holdout partition; GRPO repo-repair uses its exact
# complement. Always split with this ratio on both sides, otherwise the two
# partitions overlap and data leaks between SFT and RL training.
SWE_GYM_HOLDOUT_RATIO = 0.25

PASS_FIELDS = ("passed", "pass", "resolved", "success", "status")
PASS_STRING_VALUES = {"pass", "passed", "true", "success", "resolved", "1"}


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
    
    # Add a dummy "prompt" key for compatibility with trl
    swe_ds = swe_ds.map(lambda x: {"prompt": "Dummy"})
    
    return swe_ds

# mirroring the other data methods though not strictly doing much
def get_swe_gym_repo_repair_dataset(
    dataset_name: str,
    holdout_ratio: float = SWE_GYM_HOLDOUT_RATIO,
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
    return _get_swe_gym_split(dataset_name, holdout_partition=False, holdout_ratio=holdout_ratio)

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


def get_swe_gym_formatted_sft_dataset(
    dataset_name: str,
    only_passed: bool = True,
    pass_field: Optional[str] = None,
    max_passed_per_instance: Optional[int] = 4,
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
        
    Returns:
        The formatted dataset ready for SFT training
    """
    logger.info(f"Loading curated SFT dataset: {dataset_name}")
    
    # Load the curated dataset
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
    
    return dataset

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
