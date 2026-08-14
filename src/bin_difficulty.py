"""
Dynamically bin RL instances into relative difficulty buckets after the full
pass-rate measurement, so that easy/hard are defined by the observed
distribution instead of absolute pass counts.

Input:  difficulty.jsonl from ``merge_sft_pass_fail.py --difficulty-out``,
        one record per instance: {"instance_id", "n_passed", "n_total"}
Output: relabeled difficulty.jsonl, one record per instance:
        {"instance_id", "n_passed", "n_total",
         "difficulty_rank" (0 = easiest .. 1 = hardest),
         "difficulty_bin" (0 = hardest .. K-1 = easiest),
         "difficulty_tier" ("easy" | "medium" | "hard")}

Usage:
    uv run python src/bin_difficulty.py \
      --difficulty-in data/swe_gym_difficulty.jsonl \
      --difficulty-out data/swe_gym_difficulty_relative.jsonl \
      --num-bins 5 --min-bin-size 0.05
"""

import argparse
import hashlib
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Difficulty input not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} measured instances from {path}")
    return records


def _seeded_tiebreak(instance_id: str, seed: int) -> int:
    return int(hashlib.md5(f"{seed}:{instance_id}".encode()).hexdigest(), 16)


def _pass_rate(record: dict[str, Any]) -> float:
    n_passed = int(record["n_passed"])
    n_total = int(record["n_total"])
    return n_passed / n_total if n_total > 0 else 0.0


def _tier_for(bin_index: int, num_bins: int) -> str:
    if num_bins <= 1:
        return "medium"
    difficulty = 1.0 - bin_index / (num_bins - 1)  # 0 = easiest, 1 = hardest
    if difficulty < 1 / 3:
        return "easy"
    if difficulty > 2 / 3:
        return "hard"
    return "medium"


def _merge_small_groups(
    groups: list[list[dict[str, Any]]], min_size: int
) -> list[list[dict[str, Any]]]:
    groups = [list(g) for g in groups]
    while len(groups) > 1:
        small_idx = next(
            (i for i, g in enumerate(groups) if len(g) < min_size), None
        )
        if small_idx is None:
            break
        # Merge into the easier neighbor (previous); the first bin merges forward.
        target_idx = 1 if small_idx == 0 else small_idx - 1
        merged = groups[small_idx] + groups[target_idx]
        remaining = [
            g for i, g in enumerate(groups) if i not in (small_idx, target_idx)
        ]
        insert_pos = min(small_idx, target_idx)
        groups = remaining[:insert_pos] + [merged] + remaining[insert_pos:]
        logger.info(
            f"Merged undersized bin (size {len(groups[insert_pos])}) with "
            f"neighbor to keep min_bin_size"
        )
    return groups


def bin_instances(
    records: list[dict[str, Any]],
    num_bins: int = 5,
    min_bin_size: float = 0.05,
    n_zero_policy: str = "keep",
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Assign relative difficulty bins based on the measured pass-rate
    distribution. Levels (distinct n_passed) are aggregated into at most
    ``num_bins`` bins; undersized bins are merged with a neighbor.
    """
    if n_zero_policy not in ("keep", "exclude", "remeasure"):
        raise ValueError(
            f"Unknown n_zero_policy '{n_zero_policy}' (expected keep|exclude|remeasure)"
        )
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1")

    total = len(records)
    drop_n0 = n_zero_policy in ("exclude", "remeasure")
    kept = [
        r for r in records if not (drop_n0 and int(r["n_passed"]) <= 0)
    ] if drop_n0 else list(records)
    excluded_n0 = total - len(kept)

    # Easiest first: higher pass rate, then higher absolute n, then stable tie-break.
    sorted_recs = sorted(
        kept,
        key=lambda r: (
            -_pass_rate(r),
            -int(r["n_passed"]),
            _seeded_tiebreak(str(r["instance_id"]), seed),
        ),
    )

    # Group into levels by n_passed (assumes a fixed measurement N).
    levels: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_n: Optional[int] = None
    for record in sorted_recs:
        n = int(record["n_passed"])
        if current_n is None or n != current_n:
            if current:
                levels.append(current)
            current = []
            current_n = n
        current.append(record)
    if current:
        levels.append(current)

    # Dynamic bin count: at most num_bins, but never more than distinct levels.
    effective_k = min(num_bins, len(levels))
    target_size = math.ceil(len(kept) / effective_k) if effective_k else 0
    groups: list[list[dict[str, Any]]] = []
    idx = 0
    for g in range(effective_k - 1):
        group: list[dict[str, Any]] = []
        while (
            len(group) < target_size
            and idx < len(levels) - (effective_k - 1 - g)
        ):
            group.extend(levels[idx])
            idx += 1
        groups.append(group)
    if idx < len(levels):
        groups.append([r for level in levels[idx:] for r in level])

    min_size = (
        int(min_bin_size * total) if min_bin_size < 1 else int(min_bin_size)
    )
    groups = _merge_small_groups(groups, min_size)

    # Label bins (0 = hardest .. K-1 = easiest) and ranks (0 = easiest .. 1 = hardest).
    for group_index, group in enumerate(groups):  # group 0 is easiest
        bin_index = len(groups) - 1 - group_index
        for record in group:
            record["difficulty_bin"] = bin_index
            record["difficulty_tier"] = _tier_for(bin_index, len(groups))
    for position, record in enumerate(sorted_recs):
        record["difficulty_rank"] = (
            position / (len(sorted_recs) - 1) if len(sorted_recs) > 1 else 0.0
        )

    n_distribution: dict[str, int] = {}
    for record in kept:
        n = str(int(record["n_passed"]))
        n_distribution[n] = n_distribution.get(n, 0) + 1

    bins_report = []
    for group_index, group in enumerate(groups):
        rates = [_pass_rate(r) for r in group]
        bins_report.append(
            {
                "bin": len(groups) - 1 - group_index,
                "size": len(group),
                "avg_pass_rate": round(sum(rates) / len(rates), 4) if rates else 0.0,
                "n_passed_levels": sorted({int(r["n_passed"]) for r in group}),
            }
        )

    report = {
        "total_measured": total,
        "kept": len(kept),
        "excluded_n0": excluded_n0,
        "n_zero_policy": n_zero_policy,
        "n_distribution": dict(sorted(n_distribution.items(), key=lambda kv: int(kv[0]))),
        "distinct_n_levels": len(levels),
        "target_bins": num_bins,
        "effective_bins": len(groups),
        "min_bin_size": min_size,
        "bins": bins_report,
    }
    return kept, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dynamic relative difficulty binning for the RL curriculum"
    )
    parser.add_argument("--difficulty-in", type=Path, required=True)
    parser.add_argument("--difficulty-out", type=Path, required=True)
    parser.add_argument("--num-bins", type=int, default=5)
    parser.add_argument(
        "--min-bin-size",
        type=float,
        default=0.05,
        help="Minimum bin size: fraction of total (<1) or absolute count (>=1)",
    )
    parser.add_argument(
        "--n-zero-policy",
        default="keep",
        choices=["keep", "exclude", "remeasure"],
        help="How to treat instances with no passing rollout",
    )
    parser.add_argument(
        "--remeasure-out",
        type=Path,
        help="Where to write n=0 instance ids when --n-zero-policy=remeasure",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args()

    if args.n_zero_policy == "remeasure" and not args.remeasure_out:
        parser.error("--remeasure-out is required when --n-zero-policy=remeasure")

    records = load_records(args.difficulty_in)
    labeled, report = bin_instances(
        records,
        num_bins=args.num_bins,
        min_bin_size=args.min_bin_size,
        n_zero_policy=args.n_zero_policy,
        seed=args.seed,
    )

    args.difficulty_out.parent.mkdir(parents=True, exist_ok=True)
    with args.difficulty_out.open("w", encoding="utf-8") as f:
        for record in labeled:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(labeled)} relabeled instances to {args.difficulty_out}")

    meta_path = args.difficulty_out.with_name(args.difficulty_out.stem + ".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "input": str(args.difficulty_in),
                "num_bins": args.num_bins,
                "min_bin_size": args.min_bin_size,
                "n_zero_policy": args.n_zero_policy,
                "seed": args.seed,
                "effective_bins": report["effective_bins"],
                "total_measured": report["total_measured"],
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info(f"Binning metadata written to {meta_path}")

    if args.n_zero_policy == "remeasure":
        zero_ids = [
            str(r["instance_id"])
            for r in records
            if int(r["n_passed"]) <= 0
        ]
        args.remeasure_out.parent.mkdir(parents=True, exist_ok=True)
        args.remeasure_out.write_text(
            "\n".join(zero_ids) + ("\n" if zero_ids else ""),
            encoding="utf-8",
        )
        logger.info(
            f"Wrote {len(zero_ids)} n=0 instance ids to {args.remeasure_out}; "
            "re-run measurement with more rollouts for these instances."
        )

    summary = (
        f"Difficulty binning report: {report['total_measured']} measured, "
        f"{report['kept']} kept, {report['excluded_n0']} n=0 excluded | "
        f"n distribution {report['n_distribution']} | "
        f"{report['distinct_n_levels']} levels -> {report['effective_bins']} bins "
        f"(target {report['target_bins']}, min size {report['min_bin_size']})"
    )
    logger.info(summary)
    for bin_info in report["bins"]:
        logger.info(
            f"  bin {bin_info['bin']}: {bin_info['size']} instances, "
            f"avg pass rate {bin_info['avg_pass_rate']}, "
            f"n levels {bin_info['n_passed_levels']}"
        )
    if report["effective_bins"] <= 1:
        logger.warning(
            "Only one difficulty bin remains: no curriculum discrimination. "
            "Consider a stronger teacher or more measurement rollouts."
        )

    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(f"Report written to {args.report_out}")


if __name__ == "__main__":
    main()
