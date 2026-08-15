"""
Aggregate SWE-bench harness results across the evaluation matrix:

    swegym   - in-domain (training domain; contamination risk noted)
    verified - community reference (SWE-bench Verified)
    lite     - community reference (SWE-bench Lite)
    r2e-gym  - out-of-domain / contamination probe (R2E-Gym, SWE-Bench format)

Usage:
    python benchmarks/swe_bench/aggregate_eval_matrix.py \
      --run-id-prefix my_model_v1 \
      [--runs-dir evaluation_results] \
      [--sets swegym,verified,r2e-gym] \
      [--report-out evaluation_results/my_model_v1_matrix.md]
"""

import argparse
import json
from pathlib import Path
from typing import Any, Optional

SET_NOTES = {
    "swegym": "in-domain (SWE-Gym; training domain, contamination risk)",
    "verified": "community reference (SWE-bench Verified)",
    "lite": "community reference (SWE-bench Lite)",
    "r2e-gym": "out-of-domain / contamination probe (R2E-Gym SWE-Bench format)",
}


def load_set_results(runs_dir: Path, prefix: str, set_name: str) -> Optional[dict[str, Any]]:
    """Read instance_results.jsonl (or results.json fallback) for one set."""
    run_dir = runs_dir / f"{prefix}_{set_name}"
    instance_file = run_dir / "instance_results.jsonl"
    if instance_file.exists():
        resolved = 0
        total = 0
        with instance_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                total += 1
                if record.get("resolved", False):
                    resolved += 1
        return {
            "source": str(instance_file),
            "total": total,
            "resolved": resolved,
        }

    summary_file = run_dir / "results.json"
    if summary_file.exists():
        with summary_file.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        resolved_ids = summary.get("resolved_ids", [])
        unresolved_ids = summary.get("unresolved_ids", [])
        return {
            "source": str(summary_file),
            "total": len(resolved_ids) + len(unresolved_ids),
            "resolved": len(resolved_ids),
        }
    return None


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| eval set | instances | resolved | pass@1 | note |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        pass_rate = (
            f"{row['resolved'] / row['total']:.1%}"
            if row.get("total")
            else "-"
        )
        lines.append(
            f"| {row['set']} | {row.get('total', '-')} | "
            f"{row.get('resolved', '-')} | {pass_rate} | {row['note']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate the evaluation matrix from harness results"
    )
    parser.add_argument("--runs-dir", default="evaluation_results")
    parser.add_argument("--run-id-prefix", required=True)
    parser.add_argument("--sets", default="swegym,verified,r2e-gym")
    parser.add_argument("--report-out", help="Optional .md or .json report path")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    rows = []
    for set_name in sets:
        results = load_set_results(runs_dir, args.run_id_prefix, set_name)
        row: dict[str, Any] = {
            "set": set_name,
            "note": SET_NOTES.get(set_name, "custom eval set"),
        }
        if results is None:
            row["status"] = "no results"
        else:
            row.update(results)
            row["pass@1"] = (
                round(results["resolved"] / results["total"], 4)
                if results["total"]
                else 0.0
            )
        rows.append(row)

    for row in rows:
        if row.get("status") == "no results":
            print(f"{row['set']}: NO RESULTS (missing run dir)")
        else:
            print(
                f"{row['set']}: {row['resolved']}/{row['total']} "
                f"({row['resolved'] / row['total']:.1%}) - {row['note']}"
            )

    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if report_path.suffix == ".md":
            report_path.write_text(render_markdown(rows), encoding="utf-8")
        else:
            report_path.write_text(
                json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
