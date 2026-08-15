#!/usr/bin/env bash
set -euo pipefail

# Run the evaluation matrix for a trained model:
#   swegym   - in-domain (SWE-Gym, training domain; treat as dev with contamination risk)
#   verified - community reference (SWE-bench Verified)
#   r2e-gym  - out-of-domain / contamination probe (R2E-Gym, SWE-Bench format)
#
# Expectations:
#   - preds dir contains one preds file per set: <set>_preds.jsonl
#   - the SWE-bench harness is installed and Docker/Apptainer is available
#
# Usage:
#   benchmarks/swe_bench/run_eval_matrix.sh \
#     --preds-dir /abs/path/to/preds \
#     --run-id-prefix my_model_v1 \
#     [--sets swegym,verified,r2e-gym] \
#     [--max-workers 16] \
#     [--report-out evaluation_results/my_model_v1_matrix.json]

preds_dir=""
run_id_prefix=""
sets="swegym,verified,r2e-gym"
max_workers="16"
report_out=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preds-dir)
      preds_dir="$2"; shift 2;;
    --run-id-prefix)
      run_id_prefix="$2"; shift 2;;
    --sets)
      sets="$2"; shift 2;;
    --max-workers)
      max_workers="$2"; shift 2;;
    --report-out)
      report_out="$2"; shift 2;;
    *)
      echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

if [[ -z "$preds_dir" || -z "$run_id_prefix" ]]; then
  echo "ERROR: --preds-dir and --run-id-prefix are required" >&2
  exit 1
fi

# set -> (harness subset, split)
declare -A SUBSET_SPLIT=(
  [swegym]="swegym train"
  [verified]="verified test"
  [lite]="lite test"
  [r2e-gym]="r2e-gym test"
)

IFS=',' read -ra SET_LIST <<< "$sets"
for set_name in "${SET_LIST[@]}"; do
  set_name="$(echo "$set_name" | xargs)"  # trim whitespace
  if [[ -z "${SUBSET_SPLIT[$set_name]+x}" ]]; then
    echo "ERROR: unknown set '$set_name' (expected: swegym|verified|lite|r2e-gym)" >&2
    exit 1
  fi
  preds_file="$preds_dir/${set_name}_preds.jsonl"
  if [[ ! -f "$preds_file" ]]; then
    echo "ERROR: predictions file not found: $preds_file" >&2
    exit 1
  fi
  read -r subset split <<< "${SUBSET_SPLIT[$set_name]}"
  echo "=== Eval set: $set_name ($subset / $split) ==="
  benchmarks/swe_bench/run_harness_eval.sh \
    --subset "$subset" --split "$split" \
    --preds "$preds_file" \
    --run-id "${run_id_prefix}_${set_name}" \
    --max-workers "$max_workers"
done

AGG_ARGS=(--runs-dir evaluation_results --run-id-prefix "$run_id_prefix" --sets "$sets")
if [[ -n "$report_out" ]]; then
  AGG_ARGS+=(--report-out "$report_out")
fi
python benchmarks/swe_bench/aggregate_eval_matrix.py "${AGG_ARGS[@]}"
