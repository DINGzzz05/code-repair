#!/usr/bin/env bash
set -euo pipefail

# Apptainer-based SWE-bench harness (no Docker required).
# Same CLI as run_harness_eval.sh:
#   benchmarks/swe_bench/run_harness_eval_apptainer.sh \
#     --subset verified --split test \
#     --preds /abs/path/to/preds.jsonl \
#     --run-id my_run [--max-workers 16] [--registry docker.m.daocloud.io]

uv run python benchmarks/swe_bench/run_harness_eval_apptainer.py "$@"
