#!/usr/bin/env python3
"""
Apptainer-based SWE-bench harness evaluation (no Docker daemon, no root).

This is a drop-in replacement for ``python -m swebench.harness.run_evaluation``
for machines where Docker is unavailable but Apptainer can pull and run the
official ``swebench/sweb.eval.x86_64.*`` instance images unprivileged.

All spec generation, eval-script creation and grading reuse swebench 4.x
functions unchanged; only the container mechanics (pull / exec / patch / eval)
are implemented with ``apptainer exec``, so pass/fail semantics match the
official harness.

Usage:
  uv run python benchmarks/swe_bench/run_harness_eval_apptainer.py \
      --subset verified --split test \
      --preds /abs/path/to/preds.jsonl \
      --run-id my_run [--max-workers 16]

Registry: images are pulled via ``docker://<registry>/<image>``.
  * default: env APPTAINER_REGISTRY, else docker.m.daocloud.io
  * set --registry "" (or APPTAINER_REGISTRY="") to use Docker Hub directly

Outputs (same locations/schema as the official harness where applicable):
  logs/run_evaluation/<run_id>/<model>/<instance_id>/report.json
  logs/run_evaluation/<run_id>/<model>/<instance_id>/test_output.txt
  evaluation_results/<run_id>/instance_results.jsonl
  <model>.<run_id>.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.constants import MAP_REPO_TO_EXT, MAP_REPO_VERSION_TO_SPECS
from swebench.harness.test_spec.create_scripts import make_eval_script_list
from swebench.harness.test_spec.test_spec import TestSpec
from swebench.harness.utils import (
    get_predictions_from_file,
    load_swebench_dataset,
)

SUBSET_TO_DATASET = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "lite": "princeton-nlp/SWE-bench_Lite",
    "full": "princeton-nlp/SWE-bench",
    "swegym": "SWE-Gym/SWE-Gym",
    "swegym-lite": "SWE-Gym/SWE-Gym-Lite",
    "r2e-gym": "R2E-Gym/SWE-Bench-Verified",
}

# Patch application fallbacks (mirror swebench.harness.run_evaluation), with
# safe.directory injected so git works when /testbed is owned by a different
# user (the common case when running without --fakeroot).
APP_APPLY_CMDS = [
    "git -c safe.directory=/testbed apply --verbose",
    "git -c safe.directory=/testbed apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]

WORKDIR_IN_CONTAINER = "/testbed"
MOUNT_IN_CONTAINER = "/tmp/aeval"


def build_image_uri(image_key: str, registry: str) -> str:
    """
    Build the apptainer docker:// URI for an instance image.

    ``image_key`` is the full image reference (namespace is already baked in,
    e.g. ``swebench/sweb.eval.x86_64.<instance_id>:latest``); only the registry
    host needs to be prepended.
    """
    if registry:
        return f"docker://{registry}/{image_key}"
    return f"docker://{image_key}"


def _load_str_list(value: Any) -> list[str]:
    """Parse FAIL_TO_PASS / PASS_TO_PASS which may be JSON strings."""
    if isinstance(value, str):
        return json.loads(value)
    return list(value or [])


def build_test_spec_offline(
    instance: dict[str, Any], namespace: Optional[str] = None
) -> TestSpec:
    """
    Build a TestSpec without network access.

    swebench's ``make_test_spec()`` fetches per-repo requirements files from
    raw.githubusercontent.com to generate env/repo scripts. Those scripts are
    only needed for building images; when evaluating with prebuilt official
    instance images they are unnecessary. Skipping them lets the scorer work
    on machines where GitHub raw is unreachable.
    """
    repo = instance["repo"]
    version = instance["version"]
    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
    env_name = "testbed"
    repo_directory = f"/{env_name}"
    eval_script_list = make_eval_script_list(
        instance,
        specs,
        env_name,
        repo_directory,
        instance["base_commit"],
        instance["test_patch"],
    )
    return TestSpec(
        instance_id=instance[KEY_INSTANCE_ID],
        repo=repo,
        version=version,
        repo_script_list=[],
        eval_script_list=eval_script_list,
        env_script_list=[],
        arch="x86_64",
        FAIL_TO_PASS=_load_str_list(instance.get("FAIL_TO_PASS", [])),
        PASS_TO_PASS=_load_str_list(instance.get("PASS_TO_PASS", [])),
        language=MAP_REPO_TO_EXT[repo],
        docker_specs=specs.get("docker_specs", {}),
        namespace=namespace,
    )


def detect_fakeroot(uri: str, mount_dir: Path) -> bool:
    """Probe whether ``apptainer exec --fakeroot`` works on this machine."""
    try:
        subprocess.run(
            [
                "apptainer", "exec", "--fakeroot",
                "--bind", f"{mount_dir}:{MOUNT_IN_CONTAINER}",
                uri, "/bin/true",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        print(f"  [INFO] --fakeroot works; running containers as root")
        return True
    except Exception:
        print(
            "  [WARN] --fakeroot unavailable; running containers as the current "
            "user (patch/tests may hit permission errors)"
        )
        return False


def apptainer_exec(
    uri: str,
    mount_dir: Path,
    command: str,
    use_fakeroot: bool,
    timeout: int,
) -> subprocess.CompletedProcess:
    """Run a command inside the instance image via apptainer exec."""
    argv = ["apptainer", "exec"]
    if use_fakeroot:
        argv.append("--fakeroot")
    argv += [
        # Let git operate on the root-owned repo without --fakeroot
        "--env", "GIT_CONFIG_COUNT=1",
        "--env", "GIT_CONFIG_KEY_0=safe.directory",
        "--env", "GIT_CONFIG_VALUE_0=/testbed",
        "--bind", f"{mount_dir}:{MOUNT_IN_CONTAINER}",
        uri, "/bin/bash", "-c", command,
    ]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def run_instance(
    spec: Any,
    pred: dict[str, str],
    run_id: str,
    registry: str,
    namespace: Optional[str],
    work_dir: Path,
    timeout: int,
    use_fakeroot: bool,
) -> dict[str, Any]:
    """Evaluate one instance with apptainer; mirrors swebench run_instance."""
    instance_id = spec.instance_id
    model_name = pred.get(KEY_MODEL, "None").replace("/", "__")
    log_dir = Path("logs/run_evaluation") / run_id / model_name / instance_id
    log_dir.mkdir(parents=True, exist_ok=True)
    report_path = log_dir / LOG_REPORT
    test_output_path = log_dir / LOG_TEST_OUTPUT

    uri = build_image_uri(spec.instance_image_key, registry)
    mount_dir = work_dir / run_id / instance_id
    mount_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "instance_id": instance_id,
        "model_name_or_path": pred.get(KEY_MODEL, "None"),
        "resolved": False,
        "status": "error",
    }

    # Empty model patch: nothing to evaluate (same as official empty_patch_ids)
    if pred.get(KEY_PREDICTION) in ("", None):
        result["status"] = "empty_patch"
        return result

    patch_path = mount_dir / "patch.diff"
    patch_path.write_text(pred[KEY_PREDICTION] or "", encoding="utf-8")

    print(f"  [..] {instance_id}: applying patch", flush=True)
    # 1. Apply the model patch inside the image (same fallback chain as harness)
    apply_cmd = f"cd {WORKDIR_IN_CONTAINER} && " + " || ".join(
        f"{c} {MOUNT_IN_CONTAINER}/patch.diff" for c in APP_APPLY_CMDS
    )
    try:
        apply_res = apptainer_exec(uri, mount_dir, apply_cmd, use_fakeroot, timeout)
    except subprocess.TimeoutExpired:
        result["error"] = f"patch apply timed out after {timeout}s"
        return result
    if apply_res.returncode != 0:
        result["error"] = f"patch apply failed:\n{apply_res.stdout}\n{apply_res.stderr}"
        report_path.write_text(
            json.dumps(
                {
                    instance_id: {
                        "patch_is_None": False,
                        "patch_exists": True,
                        "patch_successfully_applied": False,
                        "resolved": False,
                    }
                },
                indent=4,
            ),
            encoding="utf-8",
        )
        return result

    print(f"  [..] {instance_id}: running tests (this can take a while)", flush=True)
    # 2. Run the eval script (applies test patch + runs tests)
    eval_path = mount_dir / "eval.sh"
    eval_path.write_text(spec.eval_script, encoding="utf-8")
    try:
        eval_res = apptainer_exec(
            uri, mount_dir, f"/bin/bash {MOUNT_IN_CONTAINER}/eval.sh",
            use_fakeroot, timeout,
        )
    except subprocess.TimeoutExpired:
        test_output_path.write_text(
            f"\n\nTimeout error: {timeout} seconds exceeded.", encoding="utf-8"
        )
        result["error"] = f"test run timed out after {timeout}s"
        return result

    combined = (eval_res.stdout or "") + (eval_res.stderr or "")
    test_output_path.write_text(combined, encoding="utf-8")

    # 3. Grade using the official swebench report logic
    try:
        report = get_eval_report(
            test_spec=spec,
            prediction=pred,
            test_log_path=str(test_output_path),
            include_tests_status=True,
        )
        report_path.write_text(json.dumps(report, indent=4), encoding="utf-8")
        instance_report = report.get(instance_id, {})
        result["resolved"] = bool(instance_report.get("resolved", False))
        result["tests_status"] = instance_report.get("tests_status")
        result["status"] = "resolved" if result["resolved"] else "unresolved"
    except Exception as exc:
        result["error"] = f"grading failed: {exc}"
        return result
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the SWE-bench harness with Apptainer (no Docker)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--subset", default="verified",
                        help="verified|lite|full|swegym|swegym-lite|r2e-gym")
    parser.add_argument("--dataset-name", default=None,
                        help="Explicit HF dataset name (overrides --subset)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--preds", required=True, help="Path to preds.json/.jsonl")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate the first N instances (smoke testing)")
    parser.add_argument("--namespace", default="swebench",
                        help='Image namespace ("none" disables it)')
    parser.add_argument(
        "--registry", default=None,
        help="OCI registry mirror for docker:// pulls (default: $APPTAINER_REGISTRY "
             "or docker.m.daocloud.io; use '' for Docker Hub)",
    )
    parser.add_argument("--work-dir", default=None,
                        help="Temp dir for eval scripts/patches (default: $TMPDIR "
                             "or /data/dzz/harness_tmp)")
    parser.add_argument("--no-fakeroot", action="store_true",
                        help="Disable --fakeroot probing; run as current user")
    args = parser.parse_args()

    registry = args.registry
    if registry is None:
        registry = os.environ.get("APPTAINER_REGISTRY", "").strip().rstrip("/") or "docker.1panel.live"
    registry = registry.strip().rstrip("/")
    namespace = None if args.namespace.lower() == "none" else args.namespace
    work_dir = Path(args.work_dir or os.environ.get("TMPDIR") or "/data/dzz/harness_tmp")
    work_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = args.dataset_name or SUBSET_TO_DATASET.get(
        args.subset.lower(), args.subset
    )
    print(f"Loading {dataset_name} ({args.split})...")
    dataset = load_swebench_dataset(dataset_name, split=args.split)
    predictions = get_predictions_from_file(args.preds, dataset_name, args.split)
    pred_map = {p[KEY_INSTANCE_ID]: p for p in predictions}

    specs = [
        build_test_spec_offline(x, namespace=namespace)
        for x in dataset
        if x[KEY_INSTANCE_ID] in pred_map
    ]
    if args.limit is not None and args.limit > 0:
        specs = specs[: args.limit]
    print(f"Loaded {len(dataset)} instances, {len(pred_map)} predictions, "
          f"{len(specs)} to evaluate")
    if not specs:
        sys.exit("No matching instances to evaluate; aborting.")

    use_fakeroot = False if args.no_fakeroot else detect_fakeroot(
        build_image_uri(specs[0].instance_image_key, registry), work_dir
    )

    payloads = [
        (spec, pred_map[spec.instance_id]) for spec in specs
    ]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                run_instance, spec, pred, args.run_id, registry, namespace,
                work_dir, args.timeout, use_fakeroot,
            ): spec.instance_id
            for spec, pred in payloads
        }
        for future in as_completed(futures):
            instance_id = futures[future]
            try:
                results.append(future.result())
                print(f"  [done] {instance_id}")
            except Exception as exc:
                print(f"  [FAIL] {instance_id}: {exc}")
                results.append(
                    {"instance_id": instance_id, "resolved": False, "status": "error",
                     "error": str(exc)}
                )

    # 4. Aggregate into instance_results.jsonl + summary (compatible with
    #    merge_sft_pass_fail.py --instance-results / --results)
    eval_results_dir = Path("evaluation_results") / args.run_id
    eval_results_dir.mkdir(parents=True, exist_ok=True)
    results_path = eval_results_dir / "instance_results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_id = {r["instance_id"]: r for r in results}
    resolved_ids = sorted(i for i, r in by_id.items() if r["resolved"])
    unresolved_ids = sorted(
        i for i, r in by_id.items()
        if r["status"] == "unresolved" and not r["resolved"]
    )
    error_ids = sorted(i for i, r in by_id.items() if r["status"] == "error")
    empty_patch_ids = sorted(i for i, r in by_id.items() if r["status"] == "empty_patch")
    incomplete_ids = sorted(set(pred_map) - set(by_id))

    summary = {
        "total_instances": len(specs),
        "submitted_instances": len(pred_map),
        "completed_instances": len(resolved_ids) + len(unresolved_ids),
        "resolved_instances": len(resolved_ids),
        "unresolved_instances": len(unresolved_ids),
        "empty_patch_instances": len(empty_patch_ids),
        "error_instances": len(error_ids),
        "completed_ids": sorted(by_id),
        "incomplete_ids": incomplete_ids,
        "empty_patch_ids": empty_patch_ids,
        "submitted_ids": sorted(pred_map),
        "resolved_ids": resolved_ids,
        "unresolved_ids": unresolved_ids,
        "error_ids": error_ids,
        "schema_version": 2,
    }
    model_name = next(iter(pred_map.values()), {}).get(KEY_MODEL, "model")
    summary_path = Path(f"{model_name.replace('/', '__')}.{args.run_id}.json")
    summary_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")

    print(f"\nResolved: {len(resolved_ids)}  Unresolved: {len(unresolved_ids)}  "
          f"Errors: {len(error_ids)}  Empty patches: {len(empty_patch_ids)}")
    print(f"instance results: {results_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
