"""
Fine-grained live difficulty loop for RL training.

While GRPO trains, every rollout immediately updates a per-instance difficulty
state with a cheap proxy signal (unified diff similarity vs. the oracle patch,
compared against a calibrated threshold). Periodically the state re-bins the
instances from the live pass-rate distribution, so the annealing curriculum
follows the *current* model instead of a stale offline measurement.

Optionally (``harness_dir``), pending rollouts are exported to a background
harness worker; real pass/fail results folded back overwrite the proxy counts.
"""

import json
import logging
import math
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from src.bin_difficulty import _merge_small_groups
from src.rewards.diff import unified_diff_similarity_reward_func

logger = logging.getLogger(__name__)


@dataclass
class LiveDifficultyConfig:
    enabled: bool = False
    pass_threshold: Optional[float] = None  # None = auto-calibrate from calibration_dataset
    calibration_dataset: Optional[str] = None  # labeled measurement dataset (with passed + diffs)
    num_bins: int = 5
    min_bin_size: float = 0.05
    rebin_every_steps: int = 200
    score_window: int = 50  # live pass rate uses the last N proxy scores per instance
    prior_decay_tau: Optional[float] = None  # None = auto (total_steps/4); >0 = exp(-step/tau); 0 = never decay
    harness_dir: Optional[str] = None  # optional async harness calibration (inbox/outbox subdirs)
    export_every_steps: int = 500


def default_diff_similarity(datum: dict[str, Any], result: dict[str, Any]) -> float:
    """Diff similarity between the oracle patch and the generated diff."""
    patch = datum.get("patch") or datum.get("oracle_diff") or ""
    generated = result.get("generated_diff") or ""
    return unified_diff_similarity_reward_func([patch], [generated])[0]


def calibrate_pass_threshold(
    dataset: Any,
    similarity_fn: Optional[Callable[[dict[str, Any]], float]] = None,
) -> float:
    """
    Choose the pseudo-pass threshold that maximizes F1 between the diff
    similarity proxy and the real harness labels in the measurement dataset.
    """
    sim = similarity_fn or default_diff_similarity
    scores: list[float] = []
    labels: list[bool] = []
    for row in dataset:
        try:
            scores.append(float(sim(row)))
            labels.append(bool(row["passed"]))
        except Exception as exc:  # unparseable diff etc.
            logger.warning(f"Skipping calibration row: {exc}")

    if not scores:
        logger.warning("No usable calibration rows; defaulting threshold to 0.5")
        return 0.5

    candidates = sorted({round(s, 4) for s in scores})
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in candidates:
        tp = sum(1 for s, l in zip(scores, labels) if s >= threshold and l)
        fp = sum(1 for s, l in zip(scores, labels) if s >= threshold and not l)
        fn = sum(1 for s, l in zip(scores, labels) if s < threshold and l)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best_f1:
            best_threshold, best_f1 = threshold, f1
    logger.info(
        f"Calibrated pseudo-pass threshold: {best_threshold:.3f} "
        f"(F1={best_f1:.3f} over {len(scores)} rollouts)"
    )
    return best_threshold


class LiveDifficultyState:
    """
    Thread-safe per-instance difficulty state combining an offline prior with
    live rollout evidence, with periodic relative re-binning.
    """

    def __init__(
        self,
        offline_map: dict[str, dict[str, Any]],
        config: LiveDifficultyConfig,
        total_steps: Optional[int] = None,
    ):
        self.config = config
        self.lock = threading.RLock()
        if config.prior_decay_tau is not None:
            self._tau = float(config.prior_decay_tau)
        elif total_steps and total_steps > 0:
            self._tau = total_steps / 4.0
        else:
            self._tau = 0.0
        self.offline: dict[str, list[int]] = {}
        for instance_id, info in offline_map.items():
            self.offline[str(instance_id)] = [
                int(info.get("n_passed", 0)),
                int(info.get("n_total", 0)),
            ]
        self.online: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        self.true_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        self.scores: dict[str, list[float]] = defaultdict(list)
        self.pending_best: dict[str, tuple[float, str]] = {}

        self.step = 0
        self.last_rebin_step = 0
        self.last_export_step = 0
        self.bins: dict[str, int] = {}
        self.bin_members: dict[int, list[str]] = {}
        self._present_bins: list[int] = []
        self.last_rebin_stats: dict[str, Any] = {}
        self.rebin()

    # ---- evidence recording ----

    def record(
        self,
        instance_id: str,
        passed: bool,
        score: Optional[float] = None,
        generated_diff: Optional[str] = None,
    ) -> None:
        with self.lock:
            iid = str(instance_id)
            self.online[iid][1] += 1
            if passed:
                self.online[iid][0] += 1
            if score is not None:
                self.scores[iid].append(float(score))
                if len(self.scores[iid]) > self.config.score_window:
                    self.scores[iid].pop(0)
                best = self.pending_best.get(iid)
                if best is None or score > best[0]:
                    self.pending_best[iid] = (float(score), generated_diff or "")

    def record_true(self, instance_id: str, passed: bool) -> None:
        """Fold a real harness result into the state (overrides proxy evidence)."""
        with self.lock:
            iid = str(instance_id)
            self.true_counts[iid][1] += 1
            if passed:
                self.true_counts[iid][0] += 1

    # ---- estimates and binning ----

    def _p_hat(self, instance_id: str) -> Optional[float]:
        n0p, n0t = self.offline.get(instance_id, [0, 0])
        lam = math.exp(-self.step / self._tau) if self._tau > 0 else 1.0
        if self.true_counts[instance_id][1] > 0:
            n_p, n_t = self.true_counts[instance_id]
        else:
            recent = self.scores.get(instance_id, [])
            if recent and self.config.pass_threshold is not None:
                # Live pass rate from the recent proxy-score window, so labels
                # track the *current* model instead of accumulating stale evidence.
                n_p = sum(1 for s in recent if s >= self.config.pass_threshold)
                n_t = len(recent)
            else:
                n_p, n_t = self.online.get(instance_id, [0, 0])
        denom = lam * n0t + n_t
        if denom <= 0:
            return None
        return (lam * n0p + n_p) / denom

    def rebin(self) -> None:
        """Recompute relative difficulty bins from the current live distribution."""
        with self.lock:
            old_bins = dict(self.bins)
            items = [
                (iid, p)
                for iid in set(self.offline) | set(self.online)
                if (p := self._p_hat(iid)) is not None
            ]
            items.sort(key=lambda x: -x[1])  # easiest first
            n = len(items)
            if n == 0:
                self.bins = {}
                self.bin_members = {}
                self._present_bins = []
                return

            k = min(self.config.num_bins, n)
            target = math.ceil(n / k)
            groups: list[list[tuple[str, float]]] = []
            idx = 0
            for _ in range(k - 1):
                groups.append(items[idx : idx + target])
                idx += target
            groups.append(items[idx:])
            min_size = (
                int(self.config.min_bin_size * n)
                if self.config.min_bin_size < 1
                else int(self.config.min_bin_size)
            )
            groups = _merge_small_groups(groups, min_size)

            new_bins: dict[str, int] = {}
            new_members: dict[int, list[str]] = defaultdict(list)
            for group_index, group in enumerate(groups):  # 0 = easiest
                bin_value = len(groups) - 1 - group_index  # 0 = hardest
                for iid, _ in group:
                    new_bins[iid] = bin_value
                    new_members[bin_value].append(iid)
            self.bins = new_bins
            self.bin_members = dict(new_members)
            self._present_bins = sorted(new_bins.values())

            drift = sum(
                1
                for iid, bin_value in new_bins.items()
                if iid in old_bins and old_bins[iid] != bin_value
            )
            became_easier = sum(
                1
                for iid, bin_value in new_bins.items()
                if iid in old_bins and old_bins[iid] < bin_value
            )
            became_harder = drift - became_easier
            self.last_rebin_stats = {
                "step": self.step,
                "num_bins": len(groups),
                "instances": n,
                "drift": drift,
                "became_easier": became_easier,
                "became_harder": became_harder,
                "bins": [
                    {
                        "bin": bin_value,
                        "size": len(members),
                        "avg_pass_rate": round(
                            sum(self._p_hat(iid) for iid in members) / len(members), 4
                        ),
                    }
                    for bin_value, members in sorted(new_members.items(), reverse=True)
                ],
            }
            logger.info(
                f"Live difficulty re-bin at step {self.step}: {len(groups)} bins, "
                f"{drift} changed ({became_easier} easier / {became_harder} harder) -> "
                f"{[(b['bin'], b['size']) for b in self.last_rebin_stats['bins']]}"
            )

    def bin(self, instance_id: str) -> int:
        with self.lock:
            return self.bins.get(str(instance_id), -1)

    def present_bins(self) -> list[int]:
        with self.lock:
            return list(self._present_bins)

    def bin_members_map(self) -> dict[int, list[str]]:
        with self.lock:
            return {k: list(v) for k, v in self.bin_members.items()}

    # ---- periodic maintenance (called by the curriculum callback) ----

    def periodic_update(self, step: int) -> None:
        with self.lock:
            self.step = int(step)
            if self.step - self.last_rebin_step >= self.config.rebin_every_steps:
                self.rebin()
                self.last_rebin_step = self.step

        if self.config.harness_dir:
            harness_dir = Path(self.config.harness_dir)
            inbox = harness_dir / "inbox"
            outbox = harness_dir / "outbox"
            if self.step - self.last_export_step >= self.config.export_every_steps:
                preds_path = inbox / f"preds_step{self.step:06d}.jsonl"
                exported = self.export_pending_preds(preds_path)
                if exported:
                    logger.info(
                        f"Exported {exported} pending rollouts to {preds_path} "
                        "for harness calibration"
                    )
                    self.last_export_step = self.step
            for result_file in sorted(outbox.glob("*.instance_results.jsonl")):
                folded = self.fold_harness_results(result_file)
                result_file.unlink(missing_ok=True)
                logger.info(f"Folded {folded} harness results from {result_file.name}")

    def export_pending_preds(self, preds_path: Path) -> int:
        """Export the best pending diff per instance for the harness worker."""
        with self.lock:
            if not self.pending_best:
                return 0
            preds_path.parent.mkdir(parents=True, exist_ok=True)
            with preds_path.open("w", encoding="utf-8") as f:
                for instance_id, (_, generated_diff) in sorted(self.pending_best.items()):
                    f.write(
                        json.dumps(
                            {
                                "instance_id": instance_id,
                                "model_name_or_path": "live-difficulty",
                                "model_patch": generated_diff,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            exported = len(self.pending_best)
            self.pending_best.clear()
            return exported

    def fold_harness_results(self, results_path: Path) -> int:
        """Fold real harness results (instance_results.jsonl) into true counts."""
        folded = 0
        with self.lock:
            with results_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    self.record_true(
                        str(record["instance_id"]),
                        bool(record.get("resolved", False)),
                    )
                    folded += 1
        return folded


def tracking_rollout_func(
    rollout_func: Callable,
    state: LiveDifficultyState,
    similarity_fn: Optional[Callable[[dict[str, Any], dict[str, Any]], float]] = None,
) -> Callable:
    """
    Wrap a GRPO rollout function so every completed rollout updates the live
    difficulty state with a proxy pseudo-pass before results are returned.
    """
    sim = similarity_fn or default_diff_similarity

    def wrapped(data, *args, **kwargs):
        results = rollout_func(data, *args, **kwargs)
        assert len(results) == len(data), (
            "rollout_func must preserve batch order to track live difficulty"
        )
        for datum, result in zip(data, results):
            score = sim(datum, result)
            passed = (
                score >= state.config.pass_threshold
                if state.config.pass_threshold is not None
                else False
            )
            state.record(
                str(datum["instance_id"]),
                passed=passed,
                score=score,
                generated_diff=result.get("generated_diff"),
            )
        return results

    return wrapped
