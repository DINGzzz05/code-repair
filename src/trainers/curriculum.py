import logging
import math
import random
from dataclasses import dataclass
from typing import Any, Optional

import torch
from trl import GRPOTrainer
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


@dataclass
class CurriculumConfig:
    """
    Annealing curriculum over per-instance difficulty bins.

    Difficulty bins come from the pass-rate measurement (``difficulty_bin`` ==
    the number of passing rollouts out of N measured per instance). Training
    starts at the easiest bin (highest pass count, default 8) and anneals to
    the hardest bin (lowest pass count, default 0).
    """

    enabled: bool = False
    start_bin: Optional[int] = None  # easiest bin; None = highest bin present
    end_bin: Optional[int] = None  # hardest bin; None = lowest bin present
    window: int = 1  # radius of active bins around the annealing center
    tau: float = 1.0  # softmax sharpness; smaller = sharper transition
    easy_mix_floor: float = 0.15  # minimum share of easiest-bin data per batch (anti-forgetting)
    seed: int = 42


class CurriculumState:
    """Shared progress tracker between the callback and the lazy dataset."""

    def __init__(self, total_steps: int, batch_size: int):
        self.total_steps = max(1, total_steps)
        self.batch_size = max(1, batch_size)
        self.callback_step = 0
        self.fetched_items = 0

    def set_step(self, step: int) -> None:
        self.callback_step = max(0, int(step))

    def on_fetch(self, count: int) -> None:
        self.fetched_items += count

    def progress(self) -> float:
        """Training progress in [0, 1] from the callback step (fallback: item counter)."""
        step = max(self.callback_step, self.fetched_items // self.batch_size)
        return min(1.0, step / self.total_steps)


class CurriculumDataset(torch.utils.data.Dataset):
    """
    Lazily samples instances from difficulty bins according to the annealing
    schedule, so the curriculum progresses even if the trainer only builds its
    dataloader once. Bins are static (read from the ``difficulty_bin`` column)
    unless a ``LiveDifficultyState`` is provided, in which case bins are
    resolved live and re-binned periodically.
    """

    def __init__(
        self,
        source: Any,
        config: CurriculumConfig,
        state: CurriculumState,
        live_state: Optional[Any] = None,
    ):
        self.config = config
        self.state = state
        self.live_state = live_state
        self.rows = list(source)
        self.rows_by_id = {str(row["instance_id"]): row for row in self.rows}
        if self.live_state is None:
            self.bins: dict[int, list[int]] = {}
            for index, row in enumerate(self.rows):
                self.bins.setdefault(int(row["difficulty_bin"]), []).append(index)
            if not self.bins:
                raise ValueError(
                    "Curriculum dataset has no difficulty_bin values. "
                    "Run the difficulty measurement first."
                )
            self.present_bins = sorted(self.bins)
            self.start_bin, self.end_bin = self._resolve_start_end(self.present_bins)
        else:
            self.bins = {}
            self.present_bins = self.live_state.present_bins()
            if not self.present_bins:
                raise ValueError(
                    "Live difficulty state has no bins yet; enable measurement labels."
                )
            self.start_bin, self.end_bin = self._resolve_start_end(self.present_bins)
        self.rng = random.Random(self.config.seed)
        logger.info(
            f"CurriculumDataset ready: {len(self.rows)} instances, "
            f"bins {self.present_bins}, annealing {self.start_bin} -> {self.end_bin}"
            + (" (live difficulty)" if self.live_state is not None else "")
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_start_end(self, present_bins: list[int]) -> tuple[int, int]:
        start_bin = (
            self.config.start_bin
            if self.config.start_bin is not None
            else present_bins[-1]
        )
        end_bin = (
            self.config.end_bin
            if self.config.end_bin is not None
            else present_bins[0]
        )
        start_bin = min(start_bin, present_bins[-1])
        end_bin = max(end_bin, present_bins[0])
        if start_bin < end_bin:
            start_bin, end_bin = end_bin, start_bin
        return start_bin, end_bin

    def _bin_weights(self, progress: float) -> tuple[list[float], list[int], int, int]:
        present_bins = (
            self.live_state.present_bins() if self.live_state is not None else self.present_bins
        )
        if not present_bins:
            raise RuntimeError("No difficulty bins available for curriculum sampling")
        start_bin, end_bin = self._resolve_start_end(present_bins)
        center = start_bin + (end_bin - start_bin) * progress
        weights = [
            math.exp(-abs(bin_value - center) / max(1e-9, self.config.tau))
            if abs(bin_value - center) <= self.config.window
            else 0.0
            for bin_value in present_bins
        ]
        total = sum(weights)
        if total <= 0:
            # Sparse bins: fall back to uniform over the present bins.
            weights = [1.0] * len(present_bins)
            total = float(len(present_bins))
        weights = [w / total for w in weights]

        # Anti-forgetting floor: keep a minimum share of the easiest bin.
        easiest_idx = present_bins.index(start_bin)
        if weights[easiest_idx] < self.config.easy_mix_floor:
            weights[easiest_idx] = self.config.easy_mix_floor
            others = sum(weights) - self.config.easy_mix_floor
            if others > 0:
                scale = (1.0 - self.config.easy_mix_floor) / others
                weights = [
                    self.config.easy_mix_floor if i == easiest_idx else w * scale
                    for i, w in enumerate(weights)
                ]
        return weights, present_bins, start_bin, end_bin

    def __getitem__(self, index: int) -> dict[str, Any]:
        self.state.on_fetch(1)
        progress = self.state.progress()
        weights, present_bins, _, _ = self._bin_weights(progress)
        bin_value = self.rng.choices(present_bins, weights=weights, k=1)[0]
        if self.live_state is not None:
            members = self.live_state.bin_members_map().get(bin_value, [])
            if not members:
                # Fallback: any instance currently in the dataset.
                members = list(self.rows_by_id)
            instance_id = self.rng.choice(members)
            return self.rows_by_id[instance_id]
        row_index = self.rng.choice(self.bins[bin_value])
        return self.rows[row_index]


class CurriculumCallback(TrainerCallback):
    def __init__(
        self,
        state: CurriculumState,
        live_state: Optional[Any] = None,
    ):
        self.state = state
        self.live_state = live_state

    def on_step_begin(self, args, state, control):
        self.state.set_step(state.global_step)
        if self.live_state is not None:
            self.live_state.periodic_update(state.global_step)


class CurriculumGRPOTrainer(GRPOTrainer):
    """
    GRPO trainer that anneals instance difficulty from easy (high pass rate) to
    hard (low pass rate) during a single run. Wraps the training dataset in a
    lazy curriculum sampler and keeps the annealing center in sync with the
    global training step.
    """

    def __init__(
        self,
        *args,
        curriculum: Optional[CurriculumConfig] = None,
        live_state: Optional[Any] = None,
        **kwargs,
    ):
        self.curriculum = curriculum or CurriculumConfig()
        self.live_state = live_state
        super().__init__(*args, **kwargs)
        if not self.curriculum.enabled:
            if self.live_state is not None:
                logger.warning(
                    "live_state provided but curriculum disabled; live difficulty "
                    "will not be consumed"
                )
            return

        base = self.train_dataset
        column_names = getattr(base, "column_names", None) or []
        if self.live_state is None and "difficulty_bin" not in column_names:
            raise ValueError(
                "Curriculum training requires a 'difficulty_bin' column in the "
                "dataset. Set run.difficulty='curriculum' and pass "
                "run.difficulty_path pointing at the difficulty.jsonl."
            )

        total_steps = self._estimate_total_steps(base)
        state = CurriculumState(
            total_steps=total_steps,
            batch_size=getattr(self.args, "generation_batch_size", 1),
        )
        self.train_dataset = CurriculumDataset(
            base, self.curriculum, state, live_state=self.live_state
        )
        self.add_callback(CurriculumCallback(state, live_state=self.live_state))
        logger.info(
            f"Curriculum annealing enabled: bins {self.train_dataset.present_bins}, "
            f"{self.train_dataset.start_bin} -> {self.train_dataset.end_bin}, "
            f"total_steps={total_steps}"
            + (" (live difficulty)" if self.live_state is not None else "")
        )

    def _estimate_total_steps(self, base) -> int:
        if self.args.max_steps and self.args.max_steps > 0:
            return int(self.args.max_steps)
        per_epoch = math.ceil(len(base) / max(1, getattr(self.args, "generation_batch_size", 1)))
        return per_epoch * max(1, self.args.num_train_epochs)
