"""Training callbacks: progress that survives a disconnect, and VRAM telemetry.

Both exist because the target scenario is a GPU rented by the hour over SSH.
A dropped connection must not cost you the knowledge of where the run got to,
and an OOM at step 900 of 1200 is much cheaper to diagnose if peak memory was
being recorded all along.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

log = logging.getLogger(__name__)


class HeartbeatCallback(TrainerCallback):
    """Write run progress to a JSON file after every logging step.

    ``tail -f`` on this file tells you steps done, throughput and projected
    finish time without attaching to the training process.
    """

    def __init__(self, path: Path, total_steps_hint: int = 0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.total_hint = total_steps_hint

    def _write(self, state: TrainerState, extra: dict | None = None) -> None:
        elapsed = time.time() - self.started
        total = state.max_steps or self.total_hint
        done = state.global_step
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = (total - done) / rate if rate > 0 and total else None
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "global_step": done,
            "max_steps": total,
            "epoch": round(state.epoch or 0.0, 4),
            "elapsed_s": round(elapsed, 1),
            "steps_per_s": round(rate, 4),
            "eta_s": round(remaining, 1) if remaining is not None else None,
            "best_metric": state.best_metric,
            **(extra or {}),
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)  # atomic: a reader never sees a half-written file

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs: object,
    ) -> None:
        self._write(state, {"last_log": logs or {}})

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        self._write(state, {"event": "checkpoint_saved"})

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        self._write(state, {"event": "train_end"})


class GpuMemoryCallback(TrainerCallback):
    """Record peak allocated and reserved VRAM, and log it once per logging step."""

    def __init__(self) -> None:
        self.peak_allocated_gb = 0.0
        self.peak_reserved_gb = 0.0

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs: object,
    ) -> None:
        try:
            import torch

            if not torch.cuda.is_available():
                return
            allocated = torch.cuda.max_memory_allocated() / 1024**3
            reserved = torch.cuda.max_memory_reserved() / 1024**3
        except Exception:  # noqa: BLE001
            return
        self.peak_allocated_gb = max(self.peak_allocated_gb, allocated)
        self.peak_reserved_gb = max(self.peak_reserved_gb, reserved)
        if logs is not None:
            logs["vram_allocated_gb"] = round(allocated, 3)
            logs["vram_reserved_gb"] = round(reserved, 3)


class ThroughputCallback(TrainerCallback):
    """Measure samples/second and tokens/second.

    These are the numbers that turn into the GPU-hours estimate in
    ``docs/GPU_HOURS.md``, so they are measured rather than assumed.
    """

    def __init__(self, seq_len: int, effective_batch: int) -> None:
        self.seq_len = seq_len
        self.effective_batch = effective_batch
        self._last_step = 0
        self._last_time = time.time()
        self.samples_per_s = 0.0
        self.tokens_per_s = 0.0

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs: object,
    ) -> None:
        now = time.time()
        steps = state.global_step - self._last_step
        dt = now - self._last_time
        if steps <= 0 or dt <= 0:
            return
        self.samples_per_s = steps * self.effective_batch / dt
        self.tokens_per_s = self.samples_per_s * self.seq_len
        self._last_step = state.global_step
        self._last_time = now
        if logs is not None:
            logs["samples_per_s"] = round(self.samples_per_s, 3)
            logs["tokens_per_s_upper_bound"] = round(self.tokens_per_s, 1)
