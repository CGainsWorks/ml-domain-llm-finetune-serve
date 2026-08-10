"""Heartbeat progress reporting.

Worth testing because it is what someone reads to decide whether to keep paying
for a rented GPU, and it is written by a process they are not attached to.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("transformers", reason="heartbeat lives beside the Trainer API")

from dllm.train.callbacks import HeartbeatCallback, _human


class FakeState:
    def __init__(self, step: int, max_steps: int = 2500, epoch: float = 0.0) -> None:
        self.global_step = step
        self.max_steps = max_steps
        self.epoch = epoch
        self.best_metric = None


class TestHuman:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(0, "0s"), (45, "45s"), (90, "1m 30s"), (3600, "1h 00m"), (4284, "1h 11m")],
    )
    def test_formatting(self, seconds: float, expected: str) -> None:
        assert _human(seconds) == expected

    def test_negative_is_clamped(self) -> None:
        assert _human(-10) == "0s"


class TestHeartbeat:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        hb = HeartbeatCallback(tmp_path / "hb.json")
        hb._write(FakeState(10))
        data = json.loads((tmp_path / "hb.json").read_text(encoding="utf-8"))
        assert data["global_step"] == 10
        assert data["max_steps"] == 2500
        assert "eta_s" in data and "eta_human" in data

    def test_reports_both_rates(self, tmp_path: Path) -> None:
        hb = HeartbeatCallback(tmp_path / "hb.json")
        hb._write(FakeState(5))
        data = json.loads((tmp_path / "hb.json").read_text(encoding="utf-8"))
        assert "steps_per_s_recent" in data
        assert "steps_per_s_cumulative" in data

    def test_eta_uses_the_recent_rate_not_the_cumulative_one(self, tmp_path: Path) -> None:
        """A slow start must not poison the projection for the whole run.

        Simulates 20 s of startup followed by fast steps: the cumulative rate is
        badly pessimistic, the windowed rate is not.
        """
        hb = HeartbeatCallback(tmp_path / "hb.json")
        # 20 s of startup, then 100 steps in 10 s.
        hb._rates(done=0, elapsed=20.0, now=1020.0)
        cumulative, recent = hb._rates(done=100, elapsed=30.0, now=1030.0)
        assert cumulative == pytest.approx(100 / 30.0)  # 3.33 steps/s, pessimistic
        assert recent == pytest.approx(10.0)  # the rate it is actually running at
        assert recent > cumulative

    def test_first_sample_falls_back_to_cumulative(self, tmp_path: Path) -> None:
        hb = HeartbeatCallback(tmp_path / "hb.json")
        cumulative, recent = hb._rates(done=10, elapsed=5.0, now=1000.0)
        assert recent == cumulative

    def test_window_is_bounded(self, tmp_path: Path) -> None:
        hb = HeartbeatCallback(tmp_path / "hb.json")
        for i in range(HeartbeatCallback.WINDOW * 3):
            hb._rates(done=i, elapsed=float(i + 1), now=1000.0 + i)
        assert len(hb._history) <= HeartbeatCallback.WINDOW

    def test_write_is_atomic(self, tmp_path: Path) -> None:
        """A reader must never catch a half-written file."""
        target = tmp_path / "hb.json"
        hb = HeartbeatCallback(target)
        for step in range(5):
            hb._write(FakeState(step))
            json.loads(target.read_text(encoding="utf-8"))
        assert not (tmp_path / "hb.tmp").exists()

    def test_extra_fields_are_merged(self, tmp_path: Path) -> None:
        hb = HeartbeatCallback(tmp_path / "hb.json")
        hb._write(FakeState(1), {"event": "checkpoint_saved"})
        data = json.loads((tmp_path / "hb.json").read_text(encoding="utf-8"))
        assert data["event"] == "checkpoint_saved"

    def test_zero_steps_does_not_divide_by_zero(self, tmp_path: Path) -> None:
        hb = HeartbeatCallback(tmp_path / "hb.json")
        hb._write(FakeState(0))
        assert json.loads((tmp_path / "hb.json").read_text(encoding="utf-8"))["global_step"] == 0
