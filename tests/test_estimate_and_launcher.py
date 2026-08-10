"""GPU-hours estimation and the vLLM launch command.

Both are pure functions over config, so they are cheap to test and they are the
two places where a silent mistake costs real money: a bad estimate leads to
renting the wrong card, a bad launch command wastes the first ten minutes of the
hour.
"""

from __future__ import annotations

import pytest

from dllm.config import load_config
from dllm.paths import CONFIGS
from dllm.serve.vllm_launcher import build_command
from dllm.train.estimate import GpuSpec, estimate, load_gpus, training_length

BASE = load_config(CONFIGS / "base.yaml")

FAKE_GPUS = [
    GpuSpec("Slow", vram_gb=8, bf16_tflops_dense=30, mfu=0.2, usd_per_hour=0.0),
    GpuSpec("Fast", vram_gb=80, bf16_tflops_dense=300, mfu=0.2, usd_per_hour=2.0),
]


class TestGpuTable:
    def test_shipped_table_parses(self) -> None:
        gpus = load_gpus()
        assert gpus
        for g in gpus:
            assert g.bf16_tflops_dense > 0
            assert 0 < g.mfu <= 1, "mfu is a fraction, not a percentage"
            assert g.vram_gb > 0

    def test_effective_throughput_is_derated(self) -> None:
        g = GpuSpec("x", 24, 100, 0.25, 1.0)
        assert g.effective_tflops == 25


class TestTrainingLength:
    def test_prefers_the_training_override(self) -> None:
        cfg = load_config(CONFIGS / "base.yaml", ["train.max_length=256"])
        assert training_length(cfg) == 256

    def test_falls_back_to_model_length(self) -> None:
        cfg = load_config(CONFIGS / "base.yaml", ["train.max_length=null"])
        assert training_length(cfg) == cfg.model.max_seq_len


class TestEstimate:
    def test_basic_shape(self) -> None:
        est = estimate(BASE, gpus=FAKE_GPUS)
        assert est["param_count"] > 0
        assert est["total_tokens"] > 0
        assert len(est["estimates"]) == 2

    def test_faster_gpu_is_faster(self) -> None:
        est = estimate(BASE, gpus=FAKE_GPUS)
        rows = {r["gpu"]: r["hours"] for r in est["estimates"]}
        assert rows["Fast"] < rows["Slow"]

    def test_ten_times_the_flops_is_a_tenth_of_the_time(self) -> None:
        est = estimate(BASE, gpus=FAKE_GPUS)
        rows = {r["gpu"]: r["hours"] for r in est["estimates"]}
        # Reported hours are rounded to 2 d.p., which is a few percent on
        # sub-hour values, so the tolerance is on the ratio rather than exact.
        assert rows["Slow"] / rows["Fast"] == pytest.approx(10.0, rel=0.05)

    def test_gradient_checkpointing_costs_50_percent_more(self) -> None:
        """6*N*T with checkpointing against 4*N*T without."""
        on = estimate(BASE, gpus=FAKE_GPUS)
        off = estimate(
            load_config(CONFIGS / "base.yaml", ["train.gradient_checkpointing=false"]),
            gpus=FAKE_GPUS,
        )
        assert on["flops_per_param_token"] == 6.0
        assert off["flops_per_param_token"] == 4.0
        ratio = on["total_petaflops"] / off["total_petaflops"]
        assert ratio == pytest.approx(1.5, rel=0.01)

    def test_more_epochs_costs_proportionally_more(self) -> None:
        one = estimate(load_config(CONFIGS / "base.yaml", ["train.epochs=1"]), gpus=FAKE_GPUS)
        two = estimate(load_config(CONFIGS / "base.yaml", ["train.epochs=2"]), gpus=FAKE_GPUS)
        assert two["total_tokens"] == pytest.approx(2 * one["total_tokens"], rel=0.01)

    def test_max_steps_overrides_epochs(self) -> None:
        est = estimate(load_config(CONFIGS / "base.yaml", ["train.max_steps=100"]), gpus=FAKE_GPUS)
        assert est["total_samples"] == 100 * est["effective_batch"]

    def test_calibration_replaces_the_model(self) -> None:
        est = estimate(BASE, gpus=FAKE_GPUS, measured_samples_per_s=10.0)
        assert est["calibrated"]["hours"] == pytest.approx(
            est["total_samples"] / 10.0 / 3600, rel=0.01
        )

    def test_memory_verdict_flags_a_model_that_will_not_fit(self) -> None:
        tiny = [GpuSpec("Tiny", vram_gb=4, bf16_tflops_dense=30, mfu=0.2, usd_per_hour=0.0)]
        est = estimate(BASE, gpus=tiny)
        assert est["estimates"][0]["fits"].startswith("no")

    def test_cost_is_hours_times_rate(self) -> None:
        est = estimate(BASE, gpus=FAKE_GPUS)
        for row in est["estimates"]:
            assert row["usd"] == pytest.approx(row["hours"] * row["usd_per_hour"], abs=0.02)


class TestVllmCommand:
    def test_includes_the_essentials(self) -> None:
        cmd = build_command(BASE.serve.vllm)
        joined = " ".join(cmd)
        assert "vllm.entrypoints.openai.api_server" in joined
        assert "--served-model-name" in cmd
        assert BASE.serve.vllm.served_model_name in cmd
        assert "--max-model-len" in cmd
        assert str(BASE.serve.vllm.max_model_len) in cmd

    def test_prefix_caching_toggles(self) -> None:
        cfg = load_config(CONFIGS / "base.yaml", ["serve.vllm.enable_prefix_caching=false"])
        assert "--enable-prefix-caching" not in build_command(cfg.serve.vllm)
        assert "--enable-prefix-caching" in build_command(BASE.serve.vllm)

    def test_extra_args_are_appended(self) -> None:
        cfg = load_config(CONFIGS / "base.yaml", ['serve.vllm.extra_args=["--seed","7"]'])
        cmd = build_command(cfg.serve.vllm)
        assert cmd[-2:] == ["--seed", "7"]

    def test_hub_id_is_passed_through_unchanged(self) -> None:
        cfg = load_config(CONFIGS / "base.yaml", ["serve.vllm.model=Qwen/Qwen2.5-1.5B-Instruct"])
        assert "Qwen/Qwen2.5-1.5B-Instruct" in build_command(cfg.serve.vllm)
