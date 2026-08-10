"""Config loading, _base_ merging, overrides and strictness."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from dllm.config import (
    PipelineConfig,
    apply_overrides,
    load_config,
    load_yaml_with_bases,
)
from dllm.paths import CONFIGS

SHIPPED = ["base.yaml", "smoke.yaml", "local_4060_8gb.yaml", "rented_a100.yaml"]


class TestShippedConfigs:
    @pytest.mark.parametrize("name", SHIPPED)
    def test_every_shipped_config_validates(self, name: str) -> None:
        """A broken config must fail here, not after the GPU meter has started."""
        cfg = load_config(CONFIGS / name)
        assert cfg.model.id
        assert cfg.eval.tasks

    @pytest.mark.parametrize("name", SHIPPED)
    def test_referenced_tasks_exist(self, name: str) -> None:
        from dllm.eval.tasks import load_spec

        cfg = load_config(CONFIGS / name)
        for task_id in cfg.eval.tasks:
            assert load_spec(task_id).id == task_id

    def test_base_model_revision_is_pinned(self) -> None:
        cfg = load_config(CONFIGS / "base.yaml")
        assert cfg.model.revision != "main", "an unpinned base makes evals unreproducible"
        assert len(cfg.model.revision) == 40

    def test_gate_thresholds_are_configured(self) -> None:
        cfg = load_config(CONFIGS / "base.yaml")
        t = cfg.eval.thresholds
        assert t.min_absolute and t.min_delta_vs_baseline and t.max_regression


class TestBaseMerging:
    def test_child_overrides_parent(self, tmp_path: Path) -> None:
        (tmp_path / "parent.yaml").write_text(
            yaml.safe_dump({"a": 1, "nested": {"x": 1, "y": 2}}), encoding="utf-8"
        )
        (tmp_path / "child.yaml").write_text(
            yaml.safe_dump({"_base_": "parent.yaml", "nested": {"y": 99}}), encoding="utf-8"
        )
        merged = load_yaml_with_bases(tmp_path / "child.yaml")
        assert merged == {"a": 1, "nested": {"x": 1, "y": 99}}

    def test_circular_include_is_caught(self, tmp_path: Path) -> None:
        (tmp_path / "a.yaml").write_text(yaml.safe_dump({"_base_": "b.yaml"}), encoding="utf-8")
        (tmp_path / "b.yaml").write_text(yaml.safe_dump({"_base_": "a.yaml"}), encoding="utf-8")
        with pytest.raises(ValueError, match="circular"):
            load_yaml_with_bases(tmp_path / "a.yaml")

    def test_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_yaml_with_bases("does_not_exist.yaml")


class TestOverrides:
    def test_nested_scalar(self) -> None:
        out = apply_overrides({"train": {"epochs": 2}}, ["train.epochs=5"])
        assert out["train"]["epochs"] == 5

    def test_type_coercion(self) -> None:
        out = apply_overrides({}, ["a.b=true", "a.c=1.5", "a.d=null", "a.e=text"])
        assert out["a"] == {"b": True, "c": 1.5, "d": None, "e": "text"}

    def test_json_value(self) -> None:
        out = apply_overrides({}, ['a=["x","y"]'])
        assert out["a"] == ["x", "y"]

    def test_malformed_override_rejected(self) -> None:
        with pytest.raises(ValueError, match=re.escape("key.path=value")):
            apply_overrides({}, ["no_equals_sign"])

    def test_override_applies_through_load_config(self) -> None:
        cfg = load_config(CONFIGS / "base.yaml", ["train.epochs=7", "lora.r=64"])
        assert cfg.train.epochs == 7
        assert cfg.lora.r == 64


class TestStrictness:
    def test_unknown_key_is_rejected(self) -> None:
        """A typo'd YAML key must not be silently ignored on a rented GPU."""
        cfg = load_yaml_with_bases(CONFIGS / "base.yaml")
        cfg["train"]["epocs"] = 3  # typo
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(cfg)

    def test_bf16_and_fp16_are_mutually_exclusive(self) -> None:
        cfg = load_yaml_with_bases(CONFIGS / "base.yaml")
        cfg["train"]["fp16"] = True
        with pytest.raises(ValidationError, match="mutually exclusive"):
            PipelineConfig.model_validate(cfg)


class TestOutputPaths:
    def test_run_name_is_interpolated(self) -> None:
        cfg = load_config(CONFIGS / "base.yaml", ["train.run_name=my-run"])
        assert cfg.resolved_output_dir().name == "my-run"
