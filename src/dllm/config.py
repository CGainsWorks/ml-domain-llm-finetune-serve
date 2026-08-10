"""Typed configuration loaded from YAML.

Design notes
------------
* Every config file may declare ``_base_: <relative path>`` and is deep-merged
  onto its base. That keeps ``train/sql_lora_24gb.yaml`` down to the handful of
  fields that actually differ from the 8 GB profile.
* ``--set a.b=c`` overrides are parsed here rather than in each CLI so that
  training, eval and serving all behave the same way.
* Configs are strict (``extra="forbid"``). A typo in a YAML key is a startup
  error, not a silently ignored field that wastes a rented GPU hour.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dllm.paths import CONFIGS


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class ModelConfig(_Strict):
    """The base checkpoint being adapted."""

    id: str = Field(description="Hugging Face repo id of the base model.")
    revision: str = Field(
        default="main",
        description="Pinned commit/tag. 'main' is allowed but discouraged: an "
        "unpinned base model makes eval numbers unreproducible.",
    )
    licence: str = Field(description="SPDX id or licence name of the base weights.")
    licence_url: str = ""
    commercial_use: bool = Field(
        description="Whether the base model licence permits commercial use. "
        "Surfaced in the model card and checked by the licence audit."
    )
    dtype: Literal["bfloat16", "float16", "float32", "auto"] = "bfloat16"
    max_seq_len: int = 1024
    trust_remote_code: bool = False
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "sdpa"


# --------------------------------------------------------------------------- #
# LoRA
# --------------------------------------------------------------------------- #
class LoraConfig_(_Strict):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: Literal["none", "all", "lora_only"] = "none"
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    modules_to_save: list[str] = Field(default_factory=list)


class QuantConfig(_Strict):
    """4-bit base weights. Needed to fit a 1.5B+ base on an 8 GB card."""

    enabled: bool = False
    bits: Literal[4, 8] = 4
    quant_type: Literal["nf4", "fp4"] = "nf4"
    double_quant: bool = True
    compute_dtype: Literal["bfloat16", "float16"] = "bfloat16"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class DataConfig(_Strict):
    source: str = Field(description="Key into the source registry in dllm.data.sources.")
    max_train_examples: int | None = None
    max_eval_examples: int | None = None
    eval_fraction: float = 0.05
    test_fraction: float = 0.05
    seed: int = 1337
    max_prompt_chars: int = 6000
    dedupe: bool = True
    require_commercial_licence: bool = Field(
        default=False,
        description="If true, the licence audit fails on any non-commercial source.",
    )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
class TrainConfig(_Strict):
    run_name: str = "lora-sft"
    output_dir: str = "artifacts/checkpoints/{run_name}"
    epochs: float = 2.0
    max_steps: int = -1
    lr: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    per_device_batch_size: int = 2
    grad_accum_steps: int = 8
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    optim: str = "paged_adamw_8bit"
    seed: int = 1337
    bf16: bool = True
    fp16: bool = False
    packing: bool = False
    max_length: int | None = Field(
        default=None,
        description="Training truncation length. Defaults to model.max_seq_len. Kept "
        "separate because the two answer different questions: model.max_seq_len is how "
        "long a prompt may be at eval and serving time, while this is how long a "
        "training sequence is padded and truncated to. On this dataset the p95 example "
        "is ~150 tokens, so training at 1024 buys nothing and costs memory.",
    )
    packing_strategy: Literal["bfd", "wrapped"] = Field(
        default="bfd",
        description="Only used when packing is on. 'bfd' (best-fit-decreasing) bin-packs "
        "whole examples into sequences without splitting them across boundaries, which "
        "is what you want when the loss is masked to the completion. 'wrapped' "
        "concatenates and cuts at max_length, splitting examples. "
        "(transformers 5 removed group_by_length; BFD packing is the replacement for "
        "cutting padding waste.)",
    )
    completion_only_loss: bool = Field(
        default=True,
        description="Mask the prompt so loss is computed on the answer only. "
        "Materially better for short-answer domains like text-to-SQL.",
    )

    # Resumability -- the whole point of renting a GPU by the hour.
    save_steps: int = 50
    save_total_limit: int = 3
    logging_steps: int = 5
    eval_steps: int = 100
    resume: Literal["auto", "never"] = "auto"

    # Tracking
    mlflow_experiment: str = "domain-llm-sft"
    mlflow_tracking_uri: str = Field(
        default="sqlite:///mlflow.db",
        description="MLflow backend. SQLite rather than the './mlruns' file store for "
        "two reasons: MLflow 3 put the file store in maintenance mode and raises "
        "unless you opt out, and the model registry -- which the promotion gate "
        "writes to -- has never worked against a file store at all.",
    )
    log_grad_norm: bool = True


# --------------------------------------------------------------------------- #
# Eval
# --------------------------------------------------------------------------- #
class GenerationConfig(_Strict):
    max_new_tokens: int = 160
    temperature: float = 0.0
    top_p: float = 1.0
    stop: list[str] = Field(default_factory=lambda: ["\n\n", "```"])
    seed: int = 1337


class BackendConfig(_Strict):
    kind: Literal["hf", "openai"] = "hf"
    # hf backend
    device: str = "auto"
    batch_size: int = 8
    load_in_4bit: bool = False
    # openai / vLLM backend
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "DLLM_API_KEY"
    model: str = ""
    concurrency: int = 8
    timeout_s: float = 120.0


class ThresholdConfig(_Strict):
    """What "good enough to deploy" means, in numbers.

    ``min_absolute`` are hard floors. ``min_delta_vs_baseline`` are the
    improvements a fine-tune must show over the recorded base-model scorecard.
    ``max_regression`` caps how far a guard metric may fall -- this is what
    catches catastrophic forgetting.
    """

    min_absolute: dict[str, float] = Field(default_factory=dict)
    min_delta_vs_baseline: dict[str, float] = Field(default_factory=dict)
    max_regression: dict[str, float] = Field(default_factory=dict)


class EvalConfig(_Strict):
    tasks: list[str] = Field(description="Task ids resolved from evals/tasks/*.yaml.")
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    limit: int | None = Field(default=None, description="Cap items per task (smoke runs).")
    baseline_scorecard: str = "evals/results/baseline.json"
    output: str = "evals/results/{label}.json"


# --------------------------------------------------------------------------- #
# Serving
# --------------------------------------------------------------------------- #
class VllmConfig(_Strict):
    model: str = "artifacts/merged/current"
    served_model_name: str = "domain-sql"
    host: str = "0.0.0.0"
    port: int = 8000
    dtype: str = "bfloat16"
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.85
    max_num_seqs: int = 64
    tensor_parallel_size: int = 1
    enable_prefix_caching: bool = True
    extra_args: list[str] = Field(default_factory=list)


class GatewayConfig(_Strict):
    host: str = "0.0.0.0"
    port: int = 8080
    upstream_base_url: str = "http://localhost:8000/v1"
    request_timeout_s: float = 120.0
    rate_limit_rpm: int = 60
    rate_limit_burst: int = 10
    rate_limit_tokens_per_min: int = 60_000
    max_prompt_chars: int = 8000
    allowed_models: list[str] = Field(default_factory=lambda: ["domain-sql"])
    cors_origins: list[str] = Field(default_factory=list)


class ServeConfig(_Strict):
    vllm: VllmConfig = Field(default_factory=VllmConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)


# --------------------------------------------------------------------------- #
# Top level
# --------------------------------------------------------------------------- #
class PipelineConfig(_Strict):
    model: ModelConfig
    lora: LoraConfig_ = Field(default_factory=LoraConfig_)
    quant: QuantConfig = Field(default_factory=QuantConfig)
    data: DataConfig
    train: TrainConfig = Field(default_factory=TrainConfig)
    eval: EvalConfig
    serve: ServeConfig = Field(default_factory=ServeConfig)

    @model_validator(mode="after")
    def _check_precision(self) -> PipelineConfig:
        if self.train.bf16 and self.train.fp16:
            raise ValueError("train.bf16 and train.fp16 are mutually exclusive")
        return self

    def resolved_output_dir(self) -> Path:
        return Path(self.train.output_dir.format(run_name=self.train.run_name))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml_with_bases(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Read a YAML file, recursively resolving ``_base_`` includes."""
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = CONFIGS / path
    p = p.resolve()

    seen = _seen or set()
    if p in seen:
        raise ValueError(f"circular _base_ include at {p}")
    seen.add(p)

    if not p.is_file():
        raise FileNotFoundError(f"config not found: {p}")

    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{p} must contain a YAML mapping at the top level")

    bases = raw.pop("_base_", None)
    if bases is None:
        return raw

    merged: dict[str, Any] = {}
    for b in [bases] if isinstance(bases, str) else bases:
        merged = _deep_merge(merged, load_yaml_with_bases((p.parent / b), seen.copy()))
    return _deep_merge(merged, raw)


def _coerce(text: str) -> Any:
    """Turn a CLI override string into a Python value."""
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``a.b.c=value`` strings onto a nested dict."""
    out = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        key, _, value = item.partition("=")
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot descend into {key!r}: {part} is a scalar")
        node[parts[-1]] = _coerce(value)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> PipelineConfig:
    """Load, merge, override and validate a pipeline config."""
    raw = apply_overrides(load_yaml_with_bases(path), overrides)
    return PipelineConfig.model_validate(raw)
