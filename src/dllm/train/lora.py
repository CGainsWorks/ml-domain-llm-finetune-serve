"""LoRA supervised fine-tuning, built to survive an interrupted rented GPU.

Resumability is the design constraint, not an afterthought:

* Checkpoints every ``train.save_steps`` optimiser steps (50 by default, which
  is a few minutes even on a slow card).
* ``resume: auto`` finds the newest checkpoint in the output directory and
  continues from it -- optimiser state, scheduler state, RNG state and dataset
  position included. Re-running the exact same command after a crash is the
  recovery procedure; there is nothing else to remember.
* The MLflow run id is persisted next to the checkpoints, so a resumed run
  appends to the original run instead of fragmenting the history into three
  half-runs that cannot be plotted together.
* ``save_total_limit`` keeps the disk from filling on a rented box with a small
  volume, which is a real way to lose a training run.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from dllm.config import PipelineConfig
from dllm.paths import DATA_PROCESSED, ROOT

log = logging.getLogger(__name__)

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
RUN_STATE_FILE = "dllm_run_state.json"


# --------------------------------------------------------------------------- #
# Resume support
# --------------------------------------------------------------------------- #
def find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Newest ``checkpoint-N`` directory in ``output_dir``, or None."""
    if not output_dir.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for child in output_dir.iterdir():
        m = CHECKPOINT_RE.match(child.name)
        if m and child.is_dir():
            candidates.append((int(m.group(1)), child))
    if not candidates:
        return None
    return max(candidates, key=lambda t: t[0])[1]


def _load_run_state(output_dir: Path) -> dict[str, Any]:
    path = output_dir / RUN_STATE_FILE
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("run state at %s is corrupt; starting a fresh MLflow run", path)
    return {}


def _save_run_state(output_dir: Path, state: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / RUN_STATE_FILE).write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def build_model_and_tokenizer(cfg: PipelineConfig) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from dllm.eval.backends.hf import resolve_dtype

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.id, revision=cfg.model.revision, trust_remote_code=cfg.model.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {
        "revision": cfg.model.revision,
        "dtype": resolve_dtype(cfg.model.dtype),
        "trust_remote_code": cfg.model.trust_remote_code,
        "attn_implementation": cfg.model.attn_implementation,
    }

    if cfg.quant.enabled:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=cfg.quant.bits == 4,
            load_in_8bit=cfg.quant.bits == 8,
            bnb_4bit_quant_type=cfg.quant.quant_type,
            bnb_4bit_use_double_quant=cfg.quant.double_quant,
            bnb_4bit_compute_dtype=resolve_dtype(cfg.quant.compute_dtype),
        )

    model = AutoModelForCausalLM.from_pretrained(cfg.model.id, **load_kwargs)

    if cfg.quant.enabled:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.train.gradient_checkpointing
        )

    if cfg.train.gradient_checkpointing:
        # Without this, gradient checkpointing silently produces no gradients
        # for LoRA params on a frozen base: the inputs never require grad, so
        # the checkpointed segment has nothing to backpropagate into.
        model.enable_input_require_grads()

    model.config.use_cache = False  # incompatible with gradient checkpointing
    if torch.cuda.is_available() and not cfg.quant.enabled:
        model.to("cuda")
    return model, tokenizer


def build_peft_config(cfg: PipelineConfig) -> Any:
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias=cfg.lora.bias,
        target_modules=cfg.lora.target_modules,
        modules_to_save=cfg.lora.modules_to_save or None,
        task_type=TaskType.CAUSAL_LM,
    )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(cfg: PipelineConfig, resume_from: str | None = None) -> dict[str, Any]:
    """Run LoRA SFT. Returns a summary dict. Safe to re-run to resume."""
    import mlflow
    from trl import SFTConfig, SFTTrainer

    from dllm.train.callbacks import GpuMemoryCallback, HeartbeatCallback, ThroughputCallback
    from dllm.train.dataset import build_datasets

    output_dir = ROOT / cfg.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- resume decision, made before anything expensive happens ------------ #
    resume_path: Path | None = None
    if resume_from:
        resume_path = Path(resume_from)
        if not resume_path.is_dir():
            raise FileNotFoundError(f"--resume points at {resume_path}, which does not exist")
    elif cfg.train.resume == "auto":
        resume_path = find_latest_checkpoint(output_dir)
        if resume_path:
            log.info("resuming from %s", resume_path)
        else:
            log.info("no checkpoint found in %s; starting from scratch", output_dir)

    run_state = _load_run_state(output_dir)

    # --- data --------------------------------------------------------------- #
    train_ds, eval_ds = build_datasets(
        DATA_PROCESSED,
        max_train=cfg.data.max_train_examples,
        max_eval=min(cfg.data.max_eval_examples or 200, 200),
    )

    model, tokenizer = build_model_and_tokenizer(cfg)
    peft_config = build_peft_config(cfg)

    effective_batch = cfg.train.per_device_batch_size * cfg.train.grad_accum_steps
    train_max_length = cfg.train.max_length or cfg.model.max_seq_len

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.train.epochs,
        max_steps=cfg.train.max_steps,
        learning_rate=cfg.train.lr,
        lr_scheduler_type=cfg.train.lr_scheduler,
        warmup_ratio=cfg.train.warmup_ratio,
        weight_decay=cfg.train.weight_decay,
        per_device_train_batch_size=cfg.train.per_device_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum_steps,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=cfg.train.max_grad_norm,
        optim=cfg.train.optim,
        seed=cfg.train.seed,
        data_seed=cfg.train.seed,
        bf16=cfg.train.bf16,
        fp16=cfg.train.fp16,
        max_length=train_max_length,
        packing=cfg.train.packing,
        packing_strategy=cfg.train.packing_strategy,
        completion_only_loss=cfg.train.completion_only_loss,
        logging_steps=cfg.train.logging_steps,
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        save_strategy="steps",
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=cfg.train.eval_steps,
        load_best_model_at_end=False,  # the eval gate selects, not the loss
        report_to=["mlflow"],
        run_name=cfg.train.run_name,
        dataloader_num_workers=0,  # Windows + HF datasets: workers deadlock
        remove_unused_columns=True,
    )

    heartbeat = HeartbeatCallback(output_dir / "heartbeat.json")
    vram = GpuMemoryCallback()
    throughput = ThroughputCallback(train_max_length, effective_batch)

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[heartbeat, vram, throughput],
    )

    # --- MLflow ------------------------------------------------------------- #
    mlflow.set_tracking_uri(cfg.train.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.train.mlflow_experiment)
    # transformers' MLflowCallback reuses an already-active run, so starting it
    # here keeps a resumed run attached to its original history.
    existing_run_id = run_state.get("mlflow_run_id")
    os.environ.setdefault("MLFLOW_FLATTEN_PARAMS", "1")

    active = mlflow.start_run(run_id=existing_run_id, run_name=cfg.train.run_name)
    run_id = active.info.run_id
    _save_run_state(
        output_dir,
        {
            "mlflow_run_id": run_id,
            "run_name": cfg.train.run_name,
            "config": json.loads(cfg.model_dump_json()),
        },
    )

    started = time.perf_counter()
    try:
        mlflow.log_params(
            {
                "base_model": cfg.model.id,
                "base_revision": cfg.model.revision,
                "lora_r": cfg.lora.r,
                "lora_alpha": cfg.lora.alpha,
                "lora_dropout": cfg.lora.dropout,
                "lora_targets": ",".join(cfg.lora.target_modules),
                "effective_batch": effective_batch,
                "max_seq_len": cfg.model.max_seq_len,
                "completion_only_loss": cfg.train.completion_only_loss,
                "quantised": cfg.quant.enabled,
                "train_examples": len(train_ds),
                "dataset_source": cfg.data.source,
            }
        )
        trainable, total = _count_params(trainer.model)
        mlflow.log_params(
            {
                "trainable_params": trainable,
                "total_params": total,
                "trainable_pct": round(100 * trainable / total, 4) if total else 0,
            }
        )
        log.info(
            "trainable %s / %s params (%.3f%%)",
            f"{trainable:,}",
            f"{total:,}",
            100 * trainable / total if total else 0,
        )

        result = trainer.train(resume_from_checkpoint=str(resume_path) if resume_path else None)

        adapter_dir = output_dir / "adapter"
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        elapsed = time.perf_counter() - started
        summary = {
            "run_id": run_id,
            "output_dir": str(output_dir),
            "adapter_dir": str(adapter_dir),
            "resumed_from": str(resume_path) if resume_path else None,
            "global_step": int(result.global_step),
            "train_loss": float(result.training_loss),
            "wall_clock_s": round(elapsed, 1),
            "gpu_hours": round(elapsed / 3600, 4),
            "peak_vram_allocated_gb": round(vram.peak_allocated_gb, 3),
            "peak_vram_reserved_gb": round(vram.peak_reserved_gb, 3),
            "samples_per_s": round(throughput.samples_per_s, 3),
            "effective_batch": effective_batch,
            "trainable_params": trainable,
        }
        mlflow.log_metrics(
            {
                "final_train_loss": summary["train_loss"],
                "wall_clock_s": summary["wall_clock_s"],
                "gpu_hours": summary["gpu_hours"],
                "peak_vram_allocated_gb": summary["peak_vram_allocated_gb"],
                "samples_per_s": summary["samples_per_s"],
            }
        )
        summary_path = output_dir / "train_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifacts(str(adapter_dir), artifact_path="adapter")
        return summary
    finally:
        mlflow.end_run()


def _count_params(model: Any) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
