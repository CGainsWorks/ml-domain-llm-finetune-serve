"""Estimate GPU hours and rental cost before committing money to a run.

The arithmetic
--------------
A transformer forward pass costs about ``2 * N * T`` FLOPs for ``N`` parameters
and ``T`` tokens. The backward pass normally costs ``4 * N * T``: half computing
gradients with respect to the activations, half with respect to the weights.

LoRA freezes the base weights, so the weight-gradient half nearly vanishes --
the adapter matrices are a fraction of a percent of the parameters. That puts
LoRA training at roughly ``4 * N * T`` rather than ``6 * N * T``.

Gradient checkpointing recomputes the forward pass during backward, adding
``2 * N * T`` back and returning the total to about ``6 * N * T``. It is not
free; it buys memory with roughly 30% more compute.

Two things stop this from being a fantasy:

* **Token count is measured, not assumed.** The estimator tokenises a sample of
  the real prepared dataset with the real tokenizer. Assuming every example
  fills ``max_seq_len`` overestimates by 3-5x on text-to-SQL, where the median
  example is a couple of hundred tokens.
* **`--calibrate` beats theory.** Model FLOPs utilisation on a small LoRA run
  is genuinely hard to predict. Run 20 steps, measure seconds per step,
  extrapolate. The analytic number is there to tell you which card to rent; the
  calibrated one is there to tell you when to come back.
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dllm.config import PipelineConfig
from dllm.paths import CONFIGS, DATA_PROCESSED

log = logging.getLogger(__name__)

# Rough parameter counts, used only when the model cannot be inspected offline.
KNOWN_PARAM_COUNTS = {
    "Qwen/Qwen2.5-0.5B-Instruct": 494_000_000,
    "Qwen/Qwen2.5-1.5B-Instruct": 1_540_000_000,
    "Qwen/Qwen2.5-3B-Instruct": 3_090_000_000,
    "Qwen/Qwen2.5-7B-Instruct": 7_620_000_000,
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": 1_540_000_000,
    "Qwen/Qwen2.5-Coder-7B-Instruct": 7_620_000_000,
    "meta-llama/Llama-3.2-1B-Instruct": 1_240_000_000,
    "meta-llama/Llama-3.2-3B-Instruct": 3_210_000_000,
    "meta-llama/Llama-3.1-8B-Instruct": 8_030_000_000,
}


@dataclass
class GpuSpec:
    name: str
    vram_gb: float
    bf16_tflops_dense: float
    mfu: float
    usd_per_hour: float
    note: str = ""

    @property
    def effective_tflops(self) -> float:
        return self.bf16_tflops_dense * self.mfu


def load_gpus(path: Path | None = None) -> list[GpuSpec]:
    path = path or CONFIGS / "gpus.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [GpuSpec(**g) for g in raw["gpus"]]


# --------------------------------------------------------------------------- #
# Token accounting
# --------------------------------------------------------------------------- #
def training_length(cfg: PipelineConfig) -> int:
    """The length training actually pads and truncates to."""
    return cfg.train.max_length or cfg.model.max_seq_len


def measure_token_stats(cfg: PipelineConfig, sample: int = 800) -> dict[str, float]:
    """Tokenise a sample of the prepared training set with the real tokenizer.

    Falls back to a conservative guess if the tokenizer or data is unavailable,
    and says so in the returned dict rather than silently inventing a number.
    """
    train_path = DATA_PROCESSED / "train.jsonl"
    if not train_path.is_file():
        return {
            "mean_tokens": float(training_length(cfg)) * 0.4,
            "median_tokens": float(training_length(cfg)) * 0.4,
            "p95_tokens": float(training_length(cfg)),
            "sampled": 0,
            "measured": 0.0,
        }

    from dllm.data.prepare import read_jsonl
    from dllm.prompts import sql_messages, sql_target

    rows = read_jsonl(train_path, sample)
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            cfg.model.id, revision=cfg.model.revision, trust_remote_code=cfg.model.trust_remote_code
        )
    except Exception as e:  # noqa: BLE001
        log.warning("could not load tokenizer (%s); estimating from characters", e)
        lengths = [(len(r["question"]) + len(r["context"]) + len(r["answer"])) / 3.6 for r in rows]
        return _summarise(lengths, training_length(cfg), sampled=len(rows), measured=0.0)

    lengths: list[float] = []
    for r in rows:
        messages = sql_messages(r["question"], r["context"])
        text = (
            tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if getattr(tok, "chat_template", None)
            else "\n".join(m["content"] for m in messages)
        )
        text += sql_target(r["answer"])
        lengths.append(len(tok(text, add_special_tokens=False)["input_ids"]))

    return _summarise(lengths, training_length(cfg), sampled=len(rows), measured=1.0)


def _summarise(
    lengths: list[float], cap: int, *, sampled: int, measured: float
) -> dict[str, float]:
    if not lengths:
        return {
            "mean_tokens": float(cap) * 0.4,
            "median_tokens": float(cap) * 0.4,
            "p95_tokens": float(cap),
            "sampled": 0,
            "measured": 0.0,
        }
    capped = [min(x, cap) for x in lengths]
    ordered = sorted(capped)
    return {
        "mean_tokens": round(statistics.fmean(capped), 1),
        "median_tokens": round(statistics.median(capped), 1),
        "p95_tokens": round(ordered[int(0.95 * (len(ordered) - 1))], 1),
        "truncated_pct": round(100 * sum(x >= cap for x in lengths) / len(lengths), 2),
        "sampled": sampled,
        "measured": measured,
    }


# --------------------------------------------------------------------------- #
# The estimate
# --------------------------------------------------------------------------- #
def param_count(cfg: PipelineConfig) -> tuple[int, str]:
    """Parameter count for the base model, and how we worked it out."""
    if cfg.model.id in KNOWN_PARAM_COUNTS:
        return KNOWN_PARAM_COUNTS[cfg.model.id], "table"
    try:
        from transformers import AutoConfig

        hf = AutoConfig.from_pretrained(
            cfg.model.id, revision=cfg.model.revision, trust_remote_code=cfg.model.trust_remote_code
        )
        h = hf.hidden_size
        layers = hf.num_hidden_layers
        inter = getattr(hf, "intermediate_size", 4 * h)
        vocab = hf.vocab_size
        # attention (q,k,v,o) + MLP (gate,up,down) + embeddings, tied or not
        per_layer = 4 * h * h + 3 * h * inter
        total = layers * per_layer + 2 * vocab * h
        return int(total), "derived from config"
    except Exception as e:  # noqa: BLE001
        log.warning("could not determine parameter count (%s); assuming 1.5B", e)
        return 1_540_000_000, "fallback guess"


def estimate(
    cfg: PipelineConfig,
    *,
    gpus: list[GpuSpec] | None = None,
    measured_samples_per_s: float | None = None,
) -> dict[str, Any]:
    """Compute the training-time estimate for every known GPU."""
    gpus = gpus or load_gpus()
    n_params, param_source = param_count(cfg)
    tok_stats = measure_token_stats(cfg)

    n_examples = cfg.data.max_train_examples or 20_000
    epochs = cfg.train.epochs
    if cfg.train.max_steps and cfg.train.max_steps > 0:
        effective_batch = cfg.train.per_device_batch_size * cfg.train.grad_accum_steps
        total_samples = cfg.train.max_steps * effective_batch
    else:
        total_samples = int(n_examples * epochs)

    tokens = total_samples * tok_stats["mean_tokens"]

    # 4*N*T for LoRA; +2*N*T when gradient checkpointing recomputes the forward.
    flops_per_token_param = 6.0 if cfg.train.gradient_checkpointing else 4.0
    total_flops = flops_per_token_param * n_params * tokens

    rows: list[dict[str, Any]] = []
    for gpu in gpus:
        seconds = total_flops / (gpu.effective_tflops * 1e12)
        hours = seconds / 3600
        rows.append(
            {
                "gpu": gpu.name,
                "vram_gb": gpu.vram_gb,
                "effective_tflops": round(gpu.effective_tflops, 1),
                "hours": round(hours, 2),
                "usd_per_hour": gpu.usd_per_hour,
                "usd": round(hours * gpu.usd_per_hour, 2),
                "fits": _fits(cfg, n_params, gpu),
                "note": gpu.note,
            }
        )

    out: dict[str, Any] = {
        "model": cfg.model.id,
        "param_count": n_params,
        "param_source": param_source,
        "train_examples": n_examples,
        "epochs": epochs,
        "total_samples": total_samples,
        "token_stats": tok_stats,
        "total_tokens": int(tokens),
        "flops_per_param_token": flops_per_token_param,
        "gradient_checkpointing": cfg.train.gradient_checkpointing,
        "total_petaflops": round(total_flops / 1e15, 2),
        "effective_batch": cfg.train.per_device_batch_size * cfg.train.grad_accum_steps,
        "optimiser_steps": int(
            total_samples / (cfg.train.per_device_batch_size * cfg.train.grad_accum_steps)
        ),
        "estimates": rows,
    }

    if measured_samples_per_s:
        hours = total_samples / measured_samples_per_s / 3600
        out["calibrated"] = {
            "samples_per_s": measured_samples_per_s,
            "hours": round(hours, 2),
            "source": "measured on this machine",
        }
    return out


def _fits(cfg: PipelineConfig, n_params: int, gpu: GpuSpec) -> str:
    """Rough memory verdict. Deliberately pessimistic."""
    bytes_per_param = 0.55 if cfg.quant.enabled else 2.0
    weights = n_params * bytes_per_param / 1024**3
    activations = (
        0.75
        * cfg.train.per_device_batch_size
        * training_length(cfg)
        / 1024
        * (0.4 if cfg.train.gradient_checkpointing else 1.6)
    )
    overhead = 1.2
    needed = weights + activations + overhead
    if needed < gpu.vram_gb * 0.85:
        return f"yes (~{needed:.1f} GB)"
    if needed < gpu.vram_gb:
        return f"tight (~{needed:.1f} GB)"
    return f"no (~{needed:.1f} GB)"


def read_measured_throughput(cfg: PipelineConfig) -> float | None:
    """Pick up ``samples_per_s`` from a previous run in this output directory."""
    from dllm.paths import ROOT

    summary = ROOT / cfg.resolved_output_dir() / "train_summary.json"
    if summary.is_file():
        try:
            return float(json.loads(summary.read_text(encoding="utf-8"))["samples_per_s"])
        except Exception:  # noqa: BLE001
            return None
    heartbeat = ROOT / cfg.resolved_output_dir() / "heartbeat.json"
    if heartbeat.is_file():
        try:
            data = json.loads(heartbeat.read_text(encoding="utf-8"))
            return float(data["last_log"]["samples_per_s"])
        except Exception:  # noqa: BLE001
            return None
    return None


def render_estimate(cfg: PipelineConfig, only_gpu: str | None = None) -> str:
    """Human-readable estimate table."""
    est = estimate(cfg, measured_samples_per_s=read_measured_throughput(cfg))
    ts = est["token_stats"]

    lines = [
        f"Training-time estimate for {est['model']}",
        "=" * 60,
        f"  parameters          {est['param_count']:,} ({est['param_source']})",
        f"  train examples      {est['train_examples']:,} x {est['epochs']} epochs "
        f"= {est['total_samples']:,} samples",
        f"  optimiser steps     {est['optimiser_steps']:,} "
        f"(effective batch {est['effective_batch']})",
        f"  tokens/example      mean {ts['mean_tokens']:.0f}, median {ts['median_tokens']:.0f}, "
        f"p95 {ts['p95_tokens']:.0f}, truncated {ts.get('truncated_pct', 0):.1f}%"
        + ("  [measured]" if ts.get("measured") else "  [ESTIMATED, run data prepare first]"),
        f"  total tokens        {est['total_tokens']:,}",
        f"  compute             {est['total_petaflops']:.1f} PFLOPs "
        f"({est['flops_per_param_token']:.0f}*N*T"
        f"{', gradient checkpointing on' if est['gradient_checkpointing'] else ''})",
        "",
        f"  {'GPU':<16} {'VRAM':>5} {'eff TF':>7} {'hours':>7} {'$/h':>6} {'est $':>7}  fits",
        "  " + "-" * 72,
    ]
    for row in est["estimates"]:
        if only_gpu and only_gpu.lower() not in row["gpu"].lower():
            continue
        lines.append(
            f"  {row['gpu']:<16} {row['vram_gb']:>4.0f}G {row['effective_tflops']:>7.1f} "
            f"{row['hours']:>7.2f} {row['usd_per_hour']:>6.2f} {row['usd']:>7.2f}  {row['fits']}"
        )

    lines += [
        "",
        "  Add ~15-25 min of wall clock on top for model download, dataset prep,",
        "  the baseline eval and the post-training eval. These are not GPU-bound",
        "  but you are paying for the hour either way.",
    ]

    # The FLOPs model assumes the GPU is compute-bound. At short sequence
    # lengths it is not: per-step fixed costs dominate and the real time can be
    # several times the table. Saying so is more useful than a confident number.
    tokens_per_step = est["effective_batch"] * ts["mean_tokens"]
    if tokens_per_step < 8192:
        lines += [
            "",
            f"  CAVEAT: only ~{tokens_per_step:,.0f} tokens per optimiser step. Below roughly",
            "  8k, per-step fixed costs (kernel launches, optimiser, dataloader) matter as",
            "  much as the matmuls, so every row above is only as good as its assumed",
            "  utilisation figure -- and it can be wrong in EITHER direction. Measured on",
            "  an RTX 4060 the reference config hit 0.35 MFU, more than twice the 0.16",
            "  originally assumed, so the estimate was 2.2x pessimistic on that card.",
            "  The faster cards are the least trustworthy rows, because fixed costs are a",
            "  larger share of their step time. Raise train.per_device_batch_size or turn",
            "  on train.packing to close the gap. Calibrate before trusting any of it.",
        ]

    if "calibrated" in est:
        c = est["calibrated"]
        lines += [
            "",
            f"  CALIBRATED: {c['samples_per_s']:.2f} samples/s measured here "
            f"-> {c['hours']:.2f} h for this configuration.",
            "  Trust this over the table above; the table assumes a utilisation figure.",
        ]
    else:
        lines += [
            "",
            "  No measured throughput yet. For a real number, run ~20 steps first:",
            "    dllm train run -c <config> --set train.max_steps=20",
            "    dllm train estimate -c <config>",
        ]
    return "\n".join(lines)
