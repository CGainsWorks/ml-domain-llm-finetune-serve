"""Merge a LoRA adapter into the base weights and export a servable checkpoint.

vLLM can serve LoRA adapters directly, but a merged checkpoint is faster (no
per-token adapter matmul), simpler to version in a registry, and removes a
whole class of "which adapter was actually loaded" incidents. The cost is disk.

The verification step is the part worth keeping. Merging is a numerical
operation and a silent precision problem produces a model that is subtly worse
in a way no exception reports. After merging, the same prompts are pushed
through both the adapter-on-base model and the merged model and the results
compared. A real mismatch fails loudly here rather than showing up as an
unexplained drop in the eval scorecard.

**Comparing decoded text does not work, and the first version of this did.**
Greedy decoding is a chaotic function of the logits: one near-tie resolved
differently by a bf16 rounding difference of ~1e-3 changes a token, and every
token after it diverges. On the very first real merge this rejected a perfectly
good checkpoint because `MAX(score) ... WHERE team = "team"` became
`MAX(score) ... GROUP BY team` on an ambiguous two-column prompt. Both are
reasonable answers to the question; neither indicates a broken merge.

So verification compares **logits under teacher forcing** instead. Every
position of every probe prompt contributes a next-token argmax, giving hundreds
of comparisons rather than a handful of chaotic ones, plus a maximum absolute
logit difference. That measures the property actually wanted -- "merging did not
change the model" -- rather than a property that merely correlates with it.

The merge itself is done in float32 and cast down afterwards. Accumulating
``W + BA * (alpha/r)`` directly in bf16 rounds every element to ~3 significant
digits; doing it in fp32 costs some host RAM (this runs on CPU) and removes the
avoidable half of the error.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dllm.config import PipelineConfig
from dllm.paths import ROOT

log = logging.getLogger(__name__)

# Gate: fraction of probe positions where the fp32 merged model and the fp32
# adapter-on-base model must pick the same next token. Both do the same
# arithmetic in the same precision, so this should be essentially 1.0; anything
# below means merging genuinely altered the model.
MIN_MERGE_AGREEMENT = 0.999

# Warning only: how much the fp32 -> export-dtype cast is allowed to move
# next-token choices before it is worth saying out loud. Not a gate -- reduced
# precision changing some near-ties is what reduced precision does.
WARN_EXPORT_AGREEMENT = 0.95

VERIFY_PROMPTS = [
    ("List every player from Yorkshire.", "CREATE TABLE players (name VARCHAR, county VARCHAR)"),
    ("How many matches were played in 1994?", "CREATE TABLE matches (id VARCHAR, year VARCHAR)"),
    ("What is the highest score?", "CREATE TABLE scores (score INTEGER, team VARCHAR)"),
]


def _probe_logits(model: Any, tokenizer: Any) -> list[Any]:
    """Teacher-forced logits for each probe prompt. No sampling, no decoding."""
    import torch

    from dllm.prompts import sql_messages

    out = []
    device = next(model.parameters()).device
    for question, context in VERIFY_PROMPTS:
        messages = sql_messages(question, context)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.inference_mode():
            out.append(model(**enc).logits.float().cpu())
    return out


def compare_logits(before: list[Any], after: list[Any]) -> dict[str, float]:
    """Agreement between two models' logits over every probe position.

    ``argmax_agreement`` is the fraction of positions where both models would
    pick the same next token. That is the number to gate on: it aggregates
    hundreds of decisions instead of amplifying one.
    """

    total = 0
    agree = 0
    max_abs = 0.0
    for a, b in zip(before, after, strict=True):
        max_abs = max(max_abs, float((a - b).abs().max()))
        pa, pb = a.argmax(dim=-1), b.argmax(dim=-1)
        agree += int((pa == pb).sum())
        total += int(pa.numel())
    return {
        "positions_compared": total,
        "argmax_agreement": round(agree / total, 6) if total else 0.0,
        "max_abs_logit_diff": round(max_abs, 6),
    }


def merge_and_export(
    cfg: PipelineConfig,
    *,
    adapter_path: str,
    out_dir: str = "artifacts/merged/current",
    verify: bool = True,
    safe_serialization: bool = True,
) -> dict[str, Any]:
    """Merge ``adapter_path`` into the base model and write a full checkpoint."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from dllm.eval.backends.hf import resolve_dtype

    adapter = Path(adapter_path)
    if not adapter.is_dir():
        raise FileNotFoundError(f"adapter directory not found: {adapter}")

    target = ROOT / out_dir if not Path(out_dir).is_absolute() else Path(out_dir)
    if target.exists():
        log.warning("overwriting existing export at %s", target)
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    target_dtype = resolve_dtype(cfg.model.dtype)
    # Merge in fp32 and cast afterwards: W + BA*(alpha/r) accumulated directly
    # in bf16 rounds every element to ~3 significant digits for no reason. This
    # runs on CPU, so the cost is host RAM rather than VRAM.
    merge_dtype = torch.float32
    log.info(
        "loading base %s (%s): merging in %s, exporting %s",
        cfg.model.id,
        cfg.model.revision[:8],
        merge_dtype,
        target_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.id, revision=cfg.model.revision, trust_remote_code=cfg.model.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        cfg.model.id,
        revision=cfg.model.revision,
        dtype=merge_dtype,
        trust_remote_code=cfg.model.trust_remote_code,
    )

    peft_model = PeftModel.from_pretrained(base, str(adapter), dtype=merge_dtype)
    peft_model.eval()

    before = _probe_logits(peft_model, tokenizer) if verify else []

    log.info("merging adapter into base weights")
    merged = peft_model.merge_and_unload()
    merged.eval()
    merged.config.use_cache = True

    agreement: dict[str, Any] = {}
    if verify:
        # Two separate questions, and conflating them makes both unanswerable:
        #
        #   1. Did merging change the model?  fp32 merged vs fp32 adapter-on-base.
        #      Same arithmetic either way, so this should be near-perfect. This
        #      is what the export is gated on.
        #   2. What does exporting at a lower precision cost?  bf16 merged vs
        #      the fp32 reference. Casting a 1.5B model to ~3 significant digits
        #      genuinely moves logits, so this is reported, not gated.
        #
        # Measuring (1) against a bf16 model reports (1)+(2) together and then
        # blames the merge for the cast.
        merge_check = compare_logits(before, _probe_logits(merged, tokenizer))
        log.info(
            "merge fidelity (fp32 vs fp32): argmax agreement %.6f over %d positions, "
            "max |dlogit| %.6f",
            merge_check["argmax_agreement"],
            merge_check["positions_compared"],
            merge_check["max_abs_logit_diff"],
        )

        merged = merged.to(target_dtype)
        export_check = compare_logits(before, _probe_logits(merged, tokenizer))
        log.info(
            "export precision (%s vs fp32): argmax agreement %.6f, max |dlogit| %.6f",
            target_dtype,
            export_check["argmax_agreement"],
            export_check["max_abs_logit_diff"],
        )

        agreement = {"merge_fp32": merge_check, "export_cast": export_check}

        if merge_check["argmax_agreement"] < MIN_MERGE_AGREEMENT:
            raise RuntimeError(
                "merging changed the model; refusing to export. Required argmax "
                f"agreement >= {MIN_MERGE_AGREEMENT} in fp32, got "
                f"{merge_check['argmax_agreement']}.\n" + json.dumps(agreement, indent=2)
            )
        if export_check["argmax_agreement"] < WARN_EXPORT_AGREEMENT:
            log.warning(
                "casting to %s moved %.1f%% of next-token choices. The merge is sound, "
                "but the exported checkpoint will not score identically to the adapter. "
                "Export in float32 if that matters.",
                target_dtype,
                100 * (1 - export_check["argmax_agreement"]),
            )
    else:
        merged = merged.to(target_dtype)
    merged.save_pretrained(str(target), safe_serialization=safe_serialization)
    tokenizer.save_pretrained(str(target))

    adapter_cfg_path = adapter / "adapter_config.json"
    adapter_cfg = (
        json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
        if adapter_cfg_path.is_file()
        else {}
    )

    total_bytes = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    info = {
        "merged_dir": str(target),
        "adapter_dir": str(adapter),
        "base_model": cfg.model.id,
        "base_revision": cfg.model.revision,
        "dtype": str(target_dtype),
        "merge_dtype": str(merge_dtype),
        "verified": verify,
        "verify_prompts": len(VERIFY_PROMPTS),
        "verification": agreement,
        "lora_r": adapter_cfg.get("r"),
        "lora_alpha": adapter_cfg.get("lora_alpha"),
        "target_modules": adapter_cfg.get("target_modules"),
        "size_bytes": total_bytes,
        "size_gb": round(total_bytes / 1024**3, 3),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch": torch.__version__,
    }
    (target / "dllm_merge_info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    log.info("merged checkpoint written to %s (%.2f GB)", target, info["size_gb"])
    return info
