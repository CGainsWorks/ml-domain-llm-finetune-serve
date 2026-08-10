"""Merge a LoRA adapter into the base weights and export a servable checkpoint.

vLLM can serve LoRA adapters directly, but a merged checkpoint is faster (no
per-token adapter matmul), simpler to version in a registry, and removes a
whole class of "which adapter was actually loaded" incidents. The cost is disk.

The verification step is the part worth keeping. Merging is a numerical
operation performed in a different dtype from training, and a silent precision
problem produces a model that is subtly worse in a way no exception reports.
After merging, the same prompts are run through both the adapter-on-base model
and the merged model and the outputs compared. A mismatch fails loudly here
rather than showing up as an unexplained drop in the eval scorecard.
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

VERIFY_PROMPTS = [
    ("List every player from Yorkshire.", "CREATE TABLE players (name VARCHAR, county VARCHAR)"),
    ("How many matches were played in 1994?", "CREATE TABLE matches (id VARCHAR, year VARCHAR)"),
    ("What is the highest score?", "CREATE TABLE scores (score INTEGER, team VARCHAR)"),
]


def _generate(model: Any, tokenizer: Any, max_new_tokens: int = 48) -> list[str]:
    import torch

    from dllm.prompts import sql_messages

    outs: list[str] = []
    device = next(model.parameters()).device
    for question, context in VERIFY_PROMPTS:
        messages = sql_messages(question, context)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        outs.append(
            tokenizer.decode(gen[0, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        )
    return outs


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

    dtype = resolve_dtype(cfg.model.dtype)
    log.info("loading base %s (%s) in %s", cfg.model.id, cfg.model.revision[:8], dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.id, revision=cfg.model.revision, trust_remote_code=cfg.model.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        cfg.model.id,
        revision=cfg.model.revision,
        dtype=dtype,
        trust_remote_code=cfg.model.trust_remote_code,
    )

    peft_model = PeftModel.from_pretrained(base, str(adapter), dtype=dtype)
    peft_model.eval()

    before = _generate(peft_model, tokenizer) if verify else []

    log.info("merging adapter into base weights")
    merged = peft_model.merge_and_unload()
    merged.eval()
    merged.config.use_cache = True

    after = _generate(merged, tokenizer) if verify else []

    mismatches = [
        {"prompt": VERIFY_PROMPTS[i][0], "adapter": b, "merged": a}
        for i, (b, a) in enumerate(zip(before, after, strict=True))
        if b != a
    ]
    if verify and mismatches:
        raise RuntimeError(
            "merged model does not reproduce adapter-on-base outputs; refusing to export.\n"
            + json.dumps(mismatches, indent=2)
        )

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
        "dtype": str(dtype),
        "verified": verify,
        "verify_prompts": len(VERIFY_PROMPTS),
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
