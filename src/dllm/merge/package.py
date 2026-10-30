"""Package a trained LoRA adapter for release: files, licence, manifest, archive.

What goes in is what someone needs to reproduce the served model from the
pinned base model and nothing else:

* the adapter weights and config (``base_model_name_or_path`` plus the pinned
  ``revision``, so ``PeftModel.from_pretrained`` fetches the exact base the
  scorecard was measured on);
* the chat template the model was trained with;
* the Apache-2.0 licence text and a NOTICE with every attribution the
  licence audit requires;
* the model card.

What stays out, deliberately:

* The tokenizer files. Training does not change the tokenizer, so the base
  model's own ``tokenizer.json`` at the pinned revision is the one to load;
  the manifest says so. A copy re-serialised by a newer ``transformers`` has
  the same vocabulary and encodings but different bytes, which would only
  invite the question of which one is authoritative.
* ``training_args.bin`` is a pickle. Loading it executes code, and it embeds
  the training machine's absolute paths.
* ``dllm_merge_info.json`` and ``train_summary.json`` also carry absolute
  paths. The useful numbers from the summary are copied into the manifest
  instead.
* PEFT's auto-generated README, which is an empty template.

The manifest records every file's size and SHA-256, the base model and its
licence, the data sources and theirs, and the scorecard the adapter produced.
The archive is built deterministically (sorted entries, fixed mtime and
owner), so packaging the same adapter twice gives the same archive hash.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

from dllm.config import PipelineConfig
from dllm.paths import ROOT

ADAPTER_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "chat_template.jinja",
)
OPTIONAL_FILES = frozenset({"chat_template.jinja"})

# train_summary.json fields worth carrying. Everything path-shaped is dropped.
SUMMARY_FIELDS = (
    "run_id",
    "global_step",
    "train_loss",
    "gpu_hours",
    "peak_vram_allocated_gb",
    "samples_per_s",
    "effective_batch",
    "trainable_params",
)

ADAPTER_LICENCE = "Apache-2.0"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _notice(cfg: PipelineConfig, name: str) -> str:
    m = cfg.model
    return "\n".join(
        [
            f"# {name}: NOTICE",
            "",
            f"LoRA adapter for {m.id} (revision {m.revision}),",
            f"base model licensed {m.licence}. The adapter is released under",
            f"{ADAPTER_LICENCE}; the full text is in LICENSE.",
            "",
            "Modifications: low-rank adapter weights trained on top of the base",
            "model. The base model weights themselves are not included.",
            "",
            "## Training data attribution",
            "",
            "- b-mc2/sql-create-context (CC-BY-4.0), b-mc2 (2023).",
            "- Built from WikiSQL (BSD-3-Clause), Salesforce.",
            "- Built from Spider (CC-BY-SA-4.0), Yale LILY. The training data",
            "  derives from Spider. The prepared dataset is not redistributed here.",
            "",
            "databricks/databricks-dolly-15k was used only as an evaluation guard",
            "and is not part of the training mix.",
            "",
        ]
    )


def _deterministic_tar(src: Path, out: Path) -> None:
    """A .tar.gz whose bytes depend only on the file contents and names."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(src.rglob("*")):
            if not path.is_file():
                continue
            info = tarfile.TarInfo(f"{src.name}/{path.relative_to(src).as_posix()}")
            info.size = path.stat().st_size
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as f:
                tar.addfile(info, f)
    with out.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(buf.getvalue())


def package_adapter(
    cfg: PipelineConfig,
    *,
    adapter_dir: Path,
    out_dir: Path,
    name: str,
    version: str,
    scorecard: Path | None = None,
    train_summary: Path | None = None,
) -> dict[str, Any]:
    """Write ``out_dir/<name>-<version>/``, its archive and manifest.

    Returns the manifest. Nothing is uploaded: publishing is a separate,
    deliberate step.
    """
    adapter_dir = Path(adapter_dir)
    stem = f"{name}-{version}"
    pkg = Path(out_dir) / stem
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True)

    for fname in ADAPTER_FILES:
        src = adapter_dir / fname
        if not src.is_file():
            if fname in OPTIONAL_FILES:
                continue
            raise FileNotFoundError(f"{src} is missing; is this a PEFT adapter directory?")
        if fname == "adapter_config.json":
            ac = json.loads(src.read_text(encoding="utf-8"))
            if ac.get("base_model_name_or_path") != cfg.model.id:
                raise ValueError(
                    f"adapter was trained on {ac.get('base_model_name_or_path')!r}, "
                    f"config says {cfg.model.id!r}"
                )
            # Pin the base so loading the adapter cannot silently pick up a
            # newer upload of the same model id.
            ac["revision"] = cfg.model.revision
            (pkg / fname).write_text(
                json.dumps(ac, indent=2) + "\n", encoding="utf-8", newline="\n"
            )
        else:
            shutil.copyfile(src, pkg / fname)

    shutil.copyfile(ROOT / "LICENSE", pkg / "LICENSE")
    (pkg / "NOTICE.md").write_text(_notice(cfg, stem), encoding="utf-8", newline="\n")
    card = ROOT / "docs" / "MODEL_CARD.md"
    if card.is_file():
        shutil.copyfile(card, pkg / "MODEL_CARD.md")

    files = [
        {"name": p.name, "size_bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in sorted(pkg.iterdir())
        if p.is_file()
    ]

    archive = Path(out_dir) / f"{stem}.tar.gz"
    _deterministic_tar(pkg, archive)

    manifest: dict[str, Any] = {
        "name": name,
        "version": version,
        "kind": "peft-lora-adapter",
        "licence": ADAPTER_LICENCE,
        "base_model": {
            "id": cfg.model.id,
            "revision": cfg.model.revision,
            "licence": cfg.model.licence,
            "licence_url": cfg.model.licence_url,
        },
        "tokenizer": {
            "source": "base model",
            "id": cfg.model.id,
            "revision": cfg.model.revision,
            "note": "Not shipped. Training does not change the tokenizer; load it from the base model at this revision.",
        },
        "lora": {
            "r": cfg.lora.r,
            "alpha": cfg.lora.alpha,
            "target_modules": sorted(cfg.lora.target_modules),
        },
        "training_data": [
            {"id": "b-mc2/sql-create-context", "licence": "CC-BY-4.0"},
            {"id": "WikiSQL (upstream)", "licence": "BSD-3-Clause"},
            {"id": "Spider (upstream)", "licence": "CC-BY-SA-4.0"},
        ],
        "files": files,
        "archive": {
            "name": archive.name,
            "size_bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
        },
    }

    if train_summary and Path(train_summary).is_file():
        summary = json.loads(Path(train_summary).read_text(encoding="utf-8"))
        manifest["training"] = {k: summary[k] for k in SUMMARY_FIELDS if k in summary}

    if scorecard and Path(scorecard).is_file():
        sc = json.loads(Path(scorecard).read_text(encoding="utf-8"))
        try:
            rel = Path(scorecard).resolve().relative_to(ROOT).as_posix()
        except ValueError:
            rel = Path(scorecard).name
        manifest["evaluation"] = {
            "scorecard": rel,
            "label": sc.get("label"),
            "metrics": {
                k: v
                for k, v in sc.get("metrics", {}).items()
                if k.startswith("sql_exec_holdout.") or k == "general_guard.answers_in_prose"
            },
        }

    (Path(out_dir) / f"{stem}.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (Path(out_dir) / "SHA256SUMS").write_text(
        f"{manifest['archive']['sha256']}  {archive.name}\n", encoding="utf-8", newline="\n"
    )
    return manifest
