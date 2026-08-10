"""Download, filter, split and write the domain dataset.

Order of operations is deliberate:

1. **Licence audit first.** Nothing is downloaded until the audit passes. It is
   much easier to argue you never trained on a non-commercial dataset if you
   never fetched it.
2. **Parse-check every row.** An example whose gold SQL does not parse teaches
   the model nothing useful and pollutes the eval denominator.
3. **Execution-check the eval splits only.** Materialising a database per row
   costs about a millisecond, which is fine for a few thousand eval rows and
   wasteful for tens of thousands of training rows. The eval splits are the
   ones whose denominator has to be clean.
4. **Hash-based splitting, not shuffling.** An item's split is a function of its
   content hash, so it stays in the same split when the dataset size changes or
   when a filter is added. Re-running prep with a larger cap cannot leak a
   previous training item into the test set.
5. **Deduplicate before splitting.** Otherwise a duplicated question lands in
   train *and* test and inflates the score.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dllm.config import DataConfig, ModelConfig
from dllm.data.licence_audit import audit, write_audit
from dllm.data.sources import get_source
from dllm.eval.sql.execute import Outcome, evaluate_prediction
from dllm.eval.sql.normalise import SqlParseFailure, is_read_only, parse
from dllm.paths import DATA_PROCESSED, DOCS

log = logging.getLogger(__name__)

SPLITS = ("train", "validation", "test")


@dataclass
class PrepStats:
    source: str
    raw: int = 0
    dropped_empty: int = 0
    dropped_unparseable: int = 0
    dropped_not_select: int = 0
    dropped_too_long: int = 0
    dropped_duplicate: int = 0
    dropped_gold_unexecutable: int = 0
    kept: int = 0
    counts: dict[str, int] | None = None

    def as_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["counts"] = self.counts or {}
        return d


def _hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def assign_split(item_id: str, *, seed: int, eval_fraction: float, test_fraction: float) -> str:
    """Deterministically place an item in train / validation / test.

    Uses the item's own hash rather than a shuffled index so that the
    assignment is stable across dataset revisions and filter changes.
    """
    bucket = int(_hash(str(seed), item_id)[:8], 16) / 0xFFFFFFFF
    if bucket < test_fraction:
        return "test"
    if bucket < test_fraction + eval_fraction:
        return "validation"
    return "train"


# --------------------------------------------------------------------------- #
# Row-level cleaning
# --------------------------------------------------------------------------- #
def clean_sql_rows(rows: Iterable[dict], cfg: DataConfig, stats: PrepStats) -> Iterator[dict]:
    seen: set[str] = set()
    for row in rows:
        stats.raw += 1
        question = (row.get("question") or "").strip()
        context = (row.get("context") or "").strip()
        answer = (row.get("answer") or "").strip().rstrip(";").strip()

        if not question or not context or not answer:
            stats.dropped_empty += 1
            continue

        if len(question) + len(context) > cfg.max_prompt_chars:
            stats.dropped_too_long += 1
            continue

        try:
            expr = parse(answer)
        except SqlParseFailure:
            stats.dropped_unparseable += 1
            continue

        if not is_read_only(expr):
            stats.dropped_not_select += 1
            continue

        if cfg.dedupe:
            key = _hash(question.lower(), context.lower())
            if key in seen:
                stats.dropped_duplicate += 1
                continue
            seen.add(key)

        yield {
            "id": _hash(question, context, answer)[:16],
            "question": question,
            "context": context,
            "answer": answer,
            "source": cfg.source,
        }


def gold_is_executable(item: dict) -> bool:
    """Whether the gold query runs against a materialised version of its schema."""
    result = evaluate_prediction(
        context=item["context"], gold_sql=item["answer"], pred_sql=item["answer"]
    )
    return result.outcome is not Outcome.GOLD_UNEXECUTABLE


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    out: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit is not None and len(out) >= limit:
                break
    return out


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def prepare(
    data_cfg: DataConfig,
    model_cfg: ModelConfig | None = None,
    *,
    out_dir: Path | None = None,
    include_guard: bool = True,
    verify_eval_gold: bool = True,
) -> dict[str, Any]:
    """Run the full preparation, returning a stats dict. Writes JSONL splits."""
    out_dir = out_dir or DATA_PROCESSED
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Licence audit, before anything is fetched ----------------------- #
    source_keys = [data_cfg.source] + (["dolly_guard"] if include_guard else [])
    manifest = audit(
        source_keys,
        require_commercial=data_cfg.require_commercial_licence,
        base_model_licence=model_cfg.licence if model_cfg else None,
        base_model_commercial=model_cfg.commercial_use if model_cfg else None,
    )
    write_audit(manifest, out_dir / "licence_audit.json", DOCS / "LICENCE_AUDIT.md")
    if not manifest["passed"]:
        errs = [
            f"{f['source']}: {f['message']}" for f in manifest["findings"] if f["level"] == "error"
        ]
        raise SystemExit("Licence audit failed, refusing to download:\n  - " + "\n  - ".join(errs))
    for f in manifest["findings"]:
        log.warning("licence audit %s: %s -- %s", f["level"], f["source"], f["message"])

    # --- 2. Load ------------------------------------------------------------ #
    from datasets import load_dataset  # imported late: heavy, and unused by the gate

    src = get_source(data_cfg.source)
    log.info("loading %s (%s, %s)", src.hf_id, src.split, src.licence)
    ds = load_dataset(src.hf_id, split=src.split, revision=src.revision)

    stats = PrepStats(source=src.hf_id)
    cleaned = list(clean_sql_rows(ds, data_cfg, stats))
    log.info("cleaned %d -> %d rows", stats.raw, len(cleaned))

    # --- 3. Split ----------------------------------------------------------- #
    buckets: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for item in cleaned:
        buckets[
            assign_split(
                item["id"],
                seed=data_cfg.seed,
                eval_fraction=data_cfg.eval_fraction,
                test_fraction=data_cfg.test_fraction,
            )
        ].append(item)

    # --- 4. Verify gold executability on the eval splits -------------------- #
    if verify_eval_gold:
        for split in ("validation", "test"):
            before = len(buckets[split])
            buckets[split] = [i for i in buckets[split] if gold_is_executable(i)]
            dropped = before - len(buckets[split])
            stats.dropped_gold_unexecutable += dropped
            log.info("%s: dropped %d rows whose gold query does not execute", split, dropped)

    # --- 5. Cap ------------------------------------------------------------- #
    # Sort by id first so a cap takes a stable, reproducible subset rather than
    # whatever order the loader happened to yield.
    for split in SPLITS:
        buckets[split].sort(key=lambda r: r["id"])
    if data_cfg.max_train_examples:
        buckets["train"] = buckets["train"][: data_cfg.max_train_examples]
    if data_cfg.max_eval_examples:
        buckets["validation"] = buckets["validation"][: data_cfg.max_eval_examples]
        buckets["test"] = buckets["test"][: data_cfg.max_eval_examples]

    for split in SPLITS:
        write_jsonl(out_dir / f"{split}.jsonl", buckets[split])

    stats.kept = sum(len(v) for v in buckets.values())
    stats.counts = {s: len(buckets[s]) for s in SPLITS}

    # --- 6. Guard set ------------------------------------------------------- #
    guard_count = 0
    if include_guard:
        guard_count = _prepare_guard(out_dir, seed=data_cfg.seed, limit=200)

    summary = {
        "sql": stats.as_dict(),
        "guard_examples": guard_count,
        "licence_audit_passed": manifest["passed"],
        "output_dir": str(out_dir),
    }
    (out_dir / "prep_stats.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _prepare_guard(out_dir: Path, *, seed: int, limit: int) -> int:
    """Build the catastrophic-forgetting guard set from Dolly.

    Deliberately excludes anything that looks like a database or SQL question,
    so the guard measures *general* ability that SQL fine-tuning might erode.
    """
    from datasets import load_dataset

    src = get_source("dolly_guard")
    ds = load_dataset(src.hf_id, split=src.split, revision=src.revision)

    banned = ("sql", "select ", "database", "query", "table")
    rows: list[dict] = []
    for row in ds:
        instruction = (row.get("instruction") or "").strip()
        response = (row.get("response") or "").strip()
        context = (row.get("context") or "").strip()
        if not instruction or not response or context:
            continue  # closed-book only: no long passages to read
        if any(b in instruction.lower() for b in banned):
            continue
        if len(instruction) > 300 or len(response) < 20:
            continue
        rows.append(
            {
                "id": _hash(instruction)[:16],
                "instruction": instruction,
                "context": "",
                "reference": response,
                "category": row.get("category", ""),
            }
        )

    rows.sort(key=lambda r: _hash(str(seed), r["id"]))
    rows = rows[:limit]
    write_jsonl(out_dir / "guard.jsonl", rows)
    return len(rows)
