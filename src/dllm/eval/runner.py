"""Run a set of tasks against a backend and produce a scorecard."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

from dllm.config import EvalConfig, ModelConfig, QuantConfig
from dllm.eval.backends import Backend, build_backend
from dllm.eval.scorecard import Scorecard
from dllm.eval.tasks import build_task
from dllm.eval.tasks.base import TaskResult
from dllm.paths import ROOT

log = logging.getLogger(__name__)


def _fingerprint(items: list[dict]) -> str:
    """Stable hash of the exact items scored.

    Two scorecards claiming the same task and item count can still be scoring
    different rows if the dataset was re-prepared. The fingerprint makes that
    visible instead of turning into a mysterious delta.
    """
    h = hashlib.sha256()
    for item in items:
        h.update(str(item.get("id", "")).encode())
    return h.hexdigest()[:16]


def run_tasks(
    backend: Backend,
    eval_cfg: EvalConfig,
    *,
    root: Path | None = None,
    label: str = "run",
) -> tuple[Scorecard, list[TaskResult]]:
    """Evaluate every task in ``eval_cfg.tasks``.

    Returns the scorecard and the raw results. Per-item records are deliberately
    kept out of the scorecard -- it is a summary meant to be read and diffed --
    so they are returned alongside for the caller to write separately.
    """
    root = root or ROOT
    started = time.perf_counter()
    results: list[TaskResult] = []
    fingerprints: dict[str, dict] = {}

    for task_id in eval_cfg.tasks:
        task = build_task(task_id, root=root)
        items = task.load(eval_cfg.limit)
        if not items:
            log.warning("task %s has no items; skipping", task_id)
            continue

        log.info("task %s: generating for %d items", task_id, len(items))
        t0 = time.perf_counter()
        completions = backend.generate(task.conversations(items), eval_cfg.generation)
        gen_s = time.perf_counter() - t0

        result = task.score(items, completions)
        results.append(result)
        fingerprints[task_id] = {
            "n": len(items),
            "items_sha256_16": _fingerprint(items),
            "dataset": task.spec.dataset,
            "generation_seconds": round(gen_s, 2),
        }
        log.info(
            "task %s: %s",
            task_id,
            ", ".join(f"{k}={v}" for k, v in result.metrics.items()),
        )

    if not results:
        raise RuntimeError("no tasks produced results; check evals/tasks and data/processed")

    card = Scorecard.build(
        label,
        results,
        backend=backend.metadata(),
        generation=eval_cfg.generation.model_dump(),
        dataset_fingerprint=fingerprints,
        runtime_s=time.perf_counter() - started,
    )
    return card, results


def rescore(
    scorecard_path: str | Path,
    eval_cfg: EvalConfig,
    *,
    label: str,
    root: Path | None = None,
) -> tuple[Scorecard, Path]:
    """Re-score saved predictions without regenerating them.

    Scoring is deterministic and entirely separate from generation, so a change
    to a metric does not require paying for inference again. This matters: the
    baseline eval takes over half an hour on a consumer card, and a scoring
    change that forced a re-run would create pressure to leave the scoring
    alone.

    Reads the ``<label>.records.jsonl`` written next to the scorecard and
    replays every completion through the current scorer.
    """
    root = root or ROOT
    started = time.perf_counter()
    old_path = Path(scorecard_path)
    old = Scorecard.load(old_path)

    records_path = old_path.with_name(f"{old_path.stem}.records.jsonl")
    if not records_path.is_file():
        raise FileNotFoundError(
            f"no per-item records at {records_path}; rescoring needs them. "
            "Re-run `dllm eval run` with save_records enabled."
        )

    by_task: dict[str, list[dict]] = {}
    with records_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                by_task.setdefault(rec["task"], []).append(rec)

    results: list[TaskResult] = []
    for task_id, recs in by_task.items():
        task = build_task(task_id, root=root)
        index = {item["id"]: item for item in task.load(None)}
        missing = [r["id"] for r in recs if r["id"] not in index]
        if missing:
            raise RuntimeError(
                f"task {task_id}: {len(missing)} recorded items are not in the current "
                f"dataset (first: {missing[0]}). The data was re-prepared since this "
                "run; regenerate rather than rescore."
            )
        items = [index[r["id"]] for r in recs]
        # `raw` holds the untrimmed completion when it differed from `pred`.
        completions = [r.get("raw") or r["pred"] for r in recs]
        results.append(task.score(items, completions))
        log.info("rescored %s (%d items)", task_id, len(items))

    card = Scorecard.build(
        label,
        results,
        backend={**old.backend, "rescored_from": str(old_path)},
        generation=old.generation,
        dataset_fingerprint=old.dataset_fingerprint,
        runtime_s=time.perf_counter() - started,
    )
    out_path = root / eval_cfg.output.format(label=label)
    card.save(out_path)

    records_out = out_path.with_name(f"{out_path.stem}.records.jsonl")
    with records_out.open("w", encoding="utf-8", newline="\n") as fh:
        for result in results:
            for rec in result.records:
                fh.write(json.dumps({"task": result.task_id, **rec}, ensure_ascii=False) + "\n")

    log.info("rescored scorecard written to %s", out_path)
    return card, out_path


def evaluate(
    eval_cfg: EvalConfig,
    model_cfg: ModelConfig,
    *,
    label: str,
    model_path: str | None = None,
    adapter_path: str | None = None,
    quant: QuantConfig | None = None,
    root: Path | None = None,
    save_records: bool = True,
) -> tuple[Scorecard, Path]:
    """Build the backend, run the tasks, write the scorecard. Returns both."""
    root = root or ROOT
    with build_backend(
        eval_cfg.backend, model_cfg, model_path=model_path, adapter_path=adapter_path, quant=quant
    ) as backend:
        card, results = run_tasks(backend, eval_cfg, root=root, label=label)

    out_path = root / eval_cfg.output.format(label=label)
    card.save(out_path)
    log.info("scorecard written to %s", out_path)

    if save_records:
        # Per-item predictions live beside the scorecard but out of it: they are
        # large, and a reviewer wants to read the failures, not scroll past them.
        records_path = out_path.with_name(f"{out_path.stem}.records.jsonl")
        records_path.parent.mkdir(parents=True, exist_ok=True)
        with records_path.open("w", encoding="utf-8", newline="\n") as fh:
            for result in results:
                for rec in result.records:
                    fh.write(json.dumps({"task": result.task_id, **rec}, ensure_ascii=False) + "\n")
        log.info("per-item records written to %s", records_path)

    return card, out_path
