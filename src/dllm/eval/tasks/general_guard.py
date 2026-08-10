"""Catastrophic-forgetting guard.

Fine-tuning hard on one narrow format is very good at teaching a model to
produce that format *for everything*. A SQL fine-tune that answers "Who wrote
Pride and Prejudice?" with ``SELECT author FROM books`` has improved on the
primary metric and become useless.

This task is not trying to measure general knowledge -- scoring open-ended
answers properly needs a judge, and a judge in the promotion gate is a
liability. It measures the specific, cheaply detectable damage that this
particular fine-tune could do:

``answers_in_prose``
    Fraction of general questions answered in prose rather than SQL. **This is
    the guard metric**; ``max_regression`` on it is what blocks a promotion.

``non_empty_rate``
    Fraction producing any output at all. Catches a model that has collapsed
    into emitting the EOS token immediately.

``mean_chars``
    Mean answer length. A collapse from ~200 to ~15 characters is visible here
    before anything else notices.

``keyword_overlap``
    Crude lexical overlap with the reference answer. Weak on its own, useful as
    a trend across checkpoints. Never used as a gate.
"""

from __future__ import annotations

import re
from pathlib import Path

from dllm.eval.tasks.base import Task, TaskResult, TaskSpec, resolve_dataset_path
from dllm.prompts import Message, guard_messages

_SQL_SHAPED = re.compile(
    r"^\s*(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE)\b|\bFROM\s+\w+\s+WHERE\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[a-z0-9']+")

_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "of",
    "to",
    "in",
    "and",
    "or",
    "it",
    "that",
    "this",
    "for",
    "on",
    "with",
    "as",
    "by",
    "at",
    "from",
    "be",
}


class GeneralGuardTask(Task):
    kind = "instruction_guard"

    def __init__(self, spec: TaskSpec, root: Path) -> None:
        super().__init__(spec, root)
        self.keep_records = bool(spec.options.get("keep_records", True))

    def load(self, limit: int | None = None) -> list[dict]:
        from dllm.data.prepare import read_jsonl

        path = resolve_dataset_path(self.spec.dataset, self.root)
        if not path.is_file():
            raise FileNotFoundError(
                f"task {self.id!r} needs {path}. Run `dllm data prepare` first."
            )
        effective = limit if limit is not None else self.spec.limit
        return read_jsonl(path, effective)

    def conversations(self, items: list[dict]) -> list[list[Message]]:
        return [guard_messages(i["instruction"], i.get("context", "")) for i in items]

    def score(self, items: list[dict], completions: list[str]) -> TaskResult:
        n = len(items)
        prose = 0
        non_empty = 0
        total_chars = 0
        overlaps: list[float] = []
        records: list[dict] = []

        for item, raw in zip(items, completions, strict=True):
            text = raw.strip()
            sql_shaped = bool(_SQL_SHAPED.search(text))
            if text:
                non_empty += 1
            if text and not sql_shaped:
                prose += 1
            total_chars += len(text)

            ref_words = set(_WORD.findall(item.get("reference", "").lower())) - _STOPWORDS
            out_words = set(_WORD.findall(text.lower())) - _STOPWORDS
            overlaps.append(len(ref_words & out_words) / len(ref_words) if ref_words else 0.0)

            if self.keep_records:
                records.append(
                    {
                        # Stored in full, not truncated: `dllm eval rescore`
                        # replays these through the scorer, and a truncated
                        # completion would silently change mean_chars.
                        "id": item["id"],
                        "instruction": item["instruction"],
                        "pred": text,
                        "sql_shaped": sql_shaped,
                    }
                )

        def ratio(num: float, den: int) -> float:
            return round(num / den, 6) if den else 0.0

        metrics = {
            "answers_in_prose": ratio(prose, n),
            "non_empty_rate": ratio(non_empty, n),
            "mean_chars": round(total_chars / n, 2) if n else 0.0,
            "keyword_overlap": ratio(sum(overlaps), n),
        }
        counts = {"sql_shaped": n - prose, "empty": n - non_empty}
        return TaskResult(self.id, n, metrics, counts, records)
