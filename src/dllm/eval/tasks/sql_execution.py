"""The primary task: does the generated query return the right rows?

Metrics reported, in descending order of how much they should be trusted:

``exec_acc_nonempty``
    Execution accuracy restricted to items whose gold query returns at least
    one row. **This is the gate metric.** On items where gold returns nothing,
    a prediction that also returns nothing scores as correct even when it is
    nonsense, so those items flatter the model and are excluded here.

``exec_acc``
    Execution accuracy over every item with an executable gold query. Reported
    because it is the number most people expect to see, with the caveat above.

``schema_adherence``
    Fraction of predictions referencing only tables and columns that exist.
    Isolates hallucinated identifiers, which is the first thing fine-tuning
    fixes and the most visible failure to a user.

``valid_sql_rate``
    Fraction that parse as SQL at all. Near 1.0 for any competent instruct
    model; a drop here after fine-tuning means something went badly wrong.

``exact_match_norm``
    Normalised string equality against the gold query. Reported for continuity
    with the text-to-SQL literature. It is a *lower bound* on correctness --
    it punishes correct queries written differently -- so it must never be the
    gate metric.

Items whose gold query cannot be executed are excluded from every denominator
and counted in ``counts.gold_unexecutable``.
"""

from __future__ import annotations

from pathlib import Path

from dllm.eval.sql.execute import Outcome, evaluate_prediction
from dllm.eval.sql.normalise import SqlParseFailure, normalise, parse, strip_markdown
from dllm.eval.tasks.base import Task, TaskResult, TaskSpec, resolve_dataset_path
from dllm.prompts import SQL_PROMPT_VERSION, Message, sql_messages


class SqlExecutionTask(Task):
    kind = "sql_execution"

    def __init__(self, spec: TaskSpec, root: Path) -> None:
        super().__init__(spec, root)
        self.anchor_rows = int(spec.options.get("anchor_rows", 3))
        self.filler_rows = int(spec.options.get("filler_rows", 9))
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
        return [sql_messages(i["question"], i["context"]) for i in items]

    def score(self, items: list[dict], completions: list[str]) -> TaskResult:
        counts = {o.value: 0 for o in Outcome}
        counts["gold_nonempty"] = 0
        counts["correct_nonempty"] = 0

        valid = 0
        schema_ok = 0
        exact = 0
        records: list[dict] = []

        for item, raw in zip(items, completions, strict=True):
            pred = strip_markdown(raw)
            result = evaluate_prediction(
                context=item["context"],
                gold_sql=item["answer"],
                pred_sql=pred,
                anchor_rows=self.anchor_rows,
                filler_rows=self.filler_rows,
            )
            counts[result.outcome.value] += 1

            if result.scorable:
                try:
                    parse(pred)
                    valid += 1
                except SqlParseFailure:
                    pass
                if result.schema_ok:
                    schema_ok += 1
                try:
                    if normalise(pred) == normalise(item["answer"]):
                        exact += 1
                except Exception:  # noqa: BLE001
                    # Cosmetic metric only: never let it take down a run.
                    pass
                if not result.gold_empty:
                    counts["gold_nonempty"] += 1
                    if result.is_correct:
                        counts["correct_nonempty"] += 1

            if self.keep_records:
                records.append(
                    {
                        "id": item["id"],
                        "question": item["question"],
                        "gold": item["answer"],
                        "pred": pred,
                        "raw": raw if raw != pred else None,
                        "outcome": result.outcome.value,
                        "schema_ok": result.schema_ok,
                        "gold_empty": result.gold_empty,
                        "error": result.error or None,
                    }
                )

        scorable = len(items) - counts[Outcome.GOLD_UNEXECUTABLE.value]
        gold_nonempty = counts["gold_nonempty"]

        def ratio(num: int, den: int) -> float:
            return round(num / den, 6) if den else 0.0

        metrics = {
            "exec_acc_nonempty": ratio(counts["correct_nonempty"], gold_nonempty),
            "exec_acc": ratio(counts[Outcome.CORRECT.value], scorable),
            "schema_adherence": ratio(schema_ok, scorable),
            "valid_sql_rate": ratio(valid, scorable),
            "exact_match_norm": ratio(exact, scorable),
            "exec_error_rate": ratio(counts[Outcome.EXEC_ERROR.value], scorable),
            "unparseable_rate": ratio(counts[Outcome.UNPARSEABLE.value], scorable),
        }
        counts["scorable"] = scorable
        return TaskResult(self.id, len(items), metrics, counts, records)


PROMPT_VERSION = SQL_PROMPT_VERSION
