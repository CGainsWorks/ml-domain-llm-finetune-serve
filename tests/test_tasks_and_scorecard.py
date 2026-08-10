"""Task scoring, prompt consistency and scorecard bookkeeping."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dllm.eval.scorecard import Scorecard, render_comparison, render_single
from dllm.eval.tasks import build_task, load_spec
from dllm.eval.tasks.base import TaskResult, TaskSpec
from dllm.eval.tasks.general_guard import GeneralGuardTask
from dllm.eval.tasks.sql_execution import SqlExecutionTask
from dllm.paths import EVAL_TASKS
from dllm.prompts import GUARD_PROMPT_VERSION, SQL_PROMPT_VERSION, guard_messages, sql_messages

SHIPPED_TASKS = ["sql_exec_holdout", "sql_exec_validation", "general_guard"]


class TestTaskRegistry:
    @pytest.mark.parametrize("task_id", SHIPPED_TASKS)
    def test_shipped_tasks_load(self, task_id: str) -> None:
        spec = load_spec(task_id)
        assert spec.id == task_id
        assert spec.kind
        assert spec.description.strip()

    @pytest.mark.parametrize("task_id", SHIPPED_TASKS)
    def test_shipped_tasks_build(self, task_id: str) -> None:
        assert build_task(task_id).id == task_id

    def test_unknown_task_lists_alternatives(self) -> None:
        with pytest.raises(FileNotFoundError, match="Available"):
            load_spec("no_such_task")

    def test_every_yaml_in_the_directory_is_valid(self) -> None:
        for path in EVAL_TASKS.glob("*.yaml"):
            assert build_task(path.stem).id == path.stem


class TestPrompts:
    def test_sql_prompt_contains_schema_and_question(self) -> None:
        messages = sql_messages("How many?", "CREATE TABLE t (a VARCHAR)")
        joined = " ".join(m["content"] for m in messages)
        assert "How many?" in joined
        assert "CREATE TABLE t (a VARCHAR)" in joined

    def test_sql_prompt_is_deterministic(self) -> None:
        a = sql_messages("Q", "S")
        b = sql_messages("Q", "S")
        assert a == b

    def test_guard_prompt_has_no_sql_framing(self) -> None:
        """The guard must not itself prime the model to emit SQL."""
        joined = " ".join(m["content"] for m in guard_messages("Who wrote Hamlet?")).lower()
        assert "sql" not in joined
        assert "schema" not in joined

    def test_training_and_eval_share_the_prompt_builder(self) -> None:
        from dllm.train.dataset import to_prompt_completion

        item = {
            "question": "Q",
            "context": "CREATE TABLE t (a VARCHAR)",
            "answer": "SELECT a FROM t",
        }
        assert to_prompt_completion(item)["prompt"] == sql_messages(
            item["question"], item["context"]
        )


class TestSqlTaskScoring:
    def _task(self, tmp_path: Path) -> SqlExecutionTask:
        return SqlExecutionTask(TaskSpec(id="t", kind="sql_execution", dataset="unused"), tmp_path)

    def test_all_correct(self, tmp_path: Path) -> None:
        items = [
            {
                "id": "1",
                "question": "q",
                "context": "CREATE TABLE t (a VARCHAR, b VARCHAR)",
                "answer": 'SELECT a FROM t WHERE b = "x"',
            }
        ]
        result = self._task(tmp_path).score(items, ['SELECT a FROM t WHERE b = "x"'])
        assert result.metrics["exec_acc"] == 1.0
        assert result.metrics["exec_acc_nonempty"] == 1.0
        assert result.metrics["valid_sql_rate"] == 1.0
        assert result.metrics["schema_adherence"] == 1.0
        assert result.metrics["exact_match_norm"] == 1.0

    def test_all_wrong(self, tmp_path: Path) -> None:
        items = [
            {
                "id": "1",
                "question": "q",
                "context": "CREATE TABLE t (a VARCHAR, b VARCHAR)",
                "answer": 'SELECT a FROM t WHERE b = "x"',
            }
        ]
        result = self._task(tmp_path).score(items, ["SELECT a FROM t"])
        assert result.metrics["exec_acc"] == 0.0

    def test_gold_unexecutable_leaves_the_denominator(self, tmp_path: Path) -> None:
        items = [
            {"id": "1", "question": "q", "context": "broken", "answer": "SELECT a FROM t"},
            {
                "id": "2",
                "question": "q",
                "context": "CREATE TABLE t (a VARCHAR)",
                "answer": "SELECT a FROM t",
            },
        ]
        result = self._task(tmp_path).score(items, ["SELECT a FROM t", "SELECT a FROM t"])
        assert result.counts["gold_unexecutable"] == 1
        assert result.counts["scorable"] == 1
        assert result.metrics["exec_acc"] == 1.0, "the broken item must not count against the model"

    def test_records_are_kept(self, tmp_path: Path) -> None:
        items = [
            {
                "id": "1",
                "question": "q",
                "context": "CREATE TABLE t (a VARCHAR)",
                "answer": "SELECT a FROM t",
            }
        ]
        result = self._task(tmp_path).score(items, ["SELECT a FROM t"])
        assert result.records[0]["gold"] == "SELECT a FROM t"

    def test_mismatched_lengths_raise(self, tmp_path: Path) -> None:
        items = [
            {
                "id": "1",
                "question": "q",
                "context": "CREATE TABLE t (a VARCHAR)",
                "answer": "SELECT a FROM t",
            }
        ]
        with pytest.raises(ValueError):
            self._task(tmp_path).score(items, [])


class TestGuardTaskScoring:
    def _task(self, tmp_path: Path) -> GeneralGuardTask:
        return GeneralGuardTask(
            TaskSpec(id="g", kind="instruction_guard", dataset="unused"), tmp_path
        )

    def test_prose_answers_score_full(self, tmp_path: Path) -> None:
        items = [
            {"id": "1", "instruction": "Who wrote Hamlet?", "reference": "Shakespeare wrote it"}
        ]
        result = self._task(tmp_path).score(items, ["Hamlet was written by William Shakespeare."])
        assert result.metrics["answers_in_prose"] == 1.0
        assert result.metrics["non_empty_rate"] == 1.0

    def test_sql_answer_is_caught(self, tmp_path: Path) -> None:
        """The exact failure this task exists to detect."""
        items = [{"id": "1", "instruction": "Who wrote Hamlet?", "reference": "Shakespeare"}]
        result = self._task(tmp_path).score(
            items, ["SELECT author FROM books WHERE title = 'Hamlet'"]
        )
        assert result.metrics["answers_in_prose"] == 0.0
        assert result.counts["sql_shaped"] == 1

    def test_empty_answer_is_caught(self, tmp_path: Path) -> None:
        items = [{"id": "1", "instruction": "Who wrote Hamlet?", "reference": "Shakespeare"}]
        result = self._task(tmp_path).score(items, [""])
        assert result.metrics["non_empty_rate"] == 0.0


class TestScorecard:
    def _card(self, label: str, metrics: dict, prompt: str = SQL_PROMPT_VERSION) -> Scorecard:
        return Scorecard(
            label=label,
            created_at="2026-01-01T00:00:00+00:00",
            tasks={"t": {"task_id": "t", "n": 10, "metrics": metrics, "counts": {}}},
            metrics={f"t.{k}": v for k, v in metrics.items()},
            prompt_versions={"sql": prompt, "guard": GUARD_PROMPT_VERSION},
        )

    def test_build_flattens_metrics(self) -> None:
        result = TaskResult("mytask", 5, {"acc": 0.5})
        card = Scorecard.build("x", [result], backend={}, generation={})
        assert card.metrics["mytask.acc"] == 0.5

    def test_roundtrip(self, tmp_path: Path) -> None:
        card = self._card("a", {"acc": 0.5})
        path = card.save(tmp_path / "c.json")
        loaded = Scorecard.load(path)
        assert loaded.metrics == card.metrics
        assert loaded.label == "a"

    def test_saved_json_is_readable(self, tmp_path: Path) -> None:
        path = self._card("a", {"acc": 0.5}).save(tmp_path / "c.json")
        assert json.loads(path.read_text(encoding="utf-8"))["label"] == "a"

    def test_comparable_when_identical_shape(self) -> None:
        ok, problems = self._card("b", {"acc": 0.6}).comparable_with(self._card("a", {"acc": 0.5}))
        assert ok and not problems

    def test_not_comparable_across_prompt_versions(self) -> None:
        ok, problems = self._card("b", {"acc": 0.6}, prompt="v2").comparable_with(
            self._card("a", {"acc": 0.5})
        )
        assert not ok
        assert any("prompt" in p for p in problems)

    def test_render_single_lists_metrics(self) -> None:
        assert "acc" in render_single(self._card("a", {"acc": 0.5}))

    def test_render_comparison_shows_delta(self) -> None:
        body = render_comparison(self._card("a", {"acc": 0.5}), self._card("b", {"acc": 0.6}))
        assert "+0.1000" in body

    def test_render_comparison_marks_direction(self) -> None:
        """A rise in an error-rate metric must not be shown as an improvement."""
        base = self._card("a", {"exec_error_rate": 0.1})
        worse = self._card("b", {"exec_error_rate": 0.3})
        assert "▼" in render_comparison(base, worse)
