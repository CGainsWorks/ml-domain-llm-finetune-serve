"""Execution-based scoring: the metric everything else hangs off."""

from __future__ import annotations

from pathlib import Path

import pytest

from dllm.eval.sql.execute import Outcome, evaluate_prediction, results_match
from dllm.eval.sql.materialise import parse_schema
from dllm.eval.sql.normalise import parse

SCHEMA = "CREATE TABLE table_74 (school VARCHAR, player VARCHAR)"
GOLD = 'SELECT school FROM table_74 WHERE player = "Jared Butler"'


def score(pred: str, *, schema: str = SCHEMA, gold: str = GOLD):
    return evaluate_prediction(context=schema, gold_sql=gold, pred_sql=pred)


class TestCorrect:
    def test_identical(self) -> None:
        assert score(GOLD).outcome is Outcome.CORRECT

    def test_quote_style_is_not_an_error(self) -> None:
        assert score("SELECT school FROM table_74 WHERE player = 'Jared Butler'").is_correct

    def test_case_and_whitespace(self) -> None:
        assert score('select   school from table_74 where player = "Jared Butler"').is_correct

    def test_reordered_predicates(self) -> None:
        schema = "CREATE TABLE t (a VARCHAR, b VARCHAR, c VARCHAR)"
        gold = 'SELECT c FROM t WHERE a = "x" AND b = "y"'
        result = evaluate_prediction(
            context=schema, gold_sql=gold, pred_sql='SELECT c FROM t WHERE b = "y" AND a = "x"'
        )
        assert result.is_correct, "equivalent queries must not be punished for word order"

    def test_markdown_wrapped_prediction(self) -> None:
        assert score(f"```sql\n{GOLD}\n```").is_correct


class TestWrong:
    def test_wrong_column(self) -> None:
        assert not score('SELECT player FROM table_74 WHERE player = "Jared Butler"').is_correct

    def test_missing_filter(self) -> None:
        """The case that makes randomly-populated tables useless as a metric."""
        assert not score("SELECT school FROM table_74").is_correct

    def test_aggregate_without_filter_differs(self) -> None:
        schema = "CREATE TABLE t (id VARCHAR, team VARCHAR)"
        gold = 'SELECT COUNT(*) FROM t WHERE team = "Reds"'
        result = evaluate_prediction(
            context=schema, gold_sql=gold, pred_sql="SELECT COUNT(*) FROM t"
        )
        assert not result.is_correct

    def test_hallucinated_column_is_an_exec_error(self) -> None:
        result = score('SELECT school FROM table_74 WHERE nope = "x"')
        assert result.outcome is Outcome.EXEC_ERROR
        assert result.schema_ok is False

    def test_unparseable(self) -> None:
        assert score("this is not sql ((").outcome is Outcome.UNPARSEABLE

    def test_mutation_is_rejected(self) -> None:
        assert score("DROP TABLE table_74").outcome is Outcome.NOT_READ_ONLY

    def test_empty_prediction(self) -> None:
        assert score("").outcome is Outcome.UNPARSEABLE


class TestGoldQuality:
    def test_bad_schema_disqualifies_the_item(self) -> None:
        result = evaluate_prediction(context="not a schema", gold_sql=GOLD, pred_sql=GOLD)
        assert result.outcome is Outcome.GOLD_UNEXECUTABLE
        assert not result.scorable, "a broken item must leave the denominator"

    def test_gold_referencing_a_missing_table(self) -> None:
        result = evaluate_prediction(
            context="CREATE TABLE t (a VARCHAR)",
            gold_sql="SELECT a FROM other",
            pred_sql="SELECT a FROM t",
        )
        assert result.outcome is Outcome.GOLD_UNEXECUTABLE


class TestSchemaAdherence:
    def test_alias_is_not_a_hallucination(self) -> None:
        schema = "CREATE TABLE t (a VARCHAR, b VARCHAR)"
        gold = "SELECT b, COUNT(*) AS n FROM t GROUP BY b ORDER BY n DESC"
        result = evaluate_prediction(context=schema, gold_sql=gold, pred_sql=gold)
        assert result.schema_ok

    def test_table_alias_is_not_a_hallucination(self) -> None:
        schema = "CREATE TABLE alpha (id VARCHAR)"
        gold = "SELECT x.id FROM alpha AS x"
        result = evaluate_prediction(context=schema, gold_sql=gold, pred_sql=gold)
        assert result.schema_ok


class TestResultsMatch:
    def test_row_order_ignored_without_order_by(self) -> None:
        assert results_match([("a",), ("b",)], [("b",), ("a",)], ordered=False)

    def test_row_order_enforced_with_order_by(self) -> None:
        assert not results_match([("a",), ("b",)], [("b",), ("a",)], ordered=True)

    def test_column_order_matters(self) -> None:
        assert not results_match([(1, 2)], [(2, 1)], ordered=False)

    def test_int_and_float_are_equal(self) -> None:
        assert results_match([(1,)], [(1.0,)], ordered=False)

    def test_numeric_string_equals_number(self) -> None:
        assert results_match([("1994",)], [(1994,)], ordered=False)

    def test_case_insensitive_strings(self) -> None:
        assert results_match([("Leeds",)], [("leeds",)], ordered=False)

    def test_different_lengths(self) -> None:
        assert not results_match([(1,)], [(1,), (2,)], ordered=False)

    def test_both_empty_matches(self) -> None:
        assert results_match([], [], ordered=False)


class TestOrderByDetection:
    @pytest.mark.parametrize(
        ("gold", "ordered"),
        [
            ("SELECT a FROM t ORDER BY a", True),
            ("SELECT a FROM t", False),
        ],
    )
    def test_order_by_changes_comparison(self, gold: str, ordered: bool) -> None:
        assert ("order by" in parse(gold).sql(dialect="sqlite").lower()) is ordered


class TestMalformedPredictionsNeverRaise:
    """A single bad prediction must never abort a 500-item run.

    The unterminated-string-literal case is not hypothetical: it raised
    sqlglot's TokenError (a lexer error, not a ParseError) and killed a full
    baseline run after all generation had already completed.
    """

    @pytest.mark.parametrize(
        "pred",
        [
            "SELECT a FROM t WHERE b = 'unterminated",
            'SELECT a FROM t WHERE b = "unterminated',
            "SELECT a FROM t WHERE b = `weird",
            "SELECT " + "(" * 500,
            "\x00\x01\x02",
            "SELECT a FROM t /* unterminated comment",
            "'''",
            "SELECT * FROM t WHERE a = '\\'",
            "select" * 2000,
            "",
            "   ",
            "😀",
        ],
    )
    def test_scoring_returns_a_result_instead_of_raising(self, pred: str) -> None:
        result = score(pred)
        assert result.outcome in {
            Outcome.UNPARSEABLE,
            Outcome.EXEC_ERROR,
            Outcome.NOT_READ_ONLY,
            Outcome.WRONG,
        }
        assert not result.is_correct

    def test_task_scoring_survives_a_batch_of_garbage(self) -> None:
        from dllm.eval.tasks.base import TaskSpec
        from dllm.eval.tasks.sql_execution import SqlExecutionTask

        task = SqlExecutionTask(TaskSpec(id="t", kind="sql_execution", dataset="x"), Path("."))
        items = [
            {"id": str(i), "question": "q", "context": SCHEMA, "answer": GOLD} for i in range(4)
        ]
        preds = ["SELECT a FROM t WHERE b = 'oops", "\x00", GOLD, "((("]
        result = task.score(items, preds)
        assert result.n == 4
        assert result.metrics["exec_acc"] >= 0.0


class TestGuardsAgainstRunaway:
    def test_cartesian_product_terminates(self) -> None:
        """A pathological prediction must not hang the harness."""
        schema = "CREATE TABLE a (x VARCHAR); CREATE TABLE b (y VARCHAR)"
        result = evaluate_prediction(
            context=schema,
            gold_sql="SELECT x FROM a",
            pred_sql="SELECT a.x FROM a, b, a AS a2, b AS b2, a AS a3, b AS b3",
        )
        assert result.outcome in {Outcome.WRONG, Outcome.EXEC_ERROR}


def test_schema_parses_for_all_fixtures() -> None:
    assert parse_schema(SCHEMA).column_names() == {"school", "player"}
