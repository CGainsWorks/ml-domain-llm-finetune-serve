"""Execute a predicted query beside its gold query and compare result sets.

The comparison rules are the interesting part:

* **Order only matters when the gold query says it does.** With no ``ORDER BY``,
  SQL result order is undefined, so rows are compared as a multiset.
* **Column order is preserved.** ``SELECT a, b`` and ``SELECT b, a`` are
  different answers; sorting the columns away would hide real errors.
* **Values are compared loosely enough to survive representation noise.**
  ``1`` and ``1.0`` are equal; strings are stripped and case-folded. Anything
  looser would start accepting wrong answers.
* **A gold query that will not run disqualifies the item.** It is scored as
  ``gold_unexecutable`` and dropped from the denominator rather than counted as
  a model failure, and the count is reported so the number stays honest.
* **Empty gold results are tracked separately.** If gold returns nothing, any
  prediction that also errors into nothing looks correct. ``exec_acc_nonempty``
  restricts to items where the gold result has rows, and that is the metric the
  promotion gate uses.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import Enum
from typing import Any

from sqlglot import expressions as exp

from dllm.eval.sql.materialise import Schema, build_for_item
from dllm.eval.sql.normalise import (
    SqlParseFailure,
    is_read_only,
    parse,
    referenced_columns,
    referenced_tables,
    to_sqlite,
)

# SQLite has no wall-clock query timeout; the progress handler is the hook.
# ~2e6 VM instructions is far more than any query over a 12-row table needs.
PROGRESS_OPS = 2_000_000


class Outcome(str, Enum):
    CORRECT = "correct"
    WRONG = "wrong"
    UNPARSEABLE = "unparseable"
    NOT_READ_ONLY = "not_read_only"
    EXEC_ERROR = "exec_error"
    GOLD_UNEXECUTABLE = "gold_unexecutable"


@dataclass
class ExecResult:
    outcome: Outcome
    gold_rows: list[tuple] | None = None
    pred_rows: list[tuple] | None = None
    gold_empty: bool = False
    error: str = ""
    schema_ok: bool = True

    @property
    def is_correct(self) -> bool:
        return self.outcome is Outcome.CORRECT

    @property
    def scorable(self) -> bool:
        """Whether this item counts towards accuracy at all."""
        return self.outcome is not Outcome.GOLD_UNEXECUTABLE


def _install_guard(conn: sqlite3.Connection) -> None:
    state = {"n": 0}

    def handler() -> int:
        state["n"] += 1
        return 1 if state["n"] > 200 else 0  # non-zero aborts the query

    conn.set_progress_handler(handler, PROGRESS_OPS // 200)


def run_query(conn: sqlite3.Connection, sql: str) -> list[tuple]:
    """Run one SELECT and return its rows. Raises ``sqlite3.Error`` on failure."""
    _install_guard(conn)
    try:
        cur = conn.execute(sql)
        return cur.fetchall()
    finally:
        conn.set_progress_handler(None, 0)


def _canon_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int | float):
        f = float(v)
        return int(f) if f.is_integer() else round(f, 6)
    if isinstance(v, bytes):
        return v
    s = str(v).strip()
    # A numeric-looking string and the number itself are the same answer.
    try:
        f = float(s)
        return int(f) if f.is_integer() else round(f, 6)
    except ValueError:
        return s.casefold()


def _canon_rows(rows: list[tuple]) -> list[tuple]:
    return [tuple(_canon_value(v) for v in row) for row in rows]


def _sort_key(row: tuple) -> tuple:
    # Rows can mix types; sort on (type name, repr) so the order is total.
    return tuple((type(v).__name__, repr(v)) for v in row)


def results_match(gold: list[tuple], pred: list[tuple], *, ordered: bool) -> bool:
    """Compare two result sets under the rules described in the module docstring."""
    g, p = _canon_rows(gold), _canon_rows(pred)
    if len(g) != len(p):
        return False
    if not g:
        return True
    if len({len(r) for r in g} | {len(r) for r in p}) != 1:
        return False
    if ordered:
        return g == p
    return sorted(g, key=_sort_key) == sorted(p, key=_sort_key)


def _has_order_by(sql: str) -> bool:
    try:
        return "order by" in parse(sql).sql(dialect="sqlite").lower()
    except SqlParseFailure:
        return "order by" in sql.lower()


def query_defined_names(expr: exp.Expression) -> set[str]:
    """Names the query introduces itself: column aliases, table aliases, CTEs.

    ``SELECT COUNT(*) AS n FROM t ORDER BY n`` references ``n`` as a column even
    though no such column exists in the schema. Without this, every aliased
    query would be flagged as hallucinating an identifier.
    """
    names: set[str] = set()
    for node in expr.find_all(exp.Alias, exp.TableAlias, exp.CTE):
        alias = node.alias_or_name
        if alias:
            names.add(alias.lower())
    return names


def check_schema_adherence(pred_sql: str, schema: Schema) -> bool:
    """True if every table and column the prediction names exists in the schema.

    A separate signal from execution accuracy: it isolates *hallucinated
    identifiers*, which is the failure mode fine-tuning fixes first and the one
    most visible to a user.
    """
    try:
        expr = parse(pred_sql)
    except SqlParseFailure:
        return False

    defined = query_defined_names(expr)
    if not (referenced_tables(expr) - defined) <= schema.table_names():
        return False
    return (referenced_columns(expr) - defined) <= schema.column_names()


def evaluate_prediction(
    *,
    context: str,
    gold_sql: str,
    pred_sql: str,
    anchor_rows: int = 3,
    filler_rows: int = 9,
) -> ExecResult:
    """Score one prediction by executing it against a materialised schema.

    Never raises. A 500-item eval that dies on item 437 because a model emitted
    something the parser had not seen before costs half an hour of GPU time and
    produces nothing; an unexpected failure is scored as a wrong answer and
    recorded, which is both cheaper and more honest.
    """
    try:
        return _evaluate_prediction(
            context=context,
            gold_sql=gold_sql,
            pred_sql=pred_sql,
            anchor_rows=anchor_rows,
            filler_rows=filler_rows,
        )
    except Exception as e:  # noqa: BLE001 - deliberate catch-all, see docstring
        return ExecResult(
            Outcome.EXEC_ERROR, error=f"harness error: {type(e).__name__}: {e}", schema_ok=False
        )


def _evaluate_prediction(
    *,
    context: str,
    gold_sql: str,
    pred_sql: str,
    anchor_rows: int,
    filler_rows: int,
) -> ExecResult:
    try:
        conn, schema = build_for_item(
            context, gold_sql, anchor_rows=anchor_rows, filler_rows=filler_rows
        )
    except Exception as e:  # noqa: BLE001 - a bad schema is a dataset problem
        return ExecResult(Outcome.GOLD_UNEXECUTABLE, error=f"schema: {e}")

    try:
        try:
            gold_rows = run_query(conn, to_sqlite(gold_sql))
        except Exception as e:  # noqa: BLE001
            return ExecResult(Outcome.GOLD_UNEXECUTABLE, error=f"gold: {e}")

        gold_empty = not gold_rows
        schema_ok = check_schema_adherence(pred_sql, schema)

        try:
            expr = parse(pred_sql)
        except SqlParseFailure as e:
            return ExecResult(
                Outcome.UNPARSEABLE,
                gold_rows=gold_rows,
                gold_empty=gold_empty,
                error=str(e),
                schema_ok=False,
            )

        if not is_read_only(expr):
            return ExecResult(
                Outcome.NOT_READ_ONLY,
                gold_rows=gold_rows,
                gold_empty=gold_empty,
                error=f"prediction is a {type(expr).__name__}, not a query",
                schema_ok=schema_ok,
            )

        try:
            pred_rows = run_query(conn, expr.sql(dialect="sqlite"))
        except Exception as e:  # noqa: BLE001
            return ExecResult(
                Outcome.EXEC_ERROR,
                gold_rows=gold_rows,
                gold_empty=gold_empty,
                error=str(e),
                schema_ok=schema_ok,
            )

        ordered = _has_order_by(gold_sql)
        correct = results_match(gold_rows, pred_rows, ordered=ordered)
        return ExecResult(
            Outcome.CORRECT if correct else Outcome.WRONG,
            gold_rows=gold_rows,
            pred_rows=pred_rows,
            gold_empty=gold_empty,
            schema_ok=schema_ok,
        )
    finally:
        conn.close()
