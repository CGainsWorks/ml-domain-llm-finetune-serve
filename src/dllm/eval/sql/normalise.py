"""Canonicalising SQL so that string comparison means something.

Two problems this solves.

1. **Dialect.** ``sql-create-context`` writes string literals with double
   quotes (``WHERE player = "Jared Butler"``), which is MySQL-flavoured. SQLite
   reads a double-quoted token as an *identifier* first and only falls back to
   a string literal if no such column exists -- a silent source of wrong
   results. Everything is therefore parsed as MySQL and emitted as SQLite.

2. **Cosmetics.** ``SELECT COUNT(*)`` and ``select count( * )`` are the same
   query. Exact match on raw text under-reports accuracy badly, so the
   normalised form lower-cases identifiers and keywords, drops the trailing
   semicolon and re-emits from the parse tree.

Normalised exact match is still a *secondary* metric. The primary one is
execution accuracy -- see ``dllm.eval.sql.execute``.
"""

from __future__ import annotations

import re

import sqlglot
from sqlglot import expressions as exp
from sqlglot.errors import SqlglotError

READ_DIALECT = "mysql"
WRITE_DIALECT = "sqlite"

_WS = re.compile(r"\s+")
_FENCE = re.compile(r"^\s*```(?:sql)?\s*|\s*```\s*$", re.IGNORECASE)


class SqlParseFailure(ValueError):
    """The text could not be parsed as SQL."""


def strip_markdown(text: str) -> str:
    """Remove a code fence and any leading prose the model felt like adding."""
    t = text.strip()
    t = _FENCE.sub("", t).strip()

    # Models sometimes prefix an explanation. Take from the first statement
    # keyword onwards; anything before it is commentary, not SQL.
    m = re.search(r"\b(SELECT|WITH)\b", t, re.IGNORECASE)
    if m:
        t = t[m.start() :]

    # Cut at the first blank line -- trailing explanation after the query.
    t = t.split("\n\n", 1)[0]
    return t.strip().rstrip(";").strip()


def parse(sql: str) -> exp.Expression:
    """Parse one statement, raising :class:`SqlParseFailure` on anything else."""
    cleaned = strip_markdown(sql)
    if not cleaned:
        raise SqlParseFailure("empty query")
    try:
        parsed = sqlglot.parse_one(cleaned, read=READ_DIALECT)
    except (SqlglotError, RecursionError, ValueError) as e:
        # SqlglotError is the base of ParseError *and* TokenError. Catching only
        # ParseError misses the lexer: a model that emits an unterminated string
        # literal raises TokenError, which is a perfectly ordinary bad
        # prediction and must be scored, not propagated.
        raise SqlParseFailure(str(e)) from e
    if parsed is None:
        raise SqlParseFailure("parser returned nothing")
    return parsed


def is_read_only(expr: exp.Expression) -> bool:
    """True only for SELECT-shaped statements.

    A prediction of ``DROP TABLE`` is a wrong answer, not a valid query that
    happens to return no rows. Each item runs against a throwaway in-memory
    database so this is a scoring rule, not the security boundary -- but it is
    also why the harness never points at a real database.
    """
    if isinstance(expr, exp.Select | exp.Union | exp.Except | exp.Intersect):
        return True
    if isinstance(expr, exp.Subquery):
        return is_read_only(expr.this)
    return False


def to_sqlite(sql: str) -> str:
    """Transpile a MySQL-flavoured query to SQLite text."""
    return parse(sql).sql(dialect=WRITE_DIALECT)


def normalise(sql: str) -> str:
    """Canonical, comparable form of a query.

    Falls back to whitespace-collapsed lower-case text if the query does not
    parse, so that the metric degrades rather than throwing.
    """
    try:
        expr = parse(sql)
    except SqlParseFailure:
        return _WS.sub(" ", strip_markdown(sql).lower()).strip()

    normalised = expr.sql(dialect=WRITE_DIALECT, normalize=True, pretty=False, comments=False)
    return _WS.sub(" ", normalised.lower()).strip()


def exact_match(pred: str, gold: str) -> bool:
    """Normalised string equality."""
    return normalise(pred) == normalise(gold)


def referenced_tables(expr: exp.Expression) -> set[str]:
    """Lower-cased table names the query reads from, excluding CTE aliases."""
    cte_names = {c.alias_or_name.lower() for c in expr.find_all(exp.CTE)}
    names = {t.name.lower() for t in expr.find_all(exp.Table) if t.name}
    return names - cte_names


def referenced_columns(expr: exp.Expression) -> set[str]:
    """Lower-cased bare column names appearing anywhere in the query."""
    return {c.name.lower() for c in expr.find_all(exp.Column) if c.name and c.name != "*"}
