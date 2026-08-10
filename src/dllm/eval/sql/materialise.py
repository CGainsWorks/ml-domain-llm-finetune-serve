"""Turn a bare ``CREATE TABLE`` string into a populated in-memory SQLite database.

Why this exists
---------------
``sql-create-context`` gives you a schema and a gold query but no rows. Without
rows you can only compare query *text*, and text comparison punishes every
correct-but-differently-written query (``WHERE a = 1 AND b = 2`` vs
``WHERE b = 2 AND a = 1``). Executing both queries and comparing result sets is
the metric that actually tracks correctness.

Populating the table randomly does not work: a query like
``WHERE player = 'Jared Butler'`` matches nothing, gold and prediction both
return the empty set, and every wrong answer scores as correct. So rows are
seeded from the literals in the *gold* query, guaranteeing that the gold
predicate selects a proper subset of the table.

Three details that took some care:

* **Column affinity and collation are chosen to match MySQL, not SQLite.**
  The source data is MySQL-flavoured, and MySQL's default collation is
  case-insensitive with loose numeric/string comparison. SQLite's defaults are
  neither. Without this, a model that answers ``WHERE role = 'Taylor'`` against
  a gold of ``WHERE role = "taylor"`` scores as wrong, as does ``year = '2003'``
  against ``year = 2003`` — differences of literal casing and quoting, not of
  SQL. Measured on the base model, these artefacts accounted for a large share
  of apparent failures, and fine-tuning would have "fixed" them by teaching the
  model the dataset's lowercasing convention. That would have shown up as a
  large improvement in SQL ability that was nothing of the sort.

  So text columns are declared ``TEXT COLLATE NOCASE`` and numeric columns
  ``NUMERIC``. Both then compare correctly against either literal form.
* **Filler values depend on the column name, not the table.** Two tables that
  share a column name get the same value series, so ``JOIN t2 ON t1.id = t2.id``
  actually matches rows instead of returning nothing.
* **Numeric spread.** ``MAX``/``MIN``/``ORDER BY`` need distinct values to
  discriminate a right answer from a wrong one, so fillers are never constant.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from typing import Any

import sqlglot
from sqlglot import expressions as exp

from dllm.eval.sql.normalise import READ_DIALECT, WRITE_DIALECT, SqlParseFailure

DEFAULT_FILLER_ROWS = 9
DEFAULT_ANCHOR_ROWS = 3


class SchemaError(ValueError):
    """The CREATE TABLE context could not be understood."""


@dataclass(frozen=True)
class Column:
    name: str
    declared_type: str = ""

    @property
    def is_numeric(self) -> bool:
        t = self.declared_type.upper()
        return any(k in t for k in ("INT", "REAL", "FLOAT", "DOUBLE", "DEC", "NUMERIC", "YEAR"))


@dataclass
class Table:
    name: str
    columns: list[Column] = field(default_factory=list)


@dataclass
class Schema:
    tables: dict[str, Table] = field(default_factory=dict)

    def column_names(self) -> set[str]:
        return {c.name.lower() for t in self.tables.values() for c in t.columns}

    def table_names(self) -> set[str]:
        return set(self.tables)


# --------------------------------------------------------------------------- #
# Parsing the schema
# --------------------------------------------------------------------------- #
def parse_schema(context: str) -> Schema:
    """Parse one or more ``CREATE TABLE`` statements into a :class:`Schema`."""
    if not context or not context.strip():
        raise SchemaError("empty schema context")

    try:
        statements = sqlglot.parse(context, read=READ_DIALECT)
    except Exception as e:
        # sqlglot raises several unrelated types; re-raised as SchemaError so
        # callers have one thing to catch.
        raise SchemaError(f"could not parse schema: {e}") from e

    schema = Schema()
    for stmt in statements:
        if not isinstance(stmt, exp.Create):
            continue
        target = stmt.this
        if not isinstance(target, exp.Schema):
            continue

        table_expr = target.this
        table_name = table_expr.name if isinstance(table_expr, exp.Table) else None
        if not table_name:
            continue

        columns: list[Column] = []
        for col_def in target.expressions:
            if not isinstance(col_def, exp.ColumnDef):
                continue
            kind = col_def.args.get("kind")
            columns.append(
                Column(
                    name=col_def.name, declared_type=kind.sql(dialect=WRITE_DIALECT) if kind else ""
                )
            )

        if columns:
            schema.tables[table_name.lower()] = Table(name=table_name, columns=columns)

    if not schema.tables:
        raise SchemaError("no CREATE TABLE statement with columns found in context")
    return schema


# --------------------------------------------------------------------------- #
# Pulling literals out of the gold query
# --------------------------------------------------------------------------- #
def _literal_value(node: exp.Expression) -> Any | None:
    """Convert a sqlglot literal node to a Python value, or None."""
    if isinstance(node, exp.Literal):
        if node.is_string:
            return str(node.this)
        text = str(node.this)
        try:
            return int(text)
        except ValueError:
            try:
                return float(text)
            except ValueError:
                return text
    if isinstance(node, exp.Boolean):
        return 1 if node.this else 0
    if isinstance(node, exp.Neg):
        inner = _literal_value(node.this)
        if isinstance(inner, int | float):
            return -inner
    return None


def extract_literals(query: str) -> dict[str, list[Any]]:
    """Map ``column name -> literal values compared against it`` in a query.

    Covers ``=``, ``!=``, the ordering operators, ``LIKE``, ``IN`` and
    ``BETWEEN``. Ordering operators contribute the boundary value itself, which
    is enough to guarantee the predicate is not vacuous.
    """
    try:
        expr = sqlglot.parse_one(query, read=READ_DIALECT)
    except Exception:  # noqa: BLE001
        return {}
    if expr is None:
        return {}

    out: dict[str, list[Any]] = {}

    def add(col: exp.Expression, val: exp.Expression | None) -> None:
        if not isinstance(col, exp.Column) or val is None:
            return
        value = _literal_value(val)
        if value is None:
            return
        if isinstance(value, str):
            value = value.strip("%")  # LIKE patterns
            if not value:
                return
        out.setdefault(col.name.lower(), []).append(value)

    binary_ops = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like, exp.ILike)
    for node in expr.walk():
        if isinstance(node, binary_ops):
            add(node.this, node.expression)
            add(node.expression, node.this)
        elif isinstance(node, exp.In):
            for item in node.expressions or []:
                add(node.this, item)
        elif isinstance(node, exp.Between):
            add(node.this, node.args.get("low"))
            add(node.this, node.args.get("high"))

    # Deduplicate, preserving order.
    return {k: list(dict.fromkeys(v)) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# Deterministic filler values
# --------------------------------------------------------------------------- #
def _seed_for(name: str) -> int:
    return int(hashlib.sha256(name.lower().encode()).hexdigest()[:8], 16)


def filler_value(column: Column, row: int, numeric: bool) -> Any:
    """A deterministic value for ``column`` at ``row``.

    Depends only on the column *name*, so the same column in two tables gets
    the same series and joins on it produce matches.
    """
    seed = _seed_for(column.name)
    if numeric:
        base = seed % 50
        step = 1 + (seed % 7)
        return base + row * step
    return f"{column.name.lower()}_{(seed + row) % 97}"


def _column_is_numeric(column: Column, literals: list[Any]) -> bool:
    """Prefer evidence from the gold query over the declared type.

    ``sql-create-context`` declares almost everything ``VARCHAR``, including
    columns the gold query compares against bare integers.
    """
    if literals:
        return all(isinstance(v, int | float) for v in literals)
    return column.is_numeric


# --------------------------------------------------------------------------- #
# Building the database
# --------------------------------------------------------------------------- #
def build_database(
    schema: Schema,
    literals: dict[str, list[Any]] | None = None,
    *,
    anchor_rows: int = DEFAULT_ANCHOR_ROWS,
    filler_rows: int = DEFAULT_FILLER_ROWS,
) -> sqlite3.Connection:
    """Create and populate an in-memory database for ``schema``.

    ``literals`` seeds the anchor rows so that the gold predicate matches
    something. Everything after that is deterministic filler.
    """
    literals = literals or {}
    conn = sqlite3.connect(":memory:")
    conn.row_factory = None

    for table in schema.tables.values():
        numeric = {
            c.name.lower(): _column_is_numeric(c, literals.get(c.name.lower(), []))
            for c in table.columns
        }

        # Affinity and collation chosen to reproduce MySQL comparison semantics
        # -- see the module docstring. This is the difference between measuring
        # SQL ability and measuring whether the model guessed the dataset's
        # capitalisation convention.
        cols = ", ".join(
            f'"{c.name}" NUMERIC' if numeric[c.name.lower()] else f'"{c.name}" TEXT COLLATE NOCASE'
            for c in table.columns
        )
        conn.execute(f'CREATE TABLE "{table.name}" ({cols})')

        rows: list[tuple[Any, ...]] = []

        # Anchor rows: every column that the gold query constrains gets one of
        # the values it was compared against, cycling if there are several.
        for i in range(anchor_rows):
            row: list[Any] = []
            for c in table.columns:
                vals = literals.get(c.name.lower(), [])
                if vals:
                    row.append(vals[i % len(vals)])
                else:
                    row.append(filler_value(c, i, numeric[c.name.lower()]))
            rows.append(tuple(row))

        # Filler rows, offset past the anchors so values stay distinct.
        for i in range(filler_rows):
            rows.append(
                tuple(
                    filler_value(c, anchor_rows + i, numeric[c.name.lower()]) for c in table.columns
                )
            )

        placeholders = ", ".join("?" * len(table.columns))
        conn.executemany(f'INSERT INTO "{table.name}" VALUES ({placeholders})', rows)

    conn.commit()
    return conn


def build_for_item(context: str, gold_sql: str, **kwargs: Any) -> tuple[sqlite3.Connection, Schema]:
    """Convenience: parse the schema, seed from the gold query, return both."""
    schema = parse_schema(context)
    try:
        literals = extract_literals(gold_sql)
    except SqlParseFailure:
        literals = {}
    return build_database(schema, literals, **kwargs), schema
