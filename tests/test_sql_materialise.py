"""Schema parsing and synthetic database construction."""

from __future__ import annotations

import pytest

from dllm.eval.sql.materialise import (
    Column,
    SchemaError,
    build_database,
    build_for_item,
    extract_literals,
    filler_value,
    parse_schema,
)


class TestParseSchema:
    def test_single_table(self) -> None:
        schema = parse_schema("CREATE TABLE t (a VARCHAR, b INTEGER)")
        assert schema.table_names() == {"t"}
        assert schema.column_names() == {"a", "b"}

    def test_multiple_tables(self) -> None:
        schema = parse_schema("CREATE TABLE a (x VARCHAR); CREATE TABLE b (y INTEGER)")
        assert schema.table_names() == {"a", "b"}

    def test_rejects_empty(self) -> None:
        with pytest.raises(SchemaError):
            parse_schema("")

    def test_rejects_non_create(self) -> None:
        with pytest.raises(SchemaError):
            parse_schema("SELECT 1")


class TestExtractLiterals:
    def test_equality(self) -> None:
        assert extract_literals('SELECT a FROM t WHERE b = "x"') == {"b": ["x"]}

    def test_numeric(self) -> None:
        assert extract_literals("SELECT a FROM t WHERE year = 1994") == {"year": [1994]}

    def test_comparison_boundary(self) -> None:
        assert extract_literals("SELECT a FROM t WHERE score > 10") == {"score": [10]}

    def test_in_list(self) -> None:
        got = extract_literals("SELECT a FROM t WHERE b IN ('x', 'y')")
        assert got == {"b": ["x", "y"]}

    def test_between(self) -> None:
        got = extract_literals("SELECT a FROM t WHERE n BETWEEN 1 AND 5")
        assert got == {"n": [1, 5]}

    def test_like_strips_wildcards(self) -> None:
        assert extract_literals("SELECT a FROM t WHERE b LIKE '%foo%'") == {"b": ["foo"]}

    def test_multiple_columns(self) -> None:
        got = extract_literals('SELECT c FROM t WHERE a = "x" AND b = 2')
        assert got == {"a": ["x"], "b": [2]}

    def test_unparseable_returns_empty(self) -> None:
        assert extract_literals("))(( not sql") == {}


class TestFillerValues:
    def test_deterministic(self) -> None:
        col = Column("name", "VARCHAR")
        assert filler_value(col, 3, numeric=False) == filler_value(col, 3, numeric=False)

    def test_depends_on_name_not_table(self) -> None:
        """Two tables sharing a column name must share values, or joins never match."""
        a = filler_value(Column("id", "VARCHAR"), 0, numeric=True)
        b = filler_value(Column("id", "INTEGER"), 0, numeric=True)
        assert a == b

    def test_varies_by_row(self) -> None:
        col = Column("score", "INTEGER")
        values = {filler_value(col, i, numeric=True) for i in range(6)}
        assert len(values) == 6, "constant fillers would make MAX/MIN undiscriminating"


class TestBuildDatabase:
    def test_row_count(self) -> None:
        schema = parse_schema("CREATE TABLE t (a VARCHAR)")
        conn = build_database(schema, {}, anchor_rows=3, filler_rows=9)
        assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 12

    def test_anchor_rows_carry_the_literal(self) -> None:
        schema = parse_schema("CREATE TABLE t (a VARCHAR, b VARCHAR)")
        conn = build_database(schema, {"b": ["target"]}, anchor_rows=3, filler_rows=9)
        matched = conn.execute("SELECT COUNT(*) FROM t WHERE b = 'target'").fetchone()[0]
        assert matched == 3, "the gold predicate must select a proper subset, not nothing"

    def test_numeric_literal_stays_numeric(self) -> None:
        """SQLite affinity would coerce 1994 to '1994' if the column were typed."""
        schema = parse_schema("CREATE TABLE t (year VARCHAR)")
        conn = build_database(schema, {"year": [1994]}, anchor_rows=2, filler_rows=4)
        assert conn.execute("SELECT COUNT(*) FROM t WHERE year = 1994").fetchone()[0] == 2

    def test_join_on_shared_column_matches(self) -> None:
        schema = parse_schema("CREATE TABLE a (id VARCHAR, x VARCHAR); CREATE TABLE b (id VARCHAR)")
        conn = build_database(schema, {}, anchor_rows=0, filler_rows=6)
        joined = conn.execute("SELECT COUNT(*) FROM a JOIN b ON a.id = b.id").fetchone()[0]
        assert joined > 0

    def test_build_for_item_seeds_from_gold(self) -> None:
        conn, schema = build_for_item(
            "CREATE TABLE t (a VARCHAR, b VARCHAR)",
            'SELECT a FROM t WHERE b = "seed"',
        )
        assert schema.table_names() == {"t"}
        rows = conn.execute("SELECT a FROM t WHERE b = 'seed'").fetchall()
        assert rows
