"""SQL parsing, cleaning and canonicalisation."""

from __future__ import annotations

import pytest

from dllm.eval.sql.normalise import (
    SqlParseFailure,
    exact_match,
    is_read_only,
    normalise,
    parse,
    referenced_columns,
    referenced_tables,
    strip_markdown,
    to_sqlite,
)


class TestStripMarkdown:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("SELECT 1", "SELECT 1"),
            ("```sql\nSELECT 1\n```", "SELECT 1"),
            ("```\nSELECT 1\n```", "SELECT 1"),
            ("Here is the query:\nSELECT 1", "SELECT 1"),
            ("SELECT 1;", "SELECT 1"),
            ("  SELECT 1  ", "SELECT 1"),
        ],
    )
    def test_strips_decoration(self, raw: str, expected: str) -> None:
        assert strip_markdown(raw) == expected

    def test_cuts_trailing_explanation(self) -> None:
        raw = "SELECT a FROM t\n\nThis query selects column a."
        assert strip_markdown(raw) == "SELECT a FROM t"

    def test_keeps_with_cte(self) -> None:
        assert strip_markdown("WITH x AS (SELECT 1) SELECT * FROM x").startswith("WITH")


class TestParse:
    def test_rejects_empty(self) -> None:
        with pytest.raises(SqlParseFailure):
            parse("   ")

    def test_rejects_gibberish(self) -> None:
        with pytest.raises(SqlParseFailure):
            parse("SELECT FROM WHERE ORDER GROUP )(")


class TestReadOnly:
    @pytest.mark.parametrize(
        "sql",
        ["SELECT 1", "SELECT a FROM t", "SELECT a FROM t UNION SELECT b FROM u"],
    )
    def test_select_shapes_allowed(self, sql: str) -> None:
        assert is_read_only(parse(sql))

    @pytest.mark.parametrize(
        "sql",
        [
            "DROP TABLE t",
            "DELETE FROM t",
            "UPDATE t SET a = 1",
            "INSERT INTO t VALUES (1)",
            "CREATE TABLE t (a INT)",
        ],
    )
    def test_mutations_rejected(self, sql: str) -> None:
        assert not is_read_only(parse(sql))


class TestDialect:
    def test_double_quotes_become_string_literals(self) -> None:
        """The trap: SQLite reads "x" as an identifier, MySQL as a string."""
        out = to_sqlite('SELECT a FROM t WHERE b = "hello"')
        assert "'hello'" in out
        assert '"hello"' not in out


class TestNormalise:
    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("SELECT COUNT(*) FROM t", "select count( * ) from t"),
            ('SELECT a FROM t WHERE b = "x"', "SELECT a FROM t WHERE b = 'x'"),
            ("SELECT a FROM t;", "SELECT a FROM t"),
            ("SELECT   a   FROM   t", "SELECT a FROM t"),
        ],
    )
    def test_cosmetic_differences_collapse(self, a: str, b: str) -> None:
        assert exact_match(a, b)

    def test_semantic_differences_survive(self) -> None:
        assert not exact_match("SELECT a FROM t", "SELECT b FROM t")

    def test_unparseable_degrades_instead_of_raising(self) -> None:
        assert normalise("not sql at all ((") != ""


class TestReferences:
    def test_tables_and_columns(self) -> None:
        expr = parse("SELECT a.name FROM alpha AS a JOIN beta AS b ON a.id = b.id")
        assert referenced_tables(expr) == {"alpha", "beta"}
        assert referenced_columns(expr) == {"name", "id"}

    def test_cte_name_is_not_a_table(self) -> None:
        expr = parse("WITH tmp AS (SELECT a FROM real) SELECT a FROM tmp")
        assert referenced_tables(expr) == {"real"}
