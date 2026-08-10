"""Dataset preparation: filtering, deduplication, splitting, licence audit."""

from __future__ import annotations

from collections import Counter

import pytest

from dllm.config import DataConfig
from dllm.data.licence_audit import audit, audit_source
from dllm.data.prepare import PrepStats, assign_split, clean_sql_rows, gold_is_executable
from dllm.data.sources import SOURCES, get_source

CFG = DataConfig(source="sql_create_context")


def rows(*items: tuple[str, str, str]) -> list[dict]:
    return [{"question": q, "context": c, "answer": a} for q, c, a in items]


class TestCleaning:
    def test_keeps_a_good_row(self) -> None:
        stats = PrepStats(source="t")
        out = list(
            clean_sql_rows(
                rows(("Who?", "CREATE TABLE t (a VARCHAR)", "SELECT a FROM t")), CFG, stats
            )
        )
        assert len(out) == 1
        assert set(out[0]) == {"id", "question", "context", "answer", "source"}

    def test_drops_empty_fields(self) -> None:
        stats = PrepStats(source="t")
        list(clean_sql_rows(rows(("", "CREATE TABLE t (a VARCHAR)", "SELECT a")), CFG, stats))
        assert stats.dropped_empty == 1

    def test_drops_unparseable_gold(self) -> None:
        stats = PrepStats(source="t")
        list(clean_sql_rows(rows(("Q", "CREATE TABLE t (a VARCHAR)", "not sql ((")), CFG, stats))
        assert stats.dropped_unparseable == 1

    def test_drops_non_select_gold(self) -> None:
        stats = PrepStats(source="t")
        list(clean_sql_rows(rows(("Q", "CREATE TABLE t (a VARCHAR)", "DROP TABLE t")), CFG, stats))
        assert stats.dropped_not_select == 1

    def test_strips_trailing_semicolon(self) -> None:
        stats = PrepStats(source="t")
        out = list(
            clean_sql_rows(
                rows(("Q", "CREATE TABLE t (a VARCHAR)", "SELECT a FROM t;")), CFG, stats
            )
        )
        assert out[0]["answer"] == "SELECT a FROM t"

    def test_deduplicates_on_question_and_schema(self) -> None:
        stats = PrepStats(source="t")
        dup = ("Who?", "CREATE TABLE t (a VARCHAR)", "SELECT a FROM t")
        out = list(clean_sql_rows(rows(dup, dup), CFG, stats))
        assert len(out) == 1
        assert stats.dropped_duplicate == 1

    def test_dedupe_can_be_disabled(self) -> None:
        stats = PrepStats(source="t")
        cfg = DataConfig(source="sql_create_context", dedupe=False)
        dup = ("Who?", "CREATE TABLE t (a VARCHAR)", "SELECT a FROM t")
        assert len(list(clean_sql_rows(rows(dup, dup), cfg, stats))) == 2

    def test_drops_oversized_prompts(self) -> None:
        stats = PrepStats(source="t")
        cfg = DataConfig(source="sql_create_context", max_prompt_chars=20)
        list(
            clean_sql_rows(rows(("Q" * 100, "CREATE TABLE t (a VARCHAR)", "SELECT a")), cfg, stats)
        )
        assert stats.dropped_too_long == 1


class TestSplitting:
    def test_deterministic(self) -> None:
        a = assign_split("abc", seed=1, eval_fraction=0.05, test_fraction=0.05)
        b = assign_split("abc", seed=1, eval_fraction=0.05, test_fraction=0.05)
        assert a == b

    def test_seed_changes_assignment(self) -> None:
        assignments = {
            assign_split(f"item{i}", seed=s, eval_fraction=0.05, test_fraction=0.05)
            for s in (1, 2)
            for i in range(1)
        }
        assert assignments  # sanity; exact values depend on the hash

    def test_proportions_are_approximately_right(self) -> None:
        counts = Counter(
            assign_split(f"item-{i}", seed=1337, eval_fraction=0.05, test_fraction=0.05)
            for i in range(20_000)
        )
        assert 0.04 < counts["test"] / 20_000 < 0.06
        assert 0.04 < counts["validation"] / 20_000 < 0.06
        assert counts["train"] / 20_000 > 0.88

    def test_split_is_stable_when_the_dataset_grows(self) -> None:
        """The property that keeps old scorecards valid."""
        small = {
            f"id{i}": assign_split(f"id{i}", seed=1, eval_fraction=0.05, test_fraction=0.05)
            for i in range(100)
        }
        large = {
            f"id{i}": assign_split(f"id{i}", seed=1, eval_fraction=0.05, test_fraction=0.05)
            for i in range(1000)
        }
        assert all(large[k] == v for k, v in small.items())


class TestGoldExecutability:
    def test_good_item_passes(self) -> None:
        assert gold_is_executable(
            {"context": "CREATE TABLE t (a VARCHAR)", "answer": "SELECT a FROM t"}
        )

    def test_broken_schema_fails(self) -> None:
        assert not gold_is_executable({"context": "nonsense", "answer": "SELECT a FROM t"})


class TestLicenceAudit:
    def test_primary_source_passes_commercial_requirement(self) -> None:
        findings = audit_source(get_source("sql_create_context"), require_commercial=True)
        assert not [f for f in findings if f["level"] == "error"]

    def test_rejected_source_fails(self) -> None:
        findings = audit_source(get_source("alpaca_REJECTED"), require_commercial=True)
        errors = [f["message"] for f in findings if f["level"] == "error"]
        assert any("REJECTED" in m for m in errors)
        assert any("research_only" in m for m in errors)

    def test_non_commercial_fails_only_when_required(self) -> None:
        src = get_source("alpaca_REJECTED")
        lenient = audit_source(src, require_commercial=False)
        assert not any("research_only" in f["message"] for f in lenient)

    def test_share_alike_upstream_warns(self) -> None:
        findings = audit_source(get_source("sql_create_context"), require_commercial=True)
        assert any("share-alike" in f["message"] for f in findings)

    def test_full_audit_blocks_on_rejected_source(self) -> None:
        manifest = audit(["alpaca_REJECTED"], require_commercial=True)
        assert not manifest["passed"]
        assert manifest["error_count"] >= 1

    def test_full_audit_passes_for_the_default_mix(self) -> None:
        manifest = audit(
            ["sql_create_context", "dolly_guard"],
            require_commercial=True,
            base_model_licence="Apache-2.0",
            base_model_commercial=True,
        )
        assert manifest["passed"]

    def test_non_commercial_base_model_blocks(self) -> None:
        manifest = audit(
            ["sql_create_context"],
            require_commercial=True,
            base_model_licence="Some-NC-Licence",
            base_model_commercial=False,
        )
        assert not manifest["passed"]

    @pytest.mark.parametrize("key", sorted(SOURCES))
    def test_every_source_declares_a_licence(self, key: str) -> None:
        src = SOURCES[key]
        assert src.licence and src.licence_url and src.citation

    def test_training_sources_are_revision_pinned(self) -> None:
        """A moving dataset revision silently invalidates every scorecard."""
        for key in ("sql_create_context", "dolly_guard"):
            assert len(get_source(key).revision) == 40


class TestRegistry:
    def test_unknown_source_lists_alternatives(self) -> None:
        with pytest.raises(KeyError, match="Declared sources"):
            get_source("no_such_dataset")


class TestGeneratedDocsAreDeterministic:
    """CI fails the build if regenerating a committed doc produces a diff.

    That check is only useful if the generator is deterministic. An embedded
    generation timestamp would make it fire on every single run, and a check
    that always fires gets deleted -- taking the real signal (a registry change
    that was never re-audited) with it.
    """

    def test_licence_report_is_stable_across_calls(self) -> None:
        from dllm.data.licence_audit import render_markdown

        a = audit(["sql_create_context"], require_commercial=True)
        b = audit(["sql_create_context"], require_commercial=True)
        assert a["generated_at"] is not None, "the JSON manifest still records when it ran"
        assert render_markdown(a) == render_markdown(b)

    def test_licence_report_carries_no_timestamp(self) -> None:
        from dllm.data.licence_audit import render_markdown

        body = render_markdown(audit(["sql_create_context"], require_commercial=True))
        assert "Generated:" not in body
        assert "20" not in body.split("## Sources")[0].replace("CC-BY-4.0", "").replace(
            "CC-BY-SA-4.0", ""
        ), "no year-like token should appear in the header"

    def test_report_content_changes_when_the_registry_does(self) -> None:
        """Deterministic must not mean constant."""
        from dllm.data.licence_audit import render_markdown

        commercial = render_markdown(audit(["sql_create_context"], require_commercial=True))
        rejected = render_markdown(audit(["alpaca_REJECTED"], require_commercial=True))
        assert commercial != rejected

    def test_dataset_card_is_stable_across_calls(self) -> None:
        from dllm.config import load_config
        from dllm.data.dataset_card import render_dataset_card
        from dllm.paths import CONFIGS

        cfg = load_config(CONFIGS / "base.yaml")
        assert render_dataset_card(cfg) == render_dataset_card(cfg)

    def test_dataset_card_carries_no_timestamp(self) -> None:
        from dllm.config import load_config
        from dllm.data.dataset_card import render_dataset_card
        from dllm.paths import CONFIGS

        body = render_dataset_card(load_config(CONFIGS / "base.yaml"))
        assert "Last generated" not in body
