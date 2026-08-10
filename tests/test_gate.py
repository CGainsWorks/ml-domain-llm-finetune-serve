"""The promotion gate. This is the code that stops a bad model shipping."""

from __future__ import annotations

from dllm.config import ThresholdConfig
from dllm.eval.gate import decide
from dllm.eval.scorecard import Scorecard


def card(
    label: str, metrics: dict[str, float], *, n: int = 100, prompt: str = "sql_v1"
) -> Scorecard:
    tasks = {
        "sql_exec_holdout": {"task_id": "sql_exec_holdout", "n": n, "metrics": {}, "counts": {}}
    }
    return Scorecard(
        label=label,
        created_at="2026-01-01T00:00:00+00:00",
        tasks=tasks,
        metrics=metrics,
        prompt_versions={"sql": prompt, "guard": "guard_v1"},
    )


BASE = card(
    "baseline",
    {
        "sql_exec_holdout.exec_acc_nonempty": 0.40,
        "sql_exec_holdout.valid_sql_rate": 0.99,
        "general_guard.answers_in_prose": 0.98,
    },
)


class TestAbsoluteFloors:
    def test_passes_when_met(self) -> None:
        t = ThresholdConfig(min_absolute={"sql_exec_holdout.valid_sql_rate": 0.98})
        d = decide(card("c", {"sql_exec_holdout.valid_sql_rate": 0.99}), t)
        assert d.promote

    def test_fails_when_below(self) -> None:
        t = ThresholdConfig(min_absolute={"sql_exec_holdout.valid_sql_rate": 0.98})
        d = decide(card("c", {"sql_exec_holdout.valid_sql_rate": 0.90}), t)
        assert not d.promote
        assert "needs >= 0.9800" in d.failures[0].message

    def test_missing_metric_fails_rather_than_skips(self) -> None:
        t = ThresholdConfig(min_absolute={"nope.metric": 0.5})
        d = decide(card("c", {}), t)
        assert not d.promote


class TestDeltaVsBaseline:
    def test_sufficient_improvement_passes(self) -> None:
        t = ThresholdConfig(min_delta_vs_baseline={"sql_exec_holdout.exec_acc_nonempty": 0.05})
        d = decide(card("c", {"sql_exec_holdout.exec_acc_nonempty": 0.50}), t, BASE)
        assert d.promote

    def test_insufficient_improvement_blocks(self) -> None:
        t = ThresholdConfig(min_delta_vs_baseline={"sql_exec_holdout.exec_acc_nonempty": 0.05})
        d = decide(card("c", {"sql_exec_holdout.exec_acc_nonempty": 0.42}), t, BASE)
        assert not d.promote

    def test_regression_blocks(self) -> None:
        t = ThresholdConfig(min_delta_vs_baseline={"sql_exec_holdout.exec_acc_nonempty": 0.05})
        d = decide(card("c", {"sql_exec_holdout.exec_acc_nonempty": 0.20}), t, BASE)
        assert not d.promote

    def test_missing_baseline_blocks(self) -> None:
        """Without a before-number the after-number means nothing."""
        t = ThresholdConfig(min_delta_vs_baseline={"sql_exec_holdout.exec_acc_nonempty": 0.05})
        d = decide(card("c", {"sql_exec_holdout.exec_acc_nonempty": 0.99}), t, None)
        assert not d.promote
        assert any("no baseline" in e for e in d.blocking_errors)


class TestRegressionGuard:
    def test_catastrophic_forgetting_blocks_promotion(self) -> None:
        """Better at SQL, but now answers everything with SELECT: must not ship."""
        t = ThresholdConfig(
            min_delta_vs_baseline={"sql_exec_holdout.exec_acc_nonempty": 0.05},
            max_regression={"general_guard.answers_in_prose": 0.10},
        )
        candidate = card(
            "c",
            {
                "sql_exec_holdout.exec_acc_nonempty": 0.85,  # much better
                "general_guard.answers_in_prose": 0.30,  # collapsed
            },
        )
        d = decide(candidate, t, BASE)
        assert not d.promote
        assert any("answers_in_prose" in c.message for c in d.failures)

    def test_small_regression_tolerated(self) -> None:
        t = ThresholdConfig(max_regression={"general_guard.answers_in_prose": 0.10})
        d = decide(card("c", {"general_guard.answers_in_prose": 0.93}), t, BASE)
        assert d.promote


class TestComparability:
    def test_prompt_version_change_blocks(self) -> None:
        """Changing the prompt underneath makes the delta meaningless."""
        t = ThresholdConfig(min_delta_vs_baseline={"sql_exec_holdout.exec_acc_nonempty": 0.05})
        candidate = card("c", {"sql_exec_holdout.exec_acc_nonempty": 0.99}, prompt="sql_v2")
        d = decide(candidate, t, BASE)
        assert not d.promote
        assert any("prompt versions differ" in e for e in d.blocking_errors)

    def test_item_count_change_blocks(self) -> None:
        t = ThresholdConfig(min_delta_vs_baseline={"sql_exec_holdout.exec_acc_nonempty": 0.05})
        candidate = card("c", {"sql_exec_holdout.exec_acc_nonempty": 0.99}, n=10)
        d = decide(candidate, t, BASE)
        assert not d.promote
        assert any("items in baseline" in e for e in d.blocking_errors)


class TestEmptyConfiguration:
    def test_no_thresholds_is_a_failure_not_a_pass(self) -> None:
        """A gate that approves anything is worse than no gate: it looks like one."""
        d = decide(card("c", {"a.b": 1.0}), ThresholdConfig())
        assert not d.promote
        assert any("no thresholds configured" in e for e in d.blocking_errors)


class TestRendering:
    def test_report_names_the_failing_check(self) -> None:
        t = ThresholdConfig(min_absolute={"sql_exec_holdout.valid_sql_rate": 0.98})
        d = decide(card("c", {"sql_exec_holdout.valid_sql_rate": 0.5}), t)
        report = d.render()
        assert "BLOCKED" in report
        assert "valid_sql_rate" in report
