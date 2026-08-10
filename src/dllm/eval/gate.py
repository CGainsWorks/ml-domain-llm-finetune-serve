"""The promotion gate: turn a scorecard into a deploy / do-not-deploy decision.

Three independent kinds of check, all of which must pass:

``min_absolute``
    Hard floors. "Whatever the baseline was, do not serve a model that produces
    invalid SQL more than 2% of the time."

``min_delta_vs_baseline``
    The fine-tune must actually beat the base model by a stated margin.
    Requires a baseline scorecard; without one the gate fails rather than
    silently skipping the check.

``max_regression``
    A metric may not fall by more than this. This is the catastrophic-
    forgetting guard: a model can be better at SQL and still be blocked for
    having stopped answering general questions in prose.

The gate also refuses to compare scorecards that are not comparable -- a
different prompt version or a different number of eval items makes a delta
meaningless, and reporting it anyway is how false improvements get shipped.

``decide()`` is pure. The CLI wrapper turns its result into an exit code so CI
can block on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from dllm.config import ThresholdConfig
from dllm.eval.scorecard import Scorecard

CheckStatus = Literal["pass", "fail"]


@dataclass
class Check:
    kind: str
    metric: str
    threshold: float
    observed: float | None
    baseline: float | None
    status: CheckStatus
    message: str

    @property
    def passed(self) -> bool:
        return self.status == "pass"


@dataclass
class GateDecision:
    promote: bool
    checks: list[Check] = field(default_factory=list)
    blocking_errors: list[str] = field(default_factory=list)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def as_dict(self) -> dict:
        return {
            "promote": self.promote,
            "blocking_errors": self.blocking_errors,
            "checks": [c.__dict__ for c in self.checks],
        }

    def render(self) -> str:
        lines = ["# Promotion gate", ""]
        lines.append(f"**Decision: {'PROMOTE' if self.promote else 'BLOCKED'}**")
        lines.append("")
        if self.blocking_errors:
            lines += ["## Blocking errors", ""]
            lines += [f"- {e}" for e in self.blocking_errors]
            lines.append("")
        if self.checks:
            lines += [
                "## Checks",
                "",
                "| | Check | Metric | Threshold | Baseline | Observed |",
                "|:--:|---|---|---:|---:|---:|",
            ]
            for c in self.checks:
                mark = "✅" if c.passed else "❌"
                base = "—" if c.baseline is None else f"{c.baseline:.4f}"
                obs = "—" if c.observed is None else f"{c.observed:.4f}"
                lines.append(
                    f"| {mark} | {c.kind} | `{c.metric}` | {c.threshold:+.4f} | {base} | {obs} |"
                )
            lines.append("")
        if self.failures:
            lines += ["## Why it was blocked", ""]
            lines += [f"- {c.message}" for c in self.failures]
            lines.append("")
        return "\n".join(lines)


def decide(
    candidate: Scorecard,
    thresholds: ThresholdConfig,
    baseline: Scorecard | None = None,
    *,
    require_baseline: bool = True,
) -> GateDecision:
    """Evaluate every configured threshold against ``candidate``."""
    checks: list[Check] = []
    blocking: list[str] = []

    needs_baseline = bool(thresholds.min_delta_vs_baseline or thresholds.max_regression)

    if needs_baseline and baseline is None:
        if require_baseline:
            blocking.append(
                "no baseline scorecard: delta and regression thresholds are configured "
                "but there is nothing to compare against. Run the baseline eval first."
            )
    elif baseline is not None:
        comparable, problems = candidate.comparable_with(baseline)
        if not comparable:
            blocking.extend(f"not comparable with baseline: {p}" for p in problems)

    # --- absolute floors ---------------------------------------------------- #
    for metric, floor in sorted(thresholds.min_absolute.items()):
        observed = candidate.get(metric)
        if observed is None:
            checks.append(
                Check(
                    "min_absolute",
                    metric,
                    floor,
                    None,
                    None,
                    "fail",
                    f"{metric} is not present in the candidate scorecard",
                )
            )
            continue
        ok = observed >= floor
        checks.append(
            Check(
                "min_absolute",
                metric,
                floor,
                observed,
                None,
                "pass" if ok else "fail",
                f"{metric} = {observed:.4f}, needs >= {floor:.4f}",
            )
        )

    # --- improvement over baseline ------------------------------------------ #
    for metric, min_delta in sorted(thresholds.min_delta_vs_baseline.items()):
        observed = candidate.get(metric)
        base = baseline.get(metric) if baseline else None
        if observed is None or base is None:
            checks.append(
                Check(
                    "min_delta",
                    metric,
                    min_delta,
                    observed,
                    base,
                    "fail",
                    f"{metric} missing from candidate or baseline; cannot compute a delta",
                )
            )
            continue
        delta = observed - base
        ok = delta >= min_delta
        checks.append(
            Check(
                "min_delta",
                metric,
                min_delta,
                observed,
                base,
                "pass" if ok else "fail",
                f"{metric} moved {delta:+.4f} ({base:.4f} -> {observed:.4f}), "
                f"needs at least {min_delta:+.4f}",
            )
        )

    # --- regression guards --------------------------------------------------- #
    for metric, max_drop in sorted(thresholds.max_regression.items()):
        observed = candidate.get(metric)
        base = baseline.get(metric) if baseline else None
        if observed is None or base is None:
            checks.append(
                Check(
                    "max_regression",
                    metric,
                    max_drop,
                    observed,
                    base,
                    "fail",
                    f"{metric} missing from candidate or baseline; cannot check regression",
                )
            )
            continue
        drop = base - observed
        ok = drop <= max_drop
        checks.append(
            Check(
                "max_regression",
                metric,
                max_drop,
                observed,
                base,
                "pass" if ok else "fail",
                f"{metric} fell by {drop:.4f} ({base:.4f} -> {observed:.4f}), "
                f"maximum tolerated is {max_drop:.4f}",
            )
        )

    if not checks and not blocking:
        blocking.append(
            "no thresholds configured: the gate would approve anything. "
            "Set eval.thresholds in the config."
        )

    promote = not blocking and all(c.passed for c in checks)
    return GateDecision(promote=promote, checks=checks, blocking_errors=blocking)
