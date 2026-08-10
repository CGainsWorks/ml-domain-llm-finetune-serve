"""The scorecard: one JSON file that fully describes an evaluation run.

Everything needed to decide whether two numbers are comparable lives in the
file itself -- prompt version, task ids, item counts, dataset fingerprint,
generation settings, git commit. Comparing a baseline against a fine-tune when
the prompt version changed underneath is the classic way to report an
improvement that is not real, so the gate refuses to do it.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dllm.eval.tasks.base import TaskResult
from dllm.prompts import GUARD_PROMPT_VERSION, SQL_PROMPT_VERSION

SCORECARD_SCHEMA_VERSION = 1


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


@dataclass
class Scorecard:
    label: str
    created_at: str
    tasks: dict[str, dict[str, Any]]
    metrics: dict[str, float]
    backend: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)
    prompt_versions: dict[str, str] = field(default_factory=dict)
    dataset_fingerprint: dict[str, Any] = field(default_factory=dict)
    runtime_s: float = 0.0
    git_commit: str = ""
    env: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCORECARD_SCHEMA_VERSION

    # ------------------------------------------------------------------ #
    @classmethod
    def build(
        cls,
        label: str,
        results: list[TaskResult],
        *,
        backend: dict[str, Any],
        generation: dict[str, Any],
        dataset_fingerprint: dict[str, Any] | None = None,
        runtime_s: float = 0.0,
    ) -> Scorecard:
        tasks = {r.task_id: r.as_dict() for r in results}
        metrics = {
            f"{r.task_id}.{name}": value for r in results for name, value in r.metrics.items()
        }
        return cls(
            label=label,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            tasks=tasks,
            metrics=metrics,
            backend=backend,
            generation=generation,
            prompt_versions={
                "sql": SQL_PROMPT_VERSION,
                "guard": GUARD_PROMPT_VERSION,
            },
            dataset_fingerprint=dataset_fingerprint or {},
            runtime_s=round(runtime_s, 2),
            git_commit=git_commit(),
            env={
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
        )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Scorecard:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data.pop("schema_version", None)
        return cls(**data, schema_version=SCORECARD_SCHEMA_VERSION)

    # ------------------------------------------------------------------ #
    def get(self, key: str) -> float | None:
        return self.metrics.get(key)

    def comparable_with(self, other: Scorecard) -> tuple[bool, list[str]]:
        """Whether two scorecards may legitimately be compared.

        Different prompts, different tasks or different item counts all make a
        delta meaningless. Returning the reasons rather than a bare bool means
        the gate can explain itself.
        """
        problems: list[str] = []
        if self.prompt_versions != other.prompt_versions:
            problems.append(
                f"prompt versions differ: {other.prompt_versions} vs {self.prompt_versions}"
            )
        mine, theirs = set(self.tasks), set(other.tasks)
        if mine != theirs:
            missing = theirs - mine
            extra = mine - theirs
            problems.append(f"task sets differ (missing={sorted(missing)}, extra={sorted(extra)})")
        for task in mine & theirs:
            a, b = self.tasks[task]["n"], other.tasks[task]["n"]
            if a != b:
                problems.append(f"task {task}: {b} items in baseline vs {a} here")
        return (not problems, problems)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
_HIGHER_IS_WORSE = {"exec_error_rate", "unparseable_rate"}


def _arrow(metric: str, delta: float) -> str:
    if abs(delta) < 1e-9:
        return "="
    good = delta < 0 if metric.split(".")[-1] in _HIGHER_IS_WORSE else delta > 0
    return "▲" if good else "▼"


def render_comparison(baseline: Scorecard, candidate: Scorecard) -> str:
    """Markdown scorecard: base vs fine-tuned, with deltas."""
    lines = [
        f"## Scorecard: `{baseline.label}` vs `{candidate.label}`",
        "",
        f"- Baseline run: `{baseline.created_at}` (commit `{baseline.git_commit}`)",
        f"- Candidate run: `{candidate.created_at}` (commit `{candidate.git_commit}`)",
        f"- Prompt versions: `{candidate.prompt_versions}`",
        "",
        "| Task | Metric | Base | Fine-tuned | Δ | |",
        "|---|---|---:|---:|---:|:--:|",
    ]
    for key in sorted(set(baseline.metrics) | set(candidate.metrics)):
        task, _, metric = key.partition(".")
        b = baseline.metrics.get(key)
        c = candidate.metrics.get(key)
        if b is None or c is None:
            lines.append(
                f"| `{task}` | `{metric}` | "
                f"{'—' if b is None else f'{b:.4f}'} | "
                f"{'—' if c is None else f'{c:.4f}'} | — | |"
            )
            continue
        delta = c - b
        lines.append(
            f"| `{task}` | `{metric}` | {b:.4f} | {c:.4f} | {delta:+.4f} | {_arrow(key, delta)} |"
        )

    lines += [
        "",
        "### Item counts",
        "",
        "| Task | n | scorable | gold unexecutable |",
        "|---|---:|---:|---:|",
    ]
    for task, payload in sorted(candidate.tasks.items()):
        counts = payload.get("counts", {})
        lines.append(
            f"| `{task}` | {payload['n']} | {counts.get('scorable', '—')} | "
            f"{counts.get('gold_unexecutable', '—')} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_single(card: Scorecard) -> str:
    """Markdown for one scorecard with no comparison."""
    lines = [
        f"## Scorecard: `{card.label}`",
        "",
        f"- Run: `{card.created_at}` (commit `{card.git_commit}`)",
        f"- Backend: `{card.backend.get('backend')}` — `{card.backend.get('model_path') or card.backend.get('model')}`",
        f"- Runtime: {card.runtime_s:.1f}s",
        "",
        "| Task | Metric | Value |",
        "|---|---|---:|",
    ]
    for key in sorted(card.metrics):
        task, _, metric = key.partition(".")
        lines.append(f"| `{task}` | `{metric}` | {card.metrics[key]:.4f} |")
    lines.append("")
    return "\n".join(lines)
