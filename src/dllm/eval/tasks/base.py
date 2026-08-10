"""Task interface: load items, build prompts, score completions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dllm.paths import DATA
from dllm.prompts import Message


def resolve_dataset_path(dataset: str, root: Path) -> Path:
    """Resolve a task's ``dataset`` field to a real path.

    Task definitions name their dataset as ``data/processed/test.jsonl``, which
    is readable and matches the repository layout. But the data directory is
    relocatable via ``DLLM_DATA_DIR`` -- CI uses that so a smoke run with 64
    examples cannot overwrite the real prepared splits. A leading ``data/`` is
    therefore rewritten to point at the configured directory.
    """
    p = Path(dataset)
    if p.is_absolute():
        return p
    parts = p.parts
    if parts and parts[0] == "data":
        return DATA.joinpath(*parts[1:])
    return root / p


@dataclass
class TaskSpec:
    """Declarative task definition, loaded from ``evals/tasks/<id>.yaml``."""

    id: str
    kind: str
    dataset: str
    description: str = ""
    primary_metric: str = ""
    limit: int | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    task_id: str
    n: int
    metrics: dict[str, float]
    counts: dict[str, int] = field(default_factory=dict)
    records: list[dict] = field(default_factory=list)

    def as_dict(self, include_records: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "task_id": self.task_id,
            "n": self.n,
            "metrics": self.metrics,
            "counts": self.counts,
        }
        if include_records:
            d["records"] = self.records
        return d


class Task(ABC):
    """One scoreable task."""

    kind: str = ""

    def __init__(self, spec: TaskSpec, root: Path) -> None:
        self.spec = spec
        self.root = root

    @property
    def id(self) -> str:
        return self.spec.id

    @abstractmethod
    def load(self, limit: int | None = None) -> list[dict]:
        """Read the items for this task."""

    @abstractmethod
    def conversations(self, items: list[dict]) -> list[list[Message]]:
        """One chat conversation per item, in the same order."""

    @abstractmethod
    def score(self, items: list[dict], completions: list[str]) -> TaskResult:
        """Compute metrics from the model's completions."""
