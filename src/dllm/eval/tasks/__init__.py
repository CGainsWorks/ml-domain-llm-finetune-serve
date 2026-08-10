"""Task registry: resolve a task id to a constructed :class:`Task`."""

from __future__ import annotations

from pathlib import Path

import yaml

from dllm.eval.tasks.base import Task, TaskResult, TaskSpec
from dllm.eval.tasks.general_guard import GeneralGuardTask
from dllm.eval.tasks.sql_execution import SqlExecutionTask
from dllm.paths import EVAL_TASKS, ROOT

TASK_KINDS: dict[str, type[Task]] = {
    SqlExecutionTask.kind: SqlExecutionTask,
    GeneralGuardTask.kind: GeneralGuardTask,
}


def load_spec(task_id: str, tasks_dir: Path | None = None) -> TaskSpec:
    """Read ``evals/tasks/<task_id>.yaml``."""
    tasks_dir = tasks_dir or EVAL_TASKS
    path = tasks_dir / f"{task_id}.yaml"
    if not path.is_file():
        available = ", ".join(sorted(p.stem for p in tasks_dir.glob("*.yaml"))) or "none"
        raise FileNotFoundError(f"no task definition at {path}. Available: {available}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw.setdefault("id", task_id)
    if raw["id"] != task_id:
        raise ValueError(f"{path} declares id {raw['id']!r} but is named {task_id!r}")
    return TaskSpec(**raw)


def build_task(task_id: str, *, root: Path | None = None, tasks_dir: Path | None = None) -> Task:
    spec = load_spec(task_id, tasks_dir)
    try:
        cls = TASK_KINDS[spec.kind]
    except KeyError:
        raise ValueError(
            f"task {task_id!r} has unknown kind {spec.kind!r}. "
            f"Known kinds: {', '.join(sorted(TASK_KINDS))}"
        ) from None
    return cls(spec, root or ROOT)


__all__ = ["Task", "TaskResult", "TaskSpec", "build_task", "load_spec"]
