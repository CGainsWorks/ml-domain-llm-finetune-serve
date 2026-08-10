"""Single source of truth for where things live on disk.

Every path is derived from the repository root so that scripts, CI and the
containers all agree without passing absolute paths around.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Repository root.

    Overridable with ``DLLM_REPO_ROOT`` so the containers (which copy the
    source to ``/app``) do not have to rely on ``.git`` being present.
    """
    env = os.environ.get("DLLM_REPO_ROOT")
    if env:
        return Path(env).resolve()

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    # Installed as a wheel with no source tree: fall back to the CWD.
    return Path.cwd().resolve()


ROOT = repo_root()

CONFIGS = ROOT / "configs"
EVALS = ROOT / "evals"
EVAL_TASKS = EVALS / "tasks"
EVAL_RESULTS = EVALS / "results"
DOCS = ROOT / "docs"

# Generated, git-ignored.
DATA = Path(os.environ.get("DLLM_DATA_DIR", ROOT / "data"))
DATA_RAW = DATA / "raw"
DATA_PROCESSED = DATA / "processed"
ARTIFACTS = Path(os.environ.get("DLLM_ARTIFACTS_DIR", ROOT / "artifacts"))
CHECKPOINTS = ARTIFACTS / "checkpoints"
MERGED = ARTIFACTS / "merged"
BENCH = ARTIFACTS / "bench"


def ensure(*paths: Path) -> None:
    """Create each directory if missing."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
