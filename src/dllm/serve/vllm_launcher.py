"""Build and run the vLLM OpenAI-compatible server command.

``build_command`` is separate from ``launch`` so the exact command can be
printed, committed, diffed and pasted into a Dockerfile. A serving config that
only exists as Python kwargs is one you cannot reproduce from a shell.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

from dllm.config import VllmConfig
from dllm.paths import ROOT

log = logging.getLogger(__name__)


def resolve_model_path(cfg: VllmConfig) -> str:
    """Absolute path for a local checkpoint, or the string as-is for a Hub id."""
    candidate = Path(cfg.model)
    if not candidate.is_absolute():
        candidate = ROOT / cfg.model
    return str(candidate) if candidate.is_dir() else cfg.model


def build_command(cfg: VllmConfig) -> list[str]:
    """The full ``vllm serve`` argv."""
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        resolve_model_path(cfg),
        "--served-model-name",
        cfg.served_model_name,
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
        "--dtype",
        cfg.dtype,
        "--max-model-len",
        str(cfg.max_model_len),
        "--gpu-memory-utilization",
        str(cfg.gpu_memory_utilization),
        "--max-num-seqs",
        str(cfg.max_num_seqs),
        "--tensor-parallel-size",
        str(cfg.tensor_parallel_size),
    ]
    if cfg.enable_prefix_caching:
        # Every request in this domain repeats the same system prompt and often
        # the same schema, so prefix caching is a large, free throughput win.
        cmd.append("--enable-prefix-caching")
    cmd.extend(cfg.extra_args)
    return cmd


def launch(cfg: VllmConfig) -> int:
    """Run the server in the foreground. Returns its exit code."""
    if shutil.which("nvidia-smi") is None:
        log.warning("no nvidia-smi found; vLLM requires a CUDA device and will likely fail")

    model_path = resolve_model_path(cfg)
    if not Path(model_path).is_dir() and "/" not in model_path:
        raise FileNotFoundError(
            f"serve.vllm.model={cfg.model!r} is neither a local directory nor a Hub id. "
            "Run `dllm model merge` first, or point it at a Hub model."
        )

    cmd = build_command(cfg)
    log.info("launching: %s", " ".join(cmd))
    return subprocess.call(cmd)
