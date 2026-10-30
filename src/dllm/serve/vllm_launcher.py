"""Build and run the vLLM OpenAI-compatible server command.

``build_command`` is separate from ``launch`` so the exact command can be
printed, committed, diffed and pasted into a Dockerfile. A serving config that
only exists as Python kwargs is one you cannot reproduce from a shell.
"""

from __future__ import annotations

import logging
import os
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
    if cfg.enforce_eager:
        cmd.append("--enforce-eager")
    cmd.extend(cfg.extra_args)
    return cmd


def build_env(cfg: VllmConfig) -> dict[str, str]:
    """Environment variables the server needs on top of the caller's.

    Only the CPU backend needs any: its KV cache size and its per-step timeout
    are environment variables rather than flags.
    """
    env: dict[str, str] = {}
    if cfg.device == "cpu":
        if cfg.cpu_kvcache_space_gib is not None:
            env["VLLM_CPU_KVCACHE_SPACE"] = str(cfg.cpu_kvcache_space_gib)
        if cfg.cpu_execute_timeout_s is not None:
            env["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = str(cfg.cpu_execute_timeout_s)
    return env


def launch(cfg: VllmConfig) -> int:
    """Run the server in the foreground. Returns its exit code."""
    if cfg.device == "cuda" and shutil.which("nvidia-smi") is None:
        log.warning(
            "no nvidia-smi found; serve.vllm.device=cuda needs an NVIDIA GPU. "
            "Use configs/serve_cpu.yaml for the CPU backend."
        )

    model_path = resolve_model_path(cfg)
    if not Path(model_path).is_dir() and "/" not in model_path:
        raise FileNotFoundError(
            f"serve.vllm.model={cfg.model!r} is neither a local directory nor a Hub id. "
            "Run `dllm model merge` first, or point it at a Hub model."
        )

    cmd = build_command(cfg)
    env = build_env(cfg)
    log.info("launching (%s): %s", cfg.device, " ".join(cmd))
    return subprocess.call(cmd, env={**os.environ, **env})
