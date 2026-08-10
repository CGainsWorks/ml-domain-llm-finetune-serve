"""Backend construction. Heavy imports stay inside the factory."""

from __future__ import annotations

from dllm.config import BackendConfig, ModelConfig, QuantConfig
from dllm.eval.backends.base import Backend


def build_backend(
    backend_cfg: BackendConfig,
    model_cfg: ModelConfig,
    *,
    model_path: str | None = None,
    adapter_path: str | None = None,
    quant: QuantConfig | None = None,
) -> Backend:
    """Construct the backend named by ``backend_cfg.kind``.

    ``transformers`` and ``torch`` are imported only when the ``hf`` backend is
    actually requested, so the eval harness and its unit tests still run on a
    CPU-only box with nothing but ``requirements/base.txt`` installed.
    """
    if backend_cfg.kind == "hf":
        from dllm.eval.backends.hf import HFBackend

        return HFBackend(
            model_cfg, backend_cfg, model_path=model_path, adapter_path=adapter_path, quant=quant
        )
    if backend_cfg.kind == "openai":
        from dllm.eval.backends.openai_compat import OpenAICompatBackend

        return OpenAICompatBackend(backend_cfg)
    raise ValueError(f"unknown backend kind {backend_cfg.kind!r}")


__all__ = ["Backend", "build_backend"]
