"""Backend interface: something that turns conversations into completions.

Two implementations ship: ``hf`` (local Transformers, used for the baseline and
for scoring checkpoints on the training box) and ``openai`` (any
OpenAI-compatible endpoint, which is how the vLLM server and the FastAPI
gateway get evaluated). The harness does not care which it is given, so the
same task definitions score a raw checkpoint and a deployed endpoint.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from dllm.config import GenerationConfig
from dllm.prompts import Message


class Backend(ABC):
    """Generates one completion per conversation."""

    name: str = "backend"

    @abstractmethod
    def generate(self, conversations: list[list[Message]], gen: GenerationConfig) -> list[str]:
        """Return one completion string per conversation, in the same order."""

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Everything needed to reproduce this run: model id, revision, dtype."""

    def close(self) -> None:  # noqa: B027 - optional hook, not every backend holds resources
        """Release resources. Safe to call more than once.

        Deliberately concrete and empty rather than abstract: an HTTP-backed
        backend has nothing to release, and forcing it to implement a no-op adds
        noise without adding safety.
        """

    def __enter__(self) -> Backend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
