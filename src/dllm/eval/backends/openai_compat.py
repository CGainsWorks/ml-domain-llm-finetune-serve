"""Backend for any OpenAI-compatible endpoint: vLLM directly, or the gateway.

Being able to point the same eval at the *deployed* endpoint is what makes the
scorecard describe production rather than a checkpoint on disk. Requests are
issued concurrently because a served model's whole point is batching.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

from dllm.config import BackendConfig, GenerationConfig
from dllm.eval.backends.base import Backend
from dllm.prompts import Message

log = logging.getLogger(__name__)


class OpenAICompatBackend(Backend):
    name = "openai"

    def __init__(self, cfg: BackendConfig) -> None:
        self.cfg = cfg
        self.base_url = cfg.base_url.rstrip("/")
        self.api_key = os.environ.get(cfg.api_key_env, "")
        self.model = cfg.model or self._discover_model()

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _discover_model(self) -> str:
        """Ask the server what it is serving, so the config need not repeat it."""
        try:
            r = httpx.get(f"{self.base_url}/models", headers=self._headers(), timeout=15.0)
            r.raise_for_status()
            data = r.json().get("data", [])
            if data:
                return data[0]["id"]
        except Exception as e:  # noqa: BLE001
            log.warning("could not discover model from %s: %s", self.base_url, e)
        return "unknown"

    async def _one(
        self,
        client: httpx.AsyncClient,
        sem: asyncio.Semaphore,
        messages: list[Message],
        gen: GenerationConfig,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": gen.max_new_tokens,
            "temperature": gen.temperature,
            "top_p": gen.top_p,
        }
        if gen.stop:
            payload["stop"] = gen.stop
        if gen.temperature == 0:
            payload["seed"] = gen.seed

        async with sem:
            for attempt in range(3):
                try:
                    r = await client.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers=self._headers(),
                        timeout=self.cfg.timeout_s,
                    )
                    if r.status_code == 429:
                        # The gateway's own rate limiter. Back off rather than
                        # recording an empty completion as a model failure.
                        await asyncio.sleep(2**attempt)
                        continue
                    r.raise_for_status()
                    return r.json()["choices"][0]["message"]["content"].strip()
                except Exception as e:  # noqa: BLE001
                    if attempt == 2:
                        log.error("request failed after retries: %s", e)
                        return ""
                    await asyncio.sleep(2**attempt)
        return ""

    async def _run(self, conversations: list[list[Message]], gen: GenerationConfig) -> list[str]:
        sem = asyncio.Semaphore(self.cfg.concurrency)
        limits = httpx.Limits(max_connections=self.cfg.concurrency * 2)
        async with httpx.AsyncClient(limits=limits) as client:
            return await asyncio.gather(*(self._one(client, sem, c, gen) for c in conversations))

    def generate(self, conversations: list[list[Message]], gen: GenerationConfig) -> list[str]:
        return asyncio.run(self._run(conversations, gen))

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "concurrency": self.cfg.concurrency,
        }
