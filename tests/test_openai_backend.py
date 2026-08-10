"""The OpenAI-compatible eval backend.

This is the path that makes a scorecard describe *production* rather than a
checkpoint on disk: the same task definitions, pointed at the served endpoint.
It had no coverage, and it contains the retry and rate-limit handling that
decides whether a 429 from our own gateway is recorded as a model failure.
"""

from __future__ import annotations

import httpx
import pytest

from dllm.config import BackendConfig, GenerationConfig
from dllm.eval.backends.openai_compat import OpenAICompatBackend
from dllm.prompts import sql_messages

GEN = GenerationConfig(max_new_tokens=32, temperature=0.0)
CONVOS = [sql_messages("How many?", "CREATE TABLE t (a VARCHAR)")]


@pytest.fixture
def patched_client(monkeypatch: pytest.MonkeyPatch):
    """Force every AsyncClient the backend builds onto a mock transport."""

    def install(handler) -> None:
        real = httpx.AsyncClient

        class Patched(real):
            def __init__(self, *args, **kwargs):
                kwargs["transport"] = httpx.MockTransport(handler)
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", Patched)

    return install


def cfg(**kw) -> BackendConfig:
    return BackendConfig(kind="openai", base_url="http://vllm:8000/v1", concurrency=2, **kw)


class TestModelDiscovery:
    def test_discovers_the_served_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # raise_for_status() needs the originating request attached, so build a
        # complete Response rather than a bare status+body.
        def fake_get(url, **kwargs):
            return httpx.Response(
                200,
                json={"data": [{"id": "domain-sql"}]},
                request=httpx.Request("GET", url),
            )

        monkeypatch.setattr(httpx, "get", fake_get)
        assert OpenAICompatBackend(cfg()).model == "domain-sql"

    def test_empty_model_list_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            httpx,
            "get",
            lambda url, **kw: httpx.Response(
                200, json={"data": []}, request=httpx.Request("GET", url)
            ),
        )
        assert OpenAICompatBackend(cfg()).model == "unknown"

    def test_unreachable_server_does_not_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Constructing the backend must not explode before the eval starts."""

        def boom(*a, **k):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(httpx, "get", boom)
        assert OpenAICompatBackend(cfg()).model == "unknown"

    def test_explicit_model_skips_discovery(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*a, **k):
            raise AssertionError("should not have called /models")

        monkeypatch.setattr(httpx, "get", boom)
        assert OpenAICompatBackend(cfg(model="pinned")).model == "pinned"


class TestGeneration:
    def test_returns_the_completion(self, patched_client) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "  SELECT a FROM t  "}}]}
            )

        patched_client(handler)
        out = OpenAICompatBackend(cfg(model="m")).generate(CONVOS, GEN)
        assert out == ["SELECT a FROM t"]

    def test_preserves_order(self, patched_client) -> None:
        """Requests run concurrently; results must still line up with items."""

        def handler(request: httpx.Request) -> httpx.Response:
            import json as _json

            body = _json.loads(request.content)
            marker = body["messages"][-1]["content"]
            return httpx.Response(200, json={"choices": [{"message": {"content": marker}}]})

        patched_client(handler)
        convos = [[{"role": "user", "content": f"q{i}"}] for i in range(8)]
        out = OpenAICompatBackend(cfg(model="m")).generate(convos, GEN)
        assert out == [f"q{i}" for i in range(8)]

    def test_greedy_request_pins_the_seed(self, patched_client) -> None:
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json as _json

            seen.update(_json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

        patched_client(handler)
        OpenAICompatBackend(cfg(model="m")).generate(CONVOS, GEN)
        assert seen["temperature"] == 0.0
        assert seen["seed"] == GEN.seed
        assert seen["max_tokens"] == GEN.max_new_tokens
        assert seen["stop"] == GEN.stop

    def test_server_error_yields_empty_not_an_exception(self, patched_client) -> None:
        """One bad response must not abort a 500-item eval."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        patched_client(handler)
        assert OpenAICompatBackend(cfg(model="m")).generate(CONVOS, GEN) == [""]

    def test_retries_then_succeeds(self, patched_client) -> None:
        state = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            state["n"] += 1
            if state["n"] < 2:
                return httpx.Response(503, text="warming up")
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        patched_client(handler)
        assert OpenAICompatBackend(cfg(model="m")).generate(CONVOS, GEN) == ["ok"]
        assert state["n"] >= 2

    def test_rate_limit_is_retried_not_recorded_as_a_model_failure(self, patched_client) -> None:
        """A 429 from our own gateway says nothing about the model."""
        state = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            state["n"] += 1
            if state["n"] == 1:
                return httpx.Response(429, json={"error": {"message": "slow down"}})
            return httpx.Response(200, json={"choices": [{"message": {"content": "recovered"}}]})

        patched_client(handler)
        assert OpenAICompatBackend(cfg(model="m")).generate(CONVOS, GEN) == ["recovered"]


class TestMetadata:
    def test_records_what_was_evaluated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(200, json={"data": []}))
        meta = OpenAICompatBackend(cfg(model="domain-sql")).metadata()
        assert meta["backend"] == "openai"
        assert meta["model"] == "domain-sql"
        assert meta["base_url"] == "http://vllm:8000/v1"

    def test_api_key_is_read_from_the_named_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DLLM_API_KEY", "dllm_sk_secret")
        monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(200, json={"data": []}))
        backend = OpenAICompatBackend(cfg(model="m"))
        assert backend._headers()["Authorization"] == "Bearer dllm_sk_secret"

    def test_no_auth_header_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DLLM_API_KEY", raising=False)
        monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(200, json={"data": []}))
        assert "Authorization" not in OpenAICompatBackend(cfg(model="m"))._headers()
