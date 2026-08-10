"""HTTP-level tests for the gateway.

The unit tests in ``test_gateway.py`` cover the key store and the buckets in
isolation. These drive the actual FastAPI app so that the wiring is tested too:
a rate limiter that works perfectly but is never called from the request path
is not a rate limiter.

The upstream vLLM server is replaced with an ``httpx.MockTransport``, so these
run on CPU in CI with no model and no GPU.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from dllm.config import GatewayConfig
from dllm.serve.gateway.app import _prompt_chars, create_app
from dllm.serve.gateway.auth import mint_keys

CHAT_BODY = {
    "model": "domain-sql",
    "messages": [{"role": "user", "content": "SELECT what?"}],
    "max_tokens": 32,
}


def upstream_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/models"):
        return httpx.Response(200, json={"data": [{"id": "domain-sql"}]})
    return httpx.Response(
        200,
        json={
            "id": "cmpl-1",
            "choices": [{"message": {"role": "assistant", "content": "SELECT 1"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )


@pytest.fixture
def keyed_client(monkeypatch: pytest.MonkeyPatch):
    """A running app with one valid key and a mocked upstream."""
    entry = mint_keys(1)[0]
    monkeypatch.setenv(
        "DLLM_GATEWAY_KEYS",
        json.dumps([{k: v for k, v in entry.items() if k != "key"}]),
    )
    cfg = GatewayConfig(rate_limit_rpm=600, rate_limit_burst=50, rate_limit_tokens_per_min=100_000)
    app = create_app(cfg)
    with TestClient(app) as client:
        app.state.client = httpx.AsyncClient(
            transport=httpx.MockTransport(upstream_handler),
            base_url="http://upstream/v1",
        )
        yield client, entry["key"]


@pytest.fixture
def keyless_client(monkeypatch: pytest.MonkeyPatch):
    """An app with no keys configured at all."""
    monkeypatch.delenv("DLLM_GATEWAY_KEYS", raising=False)
    monkeypatch.delenv("DLLM_GATEWAY_KEYS_FILE", raising=False)
    app = create_app(GatewayConfig())
    with TestClient(app) as client:
        app.state.client = httpx.AsyncClient(
            transport=httpx.MockTransport(upstream_handler), base_url="http://upstream/v1"
        )
        yield client


class TestHealth:
    def test_healthz_needs_no_auth(self, keyed_client) -> None:
        client, _ = keyed_client
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_readyz_reports_upstream_models(self, keyed_client) -> None:
        client, _ = keyed_client
        r = client.get("/readyz")
        assert r.status_code == 200
        assert "domain-sql" in r.json()["models"]

    def test_metrics_endpoint_serves(self, keyed_client) -> None:
        client, _ = keyed_client
        assert client.get("/metrics").status_code == 200


class TestAuth:
    def test_missing_key_is_rejected(self, keyed_client) -> None:
        client, _ = keyed_client
        assert client.post("/v1/chat/completions", json=CHAT_BODY).status_code == 401

    def test_wrong_key_is_rejected(self, keyed_client) -> None:
        client, _ = keyed_client
        r = client.post(
            "/v1/chat/completions",
            json=CHAT_BODY,
            headers={"Authorization": "Bearer dllm_sk_not_a_real_key"},
        )
        assert r.status_code == 401

    def test_malformed_header_is_rejected(self, keyed_client) -> None:
        client, key = keyed_client
        r = client.post("/v1/chat/completions", json=CHAT_BODY, headers={"Authorization": key})
        assert r.status_code == 401

    def test_valid_key_is_accepted(self, keyed_client) -> None:
        client, key = keyed_client
        r = client.post(
            "/v1/chat/completions", json=CHAT_BODY, headers={"Authorization": f"Bearer {key}"}
        )
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "SELECT 1"
        assert r.headers["X-Request-Id"]

    def test_no_keys_configured_refuses_everything(self, keyless_client) -> None:
        """It must never fail open. A misconfigured gateway serves nobody."""
        r = keyless_client.post("/v1/chat/completions", json=CHAT_BODY)
        assert r.status_code == 503
        assert "no API keys" in r.json()["detail"]


class TestRequestValidation:
    def test_model_allowlist(self, keyed_client) -> None:
        client, key = keyed_client
        r = client.post(
            "/v1/chat/completions",
            json={**CHAT_BODY, "model": "some-other-model"},
            headers={"Authorization": f"Bearer {key}"},
        )
        assert r.status_code == 400
        assert "not served here" in r.json()["detail"]

    def test_oversized_prompt_rejected_before_the_gpu(self, keyed_client) -> None:
        client, key = keyed_client
        r = client.post(
            "/v1/chat/completions",
            json={**CHAT_BODY, "messages": [{"role": "user", "content": "x" * 20_000}]},
            headers={"Authorization": f"Bearer {key}"},
        )
        assert r.status_code == 413

    def test_non_json_body(self, keyed_client) -> None:
        client, key = keyed_client
        r = client.post(
            "/v1/chat/completions",
            content=b"not json",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        assert r.status_code == 400


class TestRateLimitingInTheRequestPath:
    def test_request_limit_returns_429_with_retry_after(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entry = mint_keys(1)[0]
        monkeypatch.setenv(
            "DLLM_GATEWAY_KEYS",
            json.dumps([{k: v for k, v in entry.items() if k != "key"}]),
        )
        cfg = GatewayConfig(rate_limit_rpm=60, rate_limit_burst=2, rate_limit_tokens_per_min=10**6)
        app = create_app(cfg)
        with TestClient(app) as client:
            app.state.client = httpx.AsyncClient(
                transport=httpx.MockTransport(upstream_handler), base_url="http://upstream/v1"
            )
            headers = {"Authorization": f"Bearer {entry['key']}"}
            codes = [
                client.post("/v1/chat/completions", json=CHAT_BODY, headers=headers).status_code
                for _ in range(4)
            ]
            assert codes[:2] == [200, 200]
            assert 429 in codes[2:]

            blocked = client.post("/v1/chat/completions", json=CHAT_BODY, headers=headers)
            assert blocked.status_code == 429
            assert int(blocked.headers["Retry-After"]) >= 1
            assert blocked.json()["error"]["type"] == "rate_limit_exceeded"

    def test_token_limit_blocks_a_single_huge_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entry = mint_keys(1)[0]
        monkeypatch.setenv(
            "DLLM_GATEWAY_KEYS",
            json.dumps([{k: v for k, v in entry.items() if k != "key"}]),
        )
        cfg = GatewayConfig(rate_limit_rpm=600, rate_limit_burst=100, rate_limit_tokens_per_min=100)
        app = create_app(cfg)
        with TestClient(app) as client:
            app.state.client = httpx.AsyncClient(
                transport=httpx.MockTransport(upstream_handler), base_url="http://upstream/v1"
            )
            r = client.post(
                "/v1/chat/completions",
                json={**CHAT_BODY, "max_tokens": 5000},
                headers={"Authorization": f"Bearer {entry['key']}"},
            )
            assert r.status_code == 429
            assert "token rate limit" in r.json()["error"]["message"]


class TestPromptSizing:
    def test_chat_shape(self) -> None:
        assert _prompt_chars({"messages": [{"content": "abc"}, {"content": "de"}]}) == 5

    def test_completion_shape(self) -> None:
        assert _prompt_chars({"prompt": "abcde"}) == 5

    def test_batched_completion_shape(self) -> None:
        assert _prompt_chars({"prompt": ["ab", "cde"]}) == 5

    def test_empty(self) -> None:
        assert _prompt_chars({}) == 0
