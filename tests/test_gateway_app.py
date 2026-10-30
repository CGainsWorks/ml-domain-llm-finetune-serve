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
from dllm.serve.gateway.app import _prompt_chars, _SseUsageScanner, create_app
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


SSE_WITH_USAGE = (
    b'data: {"choices":[{"delta":{"content":"SELECT"}}]}\n\n'
    b'data: {"choices":[{"delta":{"content":" 1"}}]}\n\n'
    b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":3,"total_tokens":15}}\n\n'
    b"data: [DONE]\n\n"
)


def streaming_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    if body.get("stream"):
        return httpx.Response(
            200, content=SSE_WITH_USAGE, headers={"Content-Type": "text/event-stream"}
        )
    return upstream_handler(request)


def token_samples(metrics_text: str, key_id: str) -> dict[str, float]:
    """``{kind: value}`` for one key from the Prometheus exposition text."""
    out: dict[str, float] = {}
    for line in metrics_text.splitlines():
        if line.startswith("dllm_gateway_tokens_total{") and f'key_id="{key_id}"' in line:
            kind = line.split('kind="', 1)[1].split('"', 1)[0]
            out[kind] = float(line.rsplit(" ", 1)[1])
    return out


class TestTokenAccounting:
    """Per-key token counters: what a usage-based bill would be built from."""

    @pytest.fixture
    def two_keys(self, monkeypatch: pytest.MonkeyPatch):
        entries = mint_keys(2)
        monkeypatch.setenv(
            "DLLM_GATEWAY_KEYS",
            json.dumps([{k: v for k, v in e.items() if k != "key"} for e in entries]),
        )
        app = create_app(GatewayConfig(rate_limit_rpm=600, rate_limit_burst=50))
        with TestClient(app) as client:
            app.state.client = httpx.AsyncClient(
                transport=httpx.MockTransport(streaming_handler),
                base_url="http://upstream/v1",
            )
            yield client, entries

    def test_counts_reported_usage_per_key(self, two_keys) -> None:
        client, (a, b) = two_keys
        auth_a = {"Authorization": f"Bearer {a['key']}"}
        for _ in range(3):
            assert client.post("/v1/chat/completions", json=CHAT_BODY, headers=auth_a).is_success
        text = client.get("/metrics").text
        # The mocked upstream reports 10 prompt and 5 completion tokens a call.
        assert token_samples(text, a["key_id"]) == {"prompt": 30.0, "completion": 15.0}
        assert token_samples(text, b["key_id"]) == {}

    def test_rejected_requests_are_not_counted(self, two_keys) -> None:
        client, (a, _) = two_keys
        client.post(
            "/v1/chat/completions",
            json={**CHAT_BODY, "model": "gpt-4"},
            headers={"Authorization": f"Bearer {a['key']}"},
        )
        assert token_samples(client.get("/metrics").text, a["key_id"]) == {}

    def test_streaming_usage_is_counted_when_present(self, two_keys) -> None:
        client, (a, _) = two_keys
        r = client.post(
            "/v1/chat/completions",
            json={**CHAT_BODY, "stream": True, "stream_options": {"include_usage": True}},
            headers={"Authorization": f"Bearer {a['key']}"},
        )
        assert r.status_code == 200
        assert r.content == SSE_WITH_USAGE, "the stream must pass through unchanged"
        text = client.get("/metrics").text
        assert token_samples(text, a["key_id"]) == {"prompt": 12.0, "completion": 3.0}

    def test_secret_never_appears_in_metrics(self, two_keys) -> None:
        client, (a, _) = two_keys
        client.post(
            "/v1/chat/completions", json=CHAT_BODY, headers={"Authorization": f"Bearer {a['key']}"}
        )
        text = client.get("/metrics").text
        assert a["key"] not in text
        assert a["key_hash"] not in text


class TestSseUsageScanner:
    def test_usage_split_across_chunks(self) -> None:
        scan = _SseUsageScanner()
        for i in range(0, len(SSE_WITH_USAGE), 7):
            scan.feed(SSE_WITH_USAGE[i : i + 7])
        assert scan.usage == {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}

    def test_no_usage_chunk(self) -> None:
        scan = _SseUsageScanner()
        scan.feed(b'data: {"choices":[]}\n\ndata: [DONE]\n\n')
        assert scan.usage is None

    def test_malformed_line_is_ignored(self) -> None:
        scan = _SseUsageScanner()
        scan.feed(b'data: {"usage": nope\n\n')
        assert scan.usage is None
