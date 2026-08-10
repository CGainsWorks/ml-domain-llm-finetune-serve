"""Benchmark harness.

The load generator is the one component whose output is a *number quoted in the
README*, so its arithmetic has to be right. These tests drive it against a stub
OpenAI-compatible endpoint with known timing, which verifies the wiring, the
SSE parsing and the percentile maths without needing a GPU.

They do not produce benchmark results and must never be presented as such --
the stub has no model behind it. Real numbers come from `make bench` against a
served model.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from dllm.bench.loadgen import (
    RequestSample,
    _one_request,
    _percentiles,
    load_prompts,
    render_markdown,
)


class TestPercentiles:
    def test_empty(self) -> None:
        assert _percentiles([]) == {}

    def test_single_value(self) -> None:
        p = _percentiles([5.0])
        assert p["p50"] == 5.0 and p["p99"] == 5.0 and p["max"] == 5.0

    def test_ordering_holds(self) -> None:
        p = _percentiles([float(i) for i in range(1, 101)])
        assert p["p50"] <= p["p90"] <= p["p95"] <= p["p99"] <= p["max"]

    def test_known_values(self) -> None:
        p = _percentiles([float(i) for i in range(1, 101)])
        assert p["mean"] == pytest.approx(50.5)
        assert p["max"] == 100.0
        assert p["p50"] == pytest.approx(50.5, abs=1.0)

    def test_tail_is_not_the_mean(self) -> None:
        """The whole point of reporting percentiles on a batching server."""
        values = [10.0] * 99 + [5000.0]
        p = _percentiles(values)
        assert p["p50"] == 10.0
        assert p["max"] == 5000.0
        assert p["mean"] > p["p50"]


class TestNonStreamingRequest:
    def test_counts_completion_tokens_from_usage(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "SELECT 1"}}],
                    "usage": {"completion_tokens": 7, "total_tokens": 20},
                },
            )

        async def go() -> RequestSample:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
                return await _one_request(c, "http://x/v1/chat/completions", {}, {}, False)

        s = asyncio.run(go())
        assert s.ok and s.output_tokens == 7 and s.e2e_ms > 0

    def test_http_error_is_recorded_not_raised(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, text="slow down")

        async def go() -> RequestSample:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
                return await _one_request(c, "http://x/v1/chat/completions", {}, {}, False)

        s = asyncio.run(go())
        assert not s.ok and s.status == 429

    def test_transport_error_is_recorded_not_raised(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        async def go() -> RequestSample:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
                return await _one_request(c, "http://x/v1/chat/completions", {}, {}, False)

        s = asyncio.run(go())
        assert not s.ok and s.status == 0 and s.error


class TestStreamingRequest:
    """SSE parsing: TTFT and token counting come from here."""

    @staticmethod
    def _sse(chunks: list[str]) -> bytes:
        lines = []
        for c in chunks:
            payload = {"choices": [{"delta": {"content": c}}]}
            lines.append(f"data: {json.dumps(payload)}\n\n")
        lines.append("data: [DONE]\n\n")
        return "".join(lines).encode()

    def _run(self, body: bytes) -> RequestSample:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body, headers={"Content-Type": "text/event-stream"})

        async def go() -> RequestSample:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
                return await _one_request(c, "http://x/v1/chat/completions", {}, {}, True)

        return asyncio.run(go())

    def test_counts_content_deltas(self) -> None:
        s = self._run(self._sse(["SELECT", " ", "1"]))
        assert s.ok and s.output_tokens == 3

    def test_records_time_to_first_token(self) -> None:
        s = self._run(self._sse(["a", "b"]))
        assert s.ttft_ms is not None and s.ttft_ms >= 0
        assert s.ttft_ms <= s.e2e_ms

    def test_empty_deltas_are_not_counted(self) -> None:
        s = self._run(self._sse(["", "", "x"]))
        assert s.output_tokens == 1

    def test_malformed_chunk_does_not_abort_the_request(self) -> None:
        body = (
            b'data: {not json}\n\ndata: {"choices":[{"delta":{"content":"x"}}]}\n\ndata: [DONE]\n\n'
        )
        s = self._run(body)
        assert s.ok and s.output_tokens == 1

    def test_error_status_while_streaming(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="unavailable")

        async def go() -> RequestSample:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
                return await _one_request(c, "http://x/v1/chat/completions", {}, {}, True)

        s = asyncio.run(go())
        assert not s.ok and s.status == 503


class TestRateLimitAccounting:
    """A 429 is the gateway declining, not the server failing.

    The first real benchmark run reported 273 of 288 requests failed with
    healthy latencies and no hint that the load generator had throttled itself
    against its own gateway's 60 rpm default.
    """

    def test_429_is_classified_as_rate_limited(self) -> None:
        assert RequestSample(ok=False, status=429).rate_limited

    @pytest.mark.parametrize("status", [0, 200, 400, 500, 503])
    def test_other_statuses_are_not(self, status: int) -> None:
        assert not RequestSample(ok=False, status=status).rate_limited

    def test_report_marks_a_throttled_run_invalid(self) -> None:
        report = {
            "endpoint": "http://x/v1",
            "model": "m",
            "streaming": True,
            "max_tokens": 128,
            "requests_per_level": 48,
            "created_at": "2026-01-01T00:00:00+00:00",
            "levels": [
                {
                    "concurrency": 1,
                    "requests_per_s": 1.0,
                    "output_tokens_per_s": 10.0,
                    "ttft_ms": {"p50": 50, "p95": 90},
                    "e2e_ms": {"p50": 100, "p95": 200, "p99": 300},
                    "failures": 40,
                    "rate_limited": 40,
                }
            ],
            "summary": {
                "peak_output_tokens_per_s": 10.0,
                "peak_at_concurrency": 1,
                "p50_latency_ms_at_peak": 100,
                "p95_latency_ms_at_peak": 200,
                "total_rate_limited": 40,
                "valid": False,
            },
        }
        body = render_markdown(report)
        assert "not valid" in body
        assert "--rpm" in body, "the report must say how to fix it"

    def test_clean_run_has_no_warning(self) -> None:
        report = {
            "endpoint": "http://x/v1",
            "model": "m",
            "streaming": True,
            "max_tokens": 128,
            "requests_per_level": 48,
            "created_at": "2026-01-01T00:00:00+00:00",
            "levels": [
                {
                    "concurrency": 1,
                    "requests_per_s": 1.0,
                    "output_tokens_per_s": 10.0,
                    "ttft_ms": {"p50": 50, "p95": 90},
                    "e2e_ms": {"p50": 100, "p95": 200, "p99": 300},
                    "failures": 0,
                    "rate_limited": 0,
                }
            ],
            "summary": {
                "peak_output_tokens_per_s": 10.0,
                "peak_at_concurrency": 1,
                "p50_latency_ms_at_peak": 100,
                "p95_latency_ms_at_peak": 200,
                "total_rate_limited": 0,
                "valid": True,
            },
        }
        assert "not valid" not in render_markdown(report)


class TestBenchmarkKeyMinting:
    def test_limits_are_recorded_on_the_key(self) -> None:
        from dllm.serve.gateway.auth import mint_keys

        entry = mint_keys(1, tier="bench", rpm=100_000, tokens_per_min=10**8)[0]
        assert entry["rpm"] == 100_000
        assert entry["tokens_per_min"] == 10**8

    def test_default_key_carries_no_overrides(self) -> None:
        from dllm.serve.gateway.auth import mint_keys

        entry = mint_keys(1)[0]
        assert "rpm" not in entry and "tokens_per_min" not in entry

    def test_the_override_actually_raises_the_limit(self) -> None:
        """End to end through the limiter, not just the key record."""
        import asyncio

        from dllm.serve.gateway.ratelimit import RateLimiter

        limiter = RateLimiter(rpm=60, burst=2, tokens_per_min=1000)

        async def go():
            return [
                await limiter.check("bench", estimated_tokens=50, rpm=100_000, tokens_per_min=10**8)
                for _ in range(40)
            ]

        assert all(r.allowed for r in asyncio.run(go()))


class TestPrompts:
    def test_falls_back_when_no_prepared_split(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pathlib import Path

        import dllm.bench.loadgen as lg

        monkeypatch.setattr(lg, "DATA_PROCESSED", Path("/nonexistent"))
        prompts = load_prompts(4)
        assert len(prompts) == 4
        assert prompts[0][0]["role"] == "system"


class TestRendering:
    def test_markdown_table_has_a_row_per_level(self) -> None:
        report = {
            "endpoint": "http://x/v1",
            "model": "domain-sql",
            "streaming": True,
            "max_tokens": 128,
            "requests_per_level": 48,
            "created_at": "2026-01-01T00:00:00+00:00",
            "levels": [
                {
                    "concurrency": c,
                    "requests_per_s": 1.0,
                    "output_tokens_per_s": 10.0,
                    "ttft_ms": {"p50": 50, "p95": 90},
                    "e2e_ms": {"p50": 100, "p95": 200, "p99": 300},
                    "failures": 0,
                }
                for c in (1, 4, 16)
            ],
            "summary": {
                "peak_output_tokens_per_s": 10.0,
                "peak_at_concurrency": 16,
                "p50_latency_ms_at_peak": 100,
                "p95_latency_ms_at_peak": 200,
            },
        }
        body = render_markdown(report)
        assert body.count("\n| 1 |") == 1
        assert "| 16 |" in body
        assert "domain-sql" in body
