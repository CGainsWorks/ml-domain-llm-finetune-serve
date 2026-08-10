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
