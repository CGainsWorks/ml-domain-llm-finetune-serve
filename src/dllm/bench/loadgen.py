"""Throughput and latency benchmark against the served endpoint.

Measured, not asserted. Every number quoted in the README comes out of here.

What it reports and why:

``ttft_ms``
    Time to first token, measured from a streaming request. This is what a user
    perceives as "did it respond". Mean end-to-end latency hides it completely.

``itl_ms``
    Inter-token latency: the gap between subsequent tokens once generation is
    under way. Together with TTFT this decomposes latency into queueing plus
    decode, which is what you need to know which knob to turn.

``e2e_ms`` p50/p90/p95/p99
    Percentiles, not means. A mean latency on a batching server is close to
    meaningless -- the whole point of continuous batching is that the tail is
    where the cost lands.

``output_tokens_per_s``
    Aggregate decode throughput at that concurrency level. The number that says
    how many users a card can serve.

The sweep runs each concurrency level after a warm-up, discarding the warm-up
requests: the first request through a cold vLLM server pays CUDA graph capture
and would otherwise poison the p99.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from dllm.config import PipelineConfig
from dllm.data.prepare import read_jsonl
from dllm.paths import DATA_PROCESSED
from dllm.prompts import sql_messages

log = logging.getLogger(__name__)

WARMUP_REQUESTS = 4


@dataclass
class RequestSample:
    ok: bool
    e2e_ms: float = 0.0
    ttft_ms: float | None = None
    output_tokens: int = 0
    status: int = 0
    error: str = ""

    @property
    def rate_limited(self) -> bool:
        """A 429 is the gateway declining, not the server failing.

        Counting it as a plain failure produces the worst possible benchmark
        output: '48/48 requests failed' with healthy latencies and no hint that
        the load generator throttled itself against its own gateway.
        """
        return self.status == 429


@dataclass
class LevelResult:
    concurrency: int
    requests: int
    successes: int
    failures: int
    duration_s: float
    requests_per_s: float
    output_tokens_per_s: float
    rate_limited: int = 0
    e2e_ms: dict[str, float] = field(default_factory=dict)
    ttft_ms: dict[str, float] = field(default_factory=dict)
    itl_ms: dict[str, float] = field(default_factory=dict)


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def pct(p: float) -> float:
        idx = min(len(ordered) - 1, max(0, round(p * (len(ordered) - 1))))
        return round(ordered[idx], 2)

    return {
        "mean": round(statistics.fmean(ordered), 2),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": round(ordered[-1], 2),
    }


def load_prompts(limit: int) -> list[list[dict]]:
    """Real prompts from the held-out split, so the benchmark load is realistic."""
    path = DATA_PROCESSED / "test.jsonl"
    if path.is_file():
        rows = read_jsonl(path, limit)
        if rows:
            return [sql_messages(r["question"], r["context"]) for r in rows]
    log.warning("no prepared test split; falling back to a synthetic prompt")
    return [
        sql_messages(
            "How many players are from Yorkshire?",
            "CREATE TABLE players (name VARCHAR, county VARCHAR)",
        )
    ] * limit


async def _one_request(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    stream: bool,
) -> RequestSample:
    start = time.perf_counter()
    try:
        if not stream:
            r = await client.post(url, json=payload, headers=headers)
            elapsed = (time.perf_counter() - start) * 1000
            if r.status_code >= 400:
                return RequestSample(False, elapsed, None, 0, r.status_code, r.text[:200])
            usage = r.json().get("usage") or {}
            return RequestSample(
                True, elapsed, None, int(usage.get("completion_tokens") or 0), r.status_code
            )

        ttft: float | None = None
        tokens = 0
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            if resp.status_code >= 400:
                body = await resp.aread()
                return RequestSample(
                    False,
                    (time.perf_counter() - start) * 1000,
                    None,
                    0,
                    resp.status_code,
                    body.decode()[:200],
                )
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                if ttft is None:
                    ttft = (time.perf_counter() - start) * 1000
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                if delta.get("content"):
                    tokens += 1
        return RequestSample(True, (time.perf_counter() - start) * 1000, ttft, tokens, 200)
    except Exception as e:  # noqa: BLE001
        return RequestSample(False, (time.perf_counter() - start) * 1000, None, 0, 0, str(e)[:200])


async def _run_level(
    base_url: str,
    headers: dict[str, str],
    model: str,
    prompts: list[list[dict]],
    concurrency: int,
    total: int,
    max_tokens: int,
    stream: bool,
) -> LevelResult:
    url = f"{base_url.rstrip('/')}/chat/completions"
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(
        max_connections=concurrency * 2, max_keepalive_connections=concurrency * 2
    )

    async with httpx.AsyncClient(limits=limits, timeout=180.0) as client:

        async def worker(i: int) -> RequestSample:
            async with sem:
                payload = {
                    "model": model,
                    "messages": prompts[i % len(prompts)],
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "stream": stream,
                }
                return await _one_request(client, url, headers, payload, stream)

        # Warm-up: discard. The first requests pay one-off graph capture costs.
        await asyncio.gather(*(worker(i) for i in range(min(WARMUP_REQUESTS, total))))

        started = time.perf_counter()
        samples = await asyncio.gather(*(worker(i) for i in range(total)))
        duration = time.perf_counter() - started

    ok = [s for s in samples if s.ok]
    e2e = [s.e2e_ms for s in ok]
    ttfts = [s.ttft_ms for s in ok if s.ttft_ms is not None]
    itls = [
        (s.e2e_ms - s.ttft_ms) / (s.output_tokens - 1)
        for s in ok
        if s.ttft_ms is not None and s.output_tokens > 1
    ]
    total_out = sum(s.output_tokens for s in ok)

    return LevelResult(
        concurrency=concurrency,
        requests=total,
        successes=len(ok),
        failures=len(samples) - len(ok),
        rate_limited=sum(1 for s in samples if s.rate_limited),
        duration_s=round(duration, 3),
        requests_per_s=round(len(ok) / duration, 3) if duration else 0.0,
        output_tokens_per_s=round(total_out / duration, 2) if duration else 0.0,
        e2e_ms=_percentiles(e2e),
        ttft_ms=_percentiles(ttfts),
        itl_ms=_percentiles(itls),
    )


def run_benchmark(
    cfg: PipelineConfig,
    *,
    base_url: str | None = None,
    levels: list[int],
    requests_per_level: int = 48,
    max_tokens: int = 128,
    stream: bool = True,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Run the concurrency sweep and write a JSON report."""
    url = base_url or f"http://localhost:{cfg.serve.gateway.port}/v1"
    api_key = os.environ.get("DLLM_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    model = cfg.serve.vllm.served_model_name
    prompts = load_prompts(max(requests_per_level, 32))

    results: list[LevelResult] = []
    for level in levels:
        log.info("benchmarking concurrency=%d (%d requests)", level, requests_per_level)
        result = asyncio.run(
            _run_level(url, headers, model, prompts, level, requests_per_level, max_tokens, stream)
        )
        results.append(result)
        log.info(
            "  %d req/s, %.1f out-tok/s, p50 %.0f ms, p95 %.0f ms, TTFT p50 %s ms",
            result.requests_per_s,
            result.output_tokens_per_s,
            result.e2e_ms.get("p50", 0),
            result.e2e_ms.get("p95", 0),
            result.ttft_ms.get("p50", "n/a"),
        )
        if result.rate_limited:
            log.error(
                "  %d/%d requests were RATE LIMITED (429) by the gateway. These numbers "
                "measure the rate limiter, not the model. Mint a benchmarking key with "
                "raised limits:  dllm serve keys --rpm 100000 --tokens-per-min 100000000",
                result.rate_limited,
                result.requests,
            )
        other_failures = result.failures - result.rate_limited
        if other_failures:
            log.warning("  %d/%d requests failed", other_failures, result.requests)

    best = max(results, key=lambda r: r.output_tokens_per_s) if results else None
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "endpoint": url,
        "model": model,
        "streaming": stream,
        "max_tokens": max_tokens,
        "requests_per_level": requests_per_level,
        "levels": [asdict(r) for r in results],
        "summary": {
            "peak_output_tokens_per_s": best.output_tokens_per_s if best else 0,
            "peak_at_concurrency": best.concurrency if best else 0,
            "p50_latency_ms_at_peak": best.e2e_ms.get("p50") if best else None,
            "p95_latency_ms_at_peak": best.e2e_ms.get("p95") if best else None,
            "ttft_p50_ms_at_peak": best.ttft_ms.get("p50") if best else None,
            "total_failures": sum(r.failures for r in results),
            "total_rate_limited": sum(r.rate_limited for r in results),
            # A run with any 429s measured the gateway's limiter, not the model.
            "valid": sum(r.rate_limited for r in results) == 0,
        },
    }

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        log.info("benchmark report written to %s", out_path)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    """Benchmark table for docs/results.md."""
    lines = [
        "## Serving benchmarks",
        "",
        f"- Endpoint: `{report['endpoint']}`  ",
        f"- Model: `{report['model']}`  ",
        f"- Streaming: `{report['streaming']}`, max_tokens `{report['max_tokens']}`, "
        f"{report['requests_per_level']} requests per level  ",
        f"- Run: `{report['created_at']}`",
        "",
        "| Concurrency | req/s | out tok/s | TTFT p50 | TTFT p95 | e2e p50 | e2e p95 | e2e p99 | fail | 429 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level in report["levels"]:
        lines.append(
            f"| {level['concurrency']} | {level['requests_per_s']:.2f} | "
            f"{level['output_tokens_per_s']:.1f} | "
            f"{level['ttft_ms'].get('p50', 0):.0f} ms | {level['ttft_ms'].get('p95', 0):.0f} ms | "
            f"{level['e2e_ms'].get('p50', 0):.0f} ms | {level['e2e_ms'].get('p95', 0):.0f} ms | "
            f"{level['e2e_ms'].get('p99', 0):.0f} ms | "
            f"{level['failures'] - level.get('rate_limited', 0)} | {level.get('rate_limited', 0)} |"
        )
    s = report["summary"]
    if not s.get("valid", True):
        lines += [
            "",
            f"> **These numbers are not valid.** {s['total_rate_limited']} requests were "
            "rate limited (429) by the gateway, so the sweep measured the rate limiter "
            "rather than the model. Re-run with a benchmarking key: "
            "`dllm serve keys --rpm 100000 --tokens-per-min 100000000`.",
        ]
    lines += [
        "",
        f"Peak decode throughput: **{s['peak_output_tokens_per_s']:.1f} output tokens/s** "
        f"at concurrency {s['peak_at_concurrency']} "
        f"(p50 {s['p50_latency_ms_at_peak']} ms, p95 {s['p95_latency_ms_at_peak']} ms).",
        "",
        "Reproduce with `make bench`.",
        "",
    ]
    return "\n".join(lines)
