"""FastAPI gateway in front of vLLM.

vLLM's OpenAI-compatible server has no authentication worth the name, no
per-caller rate limiting and no request accounting. Putting it directly on a
network is how you donate a GPU to strangers. This wrapper adds the missing
parts and otherwise stays out of the way: request and response bodies are
passed through unchanged, so any OpenAI client works against it.

What it adds:
  * Bearer-token auth against hashed keys.
  * Per-key request and token rate limits, with ``Retry-After``.
  * A model allowlist, so a caller cannot ask the server for something else.
  * A prompt size cap, checked before the request reaches the GPU.
  * Streaming passthrough (SSE), because that is what a chat UI needs.
  * ``/healthz`` (is the gateway up) and ``/readyz`` (is the model up), which
    are different questions and need different answers for a load balancer.
  * Prometheus metrics, and a request id on every response for correlation.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from dllm.config import GatewayConfig
from dllm.serve.gateway.auth import ApiKey, KeyStore, extract_bearer
from dllm.serve.gateway.ratelimit import RateLimiter, estimate_request_tokens

log = logging.getLogger(__name__)

PASSTHROUGH_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/models")


def _metrics():
    """Build the metric set on a registry private to this app.

    Not the prometheus_client default registry: metric names are global there,
    so constructing a second app in the same process (a test, a config reload,
    two gateways in one worker) raises "Duplicated timeseries". A per-app
    registry makes ``create_app`` an ordinary function with no global side
    effects.
    """
    from prometheus_client import CollectorRegistry, Counter, Histogram

    registry = CollectorRegistry()
    return {
        "registry": registry,
        "requests": Counter(
            "dllm_gateway_requests_total",
            "Requests by route and status",
            ["route", "status"],
            registry=registry,
        ),
        "rejected": Counter(
            "dllm_gateway_rejected_total",
            "Rejected requests by reason",
            ["reason"],
            registry=registry,
        ),
        "latency": Histogram(
            "dllm_gateway_request_seconds",
            "End-to-end latency",
            ["route"],
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120),
            registry=registry,
        ),
    }


def create_app(cfg: GatewayConfig) -> FastAPI:
    keystore = KeyStore.load()
    limiter = RateLimiter(
        rpm=cfg.rate_limit_rpm,
        burst=cfg.rate_limit_burst,
        tokens_per_min=cfg.rate_limit_tokens_per_min,
    )
    metrics = _metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.client = httpx.AsyncClient(
            base_url=cfg.upstream_base_url.rstrip("/"),
            timeout=httpx.Timeout(cfg.request_timeout_s, connect=10.0),
        )
        log.info(
            "gateway up: upstream=%s keys=%d rpm=%d tpm=%d",
            cfg.upstream_base_url,
            len(keystore),
            cfg.rate_limit_rpm,
            cfg.rate_limit_tokens_per_min,
        )
        try:
            yield
        finally:
            await app.state.client.aclose()

    app = FastAPI(
        title="Domain LLM gateway",
        version="0.1.0",
        description="Authenticated, rate-limited OpenAI-compatible endpoint.",
        lifespan=lifespan,
    )

    if cfg.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
        )

    # ---------------------------------------------------------------- #
    # Auth dependency
    # ---------------------------------------------------------------- #
    async def require_key(authorization: str | None = Header(default=None)) -> ApiKey:
        if keystore.empty:
            metrics["rejected"].labels(reason="no_keys_configured").inc()
            raise HTTPException(
                status_code=503,
                detail="gateway has no API keys configured; refusing all requests",
            )
        record = keystore.verify(extract_bearer(authorization))
        if record is None:
            metrics["rejected"].labels(reason="bad_key").inc()
            raise HTTPException(status_code=401, detail="invalid or missing API key")
        return record

    # ---------------------------------------------------------------- #
    # Health
    # ---------------------------------------------------------------- #
    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        """Liveness: is this process serving? Says nothing about the model."""
        return {"status": "ok", "keys_configured": len(keystore)}

    @app.get("/readyz")
    async def readyz(request: Request) -> JSONResponse:
        """Readiness: can the upstream model actually serve a request?"""
        try:
            r = await request.app.state.client.get("/models", timeout=5.0)
            ready = r.status_code == 200
            models = [m["id"] for m in r.json().get("data", [])] if ready else []
        except Exception as e:  # noqa: BLE001
            return JSONResponse(status_code=503, content={"status": "not ready", "error": str(e)})
        return JSONResponse(
            status_code=200 if ready else 503,
            content={"status": "ready" if ready else "not ready", "models": models},
        )

    @app.get("/metrics")
    async def prometheus_metrics() -> PlainTextResponse:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return PlainTextResponse(
            generate_latest(metrics["registry"]).decode(), media_type=CONTENT_TYPE_LATEST
        )

    # ---------------------------------------------------------------- #
    # Proxy
    # ---------------------------------------------------------------- #
    @app.get("/v1/models")
    async def list_models(request: Request, key: ApiKey = Depends(require_key)) -> JSONResponse:
        r = await request.app.state.client.get("/models")
        return JSONResponse(status_code=r.status_code, content=r.json())

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, key: ApiKey = Depends(require_key)):
        return await _proxy(request, key, "/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request, key: ApiKey = Depends(require_key)):
        return await _proxy(request, key, "/completions")

    async def _proxy(request: Request, key: ApiKey, upstream_path: str):
        started = time.perf_counter()
        request_id = str(uuid.uuid4())

        try:
            payload = await request.json()
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="body is not valid JSON") from None
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="body must be a JSON object")

        # --- model allowlist -------------------------------------------- #
        model = payload.get("model")
        if cfg.allowed_models and model not in cfg.allowed_models:
            metrics["rejected"].labels(reason="model_not_allowed").inc()
            raise HTTPException(
                status_code=400,
                detail=f"model {model!r} is not served here. Available: {cfg.allowed_models}",
            )

        # --- size cap, before anything touches the GPU ------------------- #
        prompt_chars = _prompt_chars(payload)
        if prompt_chars > cfg.max_prompt_chars:
            metrics["rejected"].labels(reason="prompt_too_large").inc()
            raise HTTPException(
                status_code=413,
                detail=f"prompt is {prompt_chars} characters, limit is {cfg.max_prompt_chars}",
            )

        # --- rate limit -------------------------------------------------- #
        max_tokens = int(payload.get("max_tokens") or payload.get("max_completion_tokens") or 256)
        estimated = estimate_request_tokens(prompt_chars, max_tokens)
        decision = await limiter.check(
            key.key_id,
            estimated_tokens=estimated,
            rpm=key.rpm,
            tokens_per_min=key.tokens_per_min,
        )
        if not decision.allowed:
            metrics["rejected"].labels(reason=decision.limit).inc()
            return JSONResponse(
                status_code=429,
                headers={
                    "Retry-After": str(max(1, int(decision.retry_after))),
                    "X-Request-Id": request_id,
                },
                content={
                    "error": {
                        "message": decision.reason,
                        "type": "rate_limit_exceeded",
                        "retry_after_seconds": decision.retry_after,
                    }
                },
            )

        client: httpx.AsyncClient = request.app.state.client
        streaming = bool(payload.get("stream"))

        if streaming:

            async def stream() -> AsyncIterator[bytes]:
                async with client.stream("POST", upstream_path, json=payload) as upstream:
                    if upstream.status_code >= 400:
                        body = await upstream.aread()
                        metrics["requests"].labels(
                            route=upstream_path, status=str(upstream.status_code)
                        ).inc()
                        yield body
                        return
                    async for chunk in upstream.aiter_bytes():
                        yield chunk
                metrics["requests"].labels(route=upstream_path, status="200").inc()
                metrics["latency"].labels(route=upstream_path).observe(
                    time.perf_counter() - started
                )

            return StreamingResponse(
                stream(),
                media_type="text/event-stream",
                headers={"X-Request-Id": request_id, "Cache-Control": "no-cache"},
            )

        try:
            upstream = await client.post(upstream_path, json=payload)
        except httpx.RequestError as e:
            metrics["requests"].labels(route=upstream_path, status="502").inc()
            raise HTTPException(status_code=502, detail=f"upstream unreachable: {e}") from e

        elapsed = time.perf_counter() - started
        metrics["requests"].labels(route=upstream_path, status=str(upstream.status_code)).inc()
        metrics["latency"].labels(route=upstream_path).observe(elapsed)

        body = upstream.json()
        # Refund the difference between the estimate and what was actually used,
        # so a caller asking for max_tokens=2000 and getting 30 is not punished.
        usage = body.get("usage") or {}
        actual = usage.get("total_tokens")
        if isinstance(actual, int) and actual < estimated:
            await limiter.refund_tokens(key.key_id, estimated - actual)

        log.info(
            "%s key=%s model=%s status=%d prompt_chars=%d tokens=%s %.3fs id=%s",
            upstream_path,
            key.key_id,
            model,
            upstream.status_code,
            prompt_chars,
            actual,
            elapsed,
            request_id,
        )
        return JSONResponse(
            status_code=upstream.status_code,
            content=body,
            headers={"X-Request-Id": request_id},
        )

    return app


def _prompt_chars(payload: dict) -> int:
    """Total prompt size, for both the chat and completions shapes."""
    if "messages" in payload and isinstance(payload["messages"], list):
        return sum(len(str(m.get("content", ""))) for m in payload["messages"])
    prompt = payload.get("prompt", "")
    if isinstance(prompt, list):
        return sum(len(str(p)) for p in prompt)
    return len(str(prompt))
