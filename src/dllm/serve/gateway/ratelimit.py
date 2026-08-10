"""Per-key rate limiting: a request bucket and a token bucket.

Two limits, because they stop different abuse. A request-per-minute cap stops
someone hammering the endpoint with trivial prompts. It does nothing about one
request asking for 4,000 output tokens every second, which is what actually
saturates a GPU -- hence the second bucket, denominated in tokens.

Both are leaky token buckets: they refill continuously rather than resetting on
a wall-clock boundary, so a caller cannot burst 2x the limit by straddling the
top of a minute.

**Known limitation, stated rather than hidden:** this is in-process state. Two
gateway replicas each enforce the limit independently, so the effective limit
is N times the configured one. Single-replica deployments are correct; for
horizontal scale the buckets belong in Redis. See docs/MODEL_CARD.md.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class Bucket:
    """A leaky token bucket.

    ``updated`` starts unset rather than at ``time.monotonic()``. The clock is
    whatever the first caller passes in, so the bucket behaves identically under
    the real monotonic clock and under an injected one in tests. Seeding it from
    the real clock and then being handed ``now=0`` produces a negative elapsed
    time and a bucket that never refills.
    """

    capacity: float
    refill_per_second: float
    tokens: float = field(default=0.0)
    updated: float | None = field(default=None)

    def __post_init__(self) -> None:
        if self.tokens == 0.0:
            self.tokens = self.capacity

    def _refill(self, now: float) -> None:
        if self.updated is None:
            self.updated = now
            return
        elapsed = now - self.updated
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_second)
            self.updated = now

    def try_consume(self, amount: float, now: float | None = None) -> bool:
        now = now if now is not None else time.monotonic()
        self._refill(now)
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False

    def retry_after(self, amount: float, now: float | None = None) -> float:
        """Seconds until ``amount`` would be available."""
        now = now if now is not None else time.monotonic()
        self._refill(now)
        if self.tokens >= amount:
            return 0.0
        if self.refill_per_second <= 0:
            return float("inf")
        return (amount - self.tokens) / self.refill_per_second


@dataclass
class Decision:
    allowed: bool
    reason: str = ""
    retry_after: float = 0.0
    limit: str = ""


class RateLimiter:
    """Per-key request and token buckets."""

    def __init__(self, *, rpm: int, burst: int, tokens_per_min: int) -> None:
        self.rpm = rpm
        # Burst is the bucket depth; sustained rate is rpm/60 per second.
        self.burst = max(burst, 1)
        self.tokens_per_min = tokens_per_min
        self._requests: dict[str, Bucket] = {}
        self._tokens: dict[str, Bucket] = {}
        self._lock = asyncio.Lock()

    def _buckets(self, key_id: str, rpm: int | None, tpm: int | None) -> tuple[Bucket, Bucket]:
        effective_rpm = rpm or self.rpm
        effective_tpm = tpm or self.tokens_per_min
        if key_id not in self._requests:
            # Scale burst depth with the key's sustained rate. A key raised from
            # 60 to 600 rpm that kept the default burst of 10 would still be
            # throttled on its eleventh concurrent request, which is not what
            # "this key has a higher limit" is supposed to mean.
            scale = effective_rpm / self.rpm if self.rpm > 0 else 1.0
            self._requests[key_id] = Bucket(
                capacity=max(1.0, self.burst * scale),
                refill_per_second=effective_rpm / 60.0,
            )
            self._tokens[key_id] = Bucket(
                capacity=effective_tpm, refill_per_second=effective_tpm / 60.0
            )
        return self._requests[key_id], self._tokens[key_id]

    async def check(
        self,
        key_id: str,
        *,
        estimated_tokens: int,
        rpm: int | None = None,
        tokens_per_min: int | None = None,
    ) -> Decision:
        """Consume one request and ``estimated_tokens`` tokens, if available."""
        async with self._lock:
            req_bucket, tok_bucket = self._buckets(key_id, rpm, tokens_per_min)
            now = time.monotonic()

            if not req_bucket.try_consume(1, now):
                return Decision(
                    False,
                    f"request rate limit exceeded ({rpm or self.rpm}/min)",
                    round(req_bucket.retry_after(1, now), 2),
                    "requests",
                )

            if not tok_bucket.try_consume(estimated_tokens, now):
                # Give the request slot back: it was not served.
                req_bucket.tokens = min(req_bucket.capacity, req_bucket.tokens + 1)
                return Decision(
                    False,
                    f"token rate limit exceeded "
                    f"({tokens_per_min or self.tokens_per_min} tokens/min)",
                    round(tok_bucket.retry_after(estimated_tokens, now), 2),
                    "tokens",
                )

            return Decision(True)

    async def refund_tokens(self, key_id: str, amount: float) -> None:
        """Return unused token budget after a response comes back smaller than estimated."""
        if amount <= 0:
            return
        async with self._lock:
            bucket = self._tokens.get(key_id)
            if bucket:
                bucket.tokens = min(bucket.capacity, bucket.tokens + amount)


def estimate_request_tokens(prompt_chars: int, max_tokens: int) -> int:
    """Cheap up-front cost estimate for a request.

    Charged before the model runs, because the limiter has to decide before
    generation starts. ~3.6 characters per token is a reasonable English/SQL
    average; the response path refunds whatever was over-charged.
    """
    return max(1, int(prompt_chars / 3.6) + max(max_tokens, 0))
