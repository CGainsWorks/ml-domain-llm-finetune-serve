"""Gateway auth and rate limiting."""

from __future__ import annotations

import asyncio
import json

import pytest

from dllm.serve.gateway.auth import KeyStore, extract_bearer, hash_key, mint_keys
from dllm.serve.gateway.ratelimit import Bucket, RateLimiter, estimate_request_tokens


class TestKeyMinting:
    def test_plaintext_is_not_the_stored_value(self) -> None:
        entry = mint_keys(1)[0]
        assert entry["key_hash"] == hash_key(entry["key"])
        assert entry["key"] != entry["key_hash"]

    def test_keys_are_unique(self) -> None:
        keys = {e["key"] for e in mint_keys(20)}
        assert len(keys) == 20


class TestKeyStore:
    def test_accepts_a_valid_key(self) -> None:
        entry = mint_keys(1)[0]
        store = KeyStore.from_entries([{k: v for k, v in entry.items() if k != "key"}])
        assert store.verify(entry["key"]) is not None

    def test_rejects_an_unknown_key(self) -> None:
        entry = mint_keys(1)[0]
        store = KeyStore.from_entries([{k: v for k, v in entry.items() if k != "key"}])
        assert store.verify("dllm_sk_wrong") is None

    def test_rejects_empty(self) -> None:
        store = KeyStore.from_entries([])
        assert store.verify("") is None
        assert store.empty

    def test_loads_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        entry = mint_keys(1)[0]
        stored = [{k: v for k, v in entry.items() if k != "key"}]
        monkeypatch.setenv("DLLM_GATEWAY_KEYS", json.dumps(stored))
        store = KeyStore.load()
        assert store.verify(entry["key"]) is not None

    def test_bare_object_instead_of_array_is_explained(self) -> None:
        """The single most likely misconfiguration, and it must say so."""
        entry = mint_keys(1)[0]
        with pytest.raises(ValueError, match="must be a JSON array"):
            KeyStore.from_entries({k: v for k, v in entry.items() if k != "key"})

    def test_missing_required_field_names_it(self) -> None:
        with pytest.raises(ValueError, match="key_hash"):
            KeyStore.from_entries([{"key_id": "abc"}])

    def test_non_object_entry_is_explained(self) -> None:
        with pytest.raises(ValueError, match="must be an object"):
            KeyStore.from_entries(["not-an-object"])

    def test_missing_config_yields_empty_store(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No keys must mean 'refuse everything', never 'allow everything'."""
        monkeypatch.delenv("DLLM_GATEWAY_KEYS", raising=False)
        monkeypatch.delenv("DLLM_GATEWAY_KEYS_FILE", raising=False)
        assert KeyStore.load().empty


class TestBearerHeader:
    @pytest.mark.parametrize(
        ("header", "expected"),
        [
            ("Bearer abc123", "abc123"),
            ("bearer abc123", "abc123"),
            ("Basic abc123", ""),
            ("abc123", ""),
            (None, ""),
            ("Bearer", ""),
        ],
    )
    def test_extraction(self, header: str | None, expected: str) -> None:
        assert extract_bearer(header) == expected


class TestBucket:
    def test_starts_full(self) -> None:
        assert Bucket(capacity=5, refill_per_second=1).try_consume(5, now=0)

    def test_refuses_when_empty(self) -> None:
        b = Bucket(capacity=2, refill_per_second=1)
        assert b.try_consume(2, now=0)
        assert not b.try_consume(1, now=0)

    def test_refills_continuously(self) -> None:
        b = Bucket(capacity=10, refill_per_second=1)
        assert b.try_consume(10, now=0)
        assert not b.try_consume(1, now=0)
        assert b.try_consume(1, now=1.5)

    def test_does_not_exceed_capacity(self) -> None:
        b = Bucket(capacity=5, refill_per_second=100)
        b.try_consume(5, now=0)
        assert b.try_consume(5, now=1000)
        assert not b.try_consume(1, now=1000)

    def test_retry_after_is_positive_when_empty(self) -> None:
        b = Bucket(capacity=2, refill_per_second=1)
        b.try_consume(2, now=0)
        assert b.retry_after(1, now=0) == pytest.approx(1.0, abs=0.01)


class TestRateLimiter:
    def test_allows_within_burst(self) -> None:
        limiter = RateLimiter(rpm=60, burst=5, tokens_per_min=100_000)

        async def go():
            return await asyncio.gather(*(limiter.check("k", estimated_tokens=1) for _ in range(5)))

        assert all(r.allowed for r in asyncio.run(go()))

    def test_blocks_past_burst(self) -> None:
        limiter = RateLimiter(rpm=60, burst=2, tokens_per_min=100_000)

        async def go():
            out = []
            for _ in range(4):
                out.append(await limiter.check("k", estimated_tokens=1))
            return out

        results = asyncio.run(go())
        assert [r.allowed for r in results] == [True, True, False, False]
        assert results[2].limit == "requests"
        assert results[2].retry_after > 0

    def test_token_budget_blocks_independently(self) -> None:
        """A single huge request must be refused even with requests to spare."""
        limiter = RateLimiter(rpm=600, burst=100, tokens_per_min=1000)
        first = asyncio.run(limiter.check("k", estimated_tokens=900))
        second = asyncio.run(limiter.check("k", estimated_tokens=900))
        assert first.allowed
        assert not second.allowed
        assert second.limit == "tokens"

    def test_blocked_token_request_refunds_its_request_slot(self) -> None:
        limiter = RateLimiter(rpm=60, burst=2, tokens_per_min=100)

        async def go():
            blocked = await limiter.check("k", estimated_tokens=500)
            allowed = await limiter.check("k", estimated_tokens=10)
            return blocked, allowed

        blocked, allowed = asyncio.run(go())
        assert not blocked.allowed
        assert allowed.allowed, "a token-rejected request must not burn a request slot"

    def test_keys_are_isolated(self) -> None:
        limiter = RateLimiter(rpm=60, burst=1, tokens_per_min=100_000)

        async def go():
            a = await limiter.check("key-a", estimated_tokens=1)
            b = await limiter.check("key-a", estimated_tokens=1)
            c = await limiter.check("key-b", estimated_tokens=1)
            return a, b, c

        a, b, c = asyncio.run(go())
        assert a.allowed and not b.allowed and c.allowed

    def test_per_key_override(self) -> None:
        limiter = RateLimiter(rpm=1, burst=1, tokens_per_min=10)

        async def go():
            return [
                await limiter.check("vip", estimated_tokens=1, rpm=600, tokens_per_min=100_000)
                for _ in range(3)
            ]

        assert all(r.allowed for r in asyncio.run(go()))

    def test_refund_restores_budget(self) -> None:
        limiter = RateLimiter(rpm=600, burst=100, tokens_per_min=1000)

        async def go():
            await limiter.check("k", estimated_tokens=900)
            await limiter.refund_tokens("k", 800)
            return await limiter.check("k", estimated_tokens=700)

        assert asyncio.run(go()).allowed


class TestTokenEstimate:
    def test_scales_with_prompt_and_output(self) -> None:
        assert estimate_request_tokens(0, 0) >= 1
        assert estimate_request_tokens(3600, 100) == pytest.approx(1100, abs=5)

    def test_never_zero(self) -> None:
        assert estimate_request_tokens(0, 0) == 1
