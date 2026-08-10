"""API-key authentication for the gateway.

Keys are stored as SHA-256 hashes, never in plaintext. ``dllm serve keys``
prints a key once at mint time; if it is lost, mint another. The store is a
JSON file or a JSON blob in an environment variable, so it works the same in a
container, in CI and on a laptop without dragging in a database.

Comparison is constant-time. A timing side-channel on a bearer token is
unlikely to be the weak link here, but it costs one function call to remove.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

KEY_PREFIX = "dllm_sk_"
ENV_KEYS = "DLLM_GATEWAY_KEYS"
ENV_KEYS_FILE = "DLLM_GATEWAY_KEYS_FILE"


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ApiKey:
    key_id: str
    key_hash: str
    tier: str = "default"
    rpm: int | None = None
    tokens_per_min: int | None = None


class KeyStore:
    """Loaded once at startup; small enough to scan linearly."""

    def __init__(self, keys: list[ApiKey]) -> None:
        self._by_hash = {k.key_hash: k for k in keys}

    def __len__(self) -> int:
        return len(self._by_hash)

    @property
    def empty(self) -> bool:
        return not self._by_hash

    def verify(self, raw_key: str) -> ApiKey | None:
        """Return the matching key record, or None."""
        if not raw_key:
            return None
        candidate = hash_key(raw_key)
        for stored_hash, record in self._by_hash.items():
            if hmac.compare_digest(candidate, stored_hash):
                return record
        return None

    # ------------------------------------------------------------------ #
    @classmethod
    def from_entries(cls, entries: list[dict]) -> KeyStore:
        keys = [
            ApiKey(
                key_id=e["key_id"],
                key_hash=e["key_hash"],
                tier=e.get("tier", "default"),
                rpm=e.get("rpm"),
                tokens_per_min=e.get("tokens_per_min"),
            )
            for e in entries
        ]
        return cls(keys)

    @classmethod
    def load(cls) -> KeyStore:
        """Load from ``DLLM_GATEWAY_KEYS`` (JSON) or ``DLLM_GATEWAY_KEYS_FILE``."""
        blob = os.environ.get(ENV_KEYS)
        if blob:
            try:
                return cls.from_entries(json.loads(blob))
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"{ENV_KEYS} is not a valid key list: {e}") from e

        path = os.environ.get(ENV_KEYS_FILE)
        if path:
            p = Path(path)
            if not p.is_file():
                raise FileNotFoundError(f"{ENV_KEYS_FILE} points at {p}, which does not exist")
            return cls.from_entries(json.loads(p.read_text(encoding="utf-8")))

        log.warning(
            "no API keys configured (%s / %s unset). The gateway will refuse every "
            "request rather than run open to the internet.",
            ENV_KEYS,
            ENV_KEYS_FILE,
        )
        return cls([])


def mint_keys(count: int = 1, *, tier: str = "default") -> list[dict]:
    """Generate keys. The plaintext is returned once and never stored."""
    out: list[dict] = []
    for _ in range(count):
        raw = KEY_PREFIX + secrets.token_urlsafe(32)
        out.append(
            {
                "key_id": secrets.token_hex(8),
                "key": raw,  # show once; store only the hash
                "key_hash": hash_key(raw),
                "tier": tier,
            }
        )
    return out


def extract_bearer(header: str | None) -> str:
    """Pull the token out of an ``Authorization: Bearer <token>`` header."""
    if not header:
        return ""
    parts = header.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return ""
    return parts[1].strip()
