"""Stateless, self-verifying API keys.

A key is an HMAC-signed token:

    mch_<base64url(owner|scope|issued|expires|nonce)>.<hmac-sha256 hex[:32]>

Both the dashboard (which issues keys) and the API (which verifies them) hold
the same secret, MCH_API_SECRET, and nothing else — no database, no key
table, no shared disk. That is what lets the two run as separate services on
free hosting and still agree on which keys are valid. Revocation is a
comma-separated list of key ids in MCH_REVOKED_KEYS; a key id is the first
eight characters of its signature and is safe to log.

If MCH_API_SECRET is unset, `resolve_secret()` generates a random one for the
lifetime of the process. That is fine for a single container that runs both
services (deploy/start.sh does exactly this), and means nothing at all for two
separate deployments — they will simply not recognise each other's keys.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import time
from dataclasses import asdict, dataclass

PREFIX = "mch_"
SIG_LEN = 32
_SEP = "|"


class InvalidKey(ValueError):
    """The key is malformed, forged, expired or revoked. `.reason` says which."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class KeyInfo:
    key_id: str
    owner: str
    scope: str
    issued_at: int
    expires_at: int | None

    def as_dict(self) -> dict:
        return asdict(self)


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(secret: str, body: str) -> str:
    return hmac.new(secret.encode("utf-8"), body.encode("ascii"),
                    hashlib.sha256).hexdigest()[:SIG_LEN]


def resolve_secret() -> str:
    """MCH_API_SECRET, or a per-process random secret when unset."""
    secret = os.environ.get("MCH_API_SECRET", "").strip()
    if not secret:
        secret = secrets.token_urlsafe(32)
        os.environ["MCH_API_SECRET"] = secret
    return secret


def secret_is_configured() -> bool:
    return bool(os.environ.get("MCH_API_SECRET", "").strip())


def revoked_ids() -> set[str]:
    raw = os.environ.get("MCH_REVOKED_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}


def issue_key(secret: str, owner: str, scope: str = "predict",
              ttl_days: int | None = None) -> tuple[str, KeyInfo]:
    owner = (owner or "anonymous").strip()[:64].replace(_SEP, "/")
    scope = scope.strip()[:16].replace(_SEP, "/") or "predict"
    issued = int(time.time())
    expires = issued + int(ttl_days) * 86400 if ttl_days else 0
    nonce = secrets.token_hex(4)
    body = _b64e(_SEP.join(
        [owner, scope, str(issued), str(expires), nonce]).encode("utf-8"))
    sig = _sign(secret, body)
    key = f"{PREFIX}{body}.{sig}"
    return key, KeyInfo(key_id=sig[:8], owner=owner, scope=scope,
                        issued_at=issued, expires_at=expires or None)


def verify_key(secret: str, key: str,
               revoked: set[str] | None = None) -> KeyInfo:
    key = (key or "").strip()
    if not key.startswith(PREFIX) or "." not in key:
        raise InvalidKey("malformed")
    body, _, sig = key[len(PREFIX):].partition(".")
    if len(sig) != SIG_LEN or not hmac.compare_digest(_sign(secret, body), sig):
        raise InvalidKey("bad signature")
    try:
        owner, scope, issued, expires, _nonce = (
            _b64d(body).decode("utf-8").split(_SEP))
        issued_at, expires_at = int(issued), int(expires)
    except Exception as exc:  # noqa: BLE001 — any parse failure is the same
        raise InvalidKey("malformed") from exc
    if expires_at and time.time() > expires_at:
        raise InvalidKey("expired")
    key_id = sig[:8]
    if key_id in (revoked if revoked is not None else revoked_ids()):
        raise InvalidKey("revoked")
    return KeyInfo(key_id=key_id, owner=owner, scope=scope,
                   issued_at=issued_at, expires_at=expires_at or None)


def mask(key: str) -> str:
    """For display: the prefix and the last six characters only."""
    if len(key) < 14:
        return "•" * len(key)
    return f"{key[:8]}…{key[-6:]}"
