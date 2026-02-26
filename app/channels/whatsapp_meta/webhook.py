from __future__ import annotations

import hmac
from hashlib import sha256


def verify_meta_webhook_token(token: str, expected_token: str) -> bool:
    return token == expected_token


def validate_meta_signature(payload: bytes, signature: str, app_secret: str) -> bool:
    if not signature.startswith("sha256="):
        return False
    provided = signature.split("=", 1)[1].strip()
    expected = hmac.new(app_secret.encode("utf-8"), payload, sha256).hexdigest()
    return hmac.compare_digest(expected, provided)
