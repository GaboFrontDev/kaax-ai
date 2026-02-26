from __future__ import annotations

import base64
import hmac
from hashlib import sha1


def validate_twilio_signature(
    *,
    url: str,
    params: dict[str, str],
    signature: str,
    auth_token: str,
) -> bool:
    signed = url + "".join(f"{key}{value}" for key, value in sorted(params.items()))
    digest = hmac.new(auth_token.encode("utf-8"), signed.encode("utf-8"), sha1).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, signature)
