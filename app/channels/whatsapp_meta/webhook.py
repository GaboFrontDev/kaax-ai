from __future__ import annotations


def verify_meta_webhook_token(token: str, expected_token: str) -> bool:
    return token == expected_token
