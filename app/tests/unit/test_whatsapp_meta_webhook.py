import hmac
from hashlib import sha256

from app.channels.whatsapp_meta.webhook import validate_meta_signature, verify_meta_webhook_token


def test_verify_meta_webhook_token_matches_expected() -> None:
    assert verify_meta_webhook_token("secret", "secret")
    assert not verify_meta_webhook_token("wrong", "secret")


def test_validate_meta_signature_accepts_valid_signature() -> None:
    payload = b'{"object":"whatsapp_business_account"}'
    app_secret = "meta-secret"
    digest = hmac.new(app_secret.encode("utf-8"), payload, sha256).hexdigest()
    signature = f"sha256={digest}"

    assert validate_meta_signature(payload, signature, app_secret)
    assert not validate_meta_signature(payload, "sha256=bad", app_secret)
