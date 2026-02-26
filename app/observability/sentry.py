from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def configure_sentry(dsn: str | None) -> None:
    if not dsn:
        return

    try:
        import sentry_sdk
    except ImportError:
        logger.warning("Sentry DSN configured but sentry-sdk is not installed")
        return

    sentry_sdk.init(dsn=dsn)
    logger.info("Sentry configured")
