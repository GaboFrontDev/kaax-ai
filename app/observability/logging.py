from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from datetime import UTC, datetime

_thread_id_ctx: ContextVar[str | None] = ContextVar("thread_id", default=None)
_run_id_ctx: ContextVar[str | None] = ContextVar("run_id", default=None)


class ECSJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, str | int | None] = {
            "@timestamp": datetime.now(UTC).isoformat(),
            "log.level": record.levelname,
            "message": record.getMessage(),
            "log.logger": record.name,
            "thread_id": _thread_id_ctx.get(),
            "run_id": _run_id_ctx.get(),
        }
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def set_correlation(thread_id: str | None = None, run_id: str | None = None) -> None:
    _thread_id_ctx.set(thread_id)
    _run_id_ctx.set(run_id)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(ECSJsonFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level.upper())
