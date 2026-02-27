from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from datetime import UTC, datetime

_thread_id_ctx: ContextVar[str | None] = ContextVar("thread_id", default=None)
_run_id_ctx: ContextVar[str | None] = ContextVar("run_id", default=None)

_LEVEL_COLORS = {
    "DEBUG": "\033[36m",   # cyan
    "INFO": "\033[32m",    # green
    "WARNING": "\033[33m", # yellow
    "ERROR": "\033[31m",   # red
    "CRITICAL": "\033[35m",# magenta
}
_COLOR_RESET = "\033[0m"


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


class PrettyFormatter(logging.Formatter):
    def __init__(self, *, colorized: bool = True) -> None:
        super().__init__()
        self._colorized = colorized

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now(UTC).strftime("%H:%M:%S")
        thread_id = _thread_id_ctx.get() or "-"
        run_id = _run_id_ctx.get() or "-"
        level = record.levelname.upper()
        level_text = f"{level:<8}"
        if self._colorized:
            color = _LEVEL_COLORS.get(level, "")
            if color:
                level_text = f"{color}{level_text}{_COLOR_RESET}"

        line = (
            f"{timestamp} | {level_text} | {record.name} | "
            f"thread={thread_id} run={run_id} | {record.getMessage()}"
        )
        if record.exc_info:
            return f"{line}\n{self.formatException(record.exc_info)}"
        return line


def set_correlation(thread_id: str | None = None, run_id: str | None = None) -> None:
    _thread_id_ctx.set(thread_id)
    _run_id_ctx.set(run_id)


def configure_logging(
    level: str = "INFO",
    *,
    log_format: str = "json",
    colorized: bool = True,
) -> None:
    handler = logging.StreamHandler()
    resolved_format = (log_format or "json").strip().lower()
    if resolved_format == "pretty":
        handler.setFormatter(PrettyFormatter(colorized=colorized))
    else:
        handler.setFormatter(ECSJsonFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level.upper())
