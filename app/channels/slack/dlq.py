from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(slots=True)
class DeadLetterEntry:
    payload: dict[str, Any]
    error: str
    created_at: str


class SlackDeadLetterQueue:
    def __init__(self, max_size: int = 200) -> None:
        self._max_size = max_size
        self._queue: deque[DeadLetterEntry] = deque(maxlen=max_size)
        self._lock = asyncio.Lock()

    async def enqueue(self, payload: dict[str, Any], error: str) -> None:
        async with self._lock:
            self._queue.append(
                DeadLetterEntry(
                    payload=payload,
                    error=error,
                    created_at=datetime.now(UTC).isoformat(),
                )
            )

    async def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        async with self._lock:
            items = list(self._queue)[-limit:]
            return [
                {
                    "payload": item.payload,
                    "error": item.error,
                    "created_at": item.created_at,
                }
                for item in items
            ]
