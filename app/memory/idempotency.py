from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any


@dataclass(slots=True)
class IdempotencyStatus:
    state: str
    response: dict[str, Any] | None = None


@dataclass(slots=True)
class _Entry:
    state: str
    expires_at: datetime
    response: dict[str, Any] | None = None


class InMemoryIdempotencyStore:
    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._ttl = timedelta(seconds=ttl_seconds)
        self._entries: dict[tuple[str, str], _Entry] = {}
        self._lock = asyncio.Lock()

    async def begin(self, *, thread_id: str, request_id: str) -> IdempotencyStatus:
        now = datetime.now(UTC)
        key = (thread_id, request_id)

        async with self._lock:
            self._prune_locked(now)
            existing = self._entries.get(key)
            if existing is None:
                self._entries[key] = _Entry(state="in_progress", expires_at=now + self._ttl)
                return IdempotencyStatus(state="new")

            if existing.state == "completed":
                response = copy.deepcopy(existing.response) if existing.response is not None else None
                return IdempotencyStatus(state="replay", response=response)

            return IdempotencyStatus(state="in_progress")

    async def complete(self, *, thread_id: str, request_id: str, response: dict[str, Any]) -> None:
        now = datetime.now(UTC)
        key = (thread_id, request_id)

        async with self._lock:
            self._entries[key] = _Entry(
                state="completed",
                response=copy.deepcopy(response),
                expires_at=now + self._ttl,
            )

    async def fail(self, *, thread_id: str, request_id: str) -> None:
        key = (thread_id, request_id)
        async with self._lock:
            self._entries.pop(key, None)

    async def cleanup_expired(self) -> None:
        async with self._lock:
            self._prune_locked(datetime.now(UTC))

    def _prune_locked(self, now: datetime) -> None:
        for key in list(self._entries.keys()):
            if self._entries[key].expires_at < now:
                self._entries.pop(key, None)
