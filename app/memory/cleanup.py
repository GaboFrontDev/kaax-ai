from __future__ import annotations

import asyncio
import random
from contextlib import suppress

from app.memory.attachments_store import AttachmentStore
from app.memory.idempotency import InMemoryIdempotencyStore
from app.memory.session_manager import SessionManager


class CleanupWorker:
    def __init__(
        self,
        session_manager: SessionManager,
        attachment_store: AttachmentStore,
        idempotency_store: InMemoryIdempotencyStore | None = None,
        *,
        interval_seconds: int,
        jitter_seconds: int,
    ) -> None:
        self._session_manager = session_manager
        self._attachment_store = attachment_store
        self._idempotency_store = idempotency_store
        self._interval_seconds = interval_seconds
        self._jitter_seconds = jitter_seconds
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop(), name="session-cleanup")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _loop(self) -> None:
        while True:
            jitter = random.randint(0, self._jitter_seconds) if self._jitter_seconds > 0 else 0
            await asyncio.sleep(self._interval_seconds + jitter)
            await self._session_manager.cleanup_expired_sessions()
            await self._attachment_store.cleanup_expired()
            if self._idempotency_store is not None:
                await self._idempotency_store.cleanup_expired()
