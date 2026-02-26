from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, AsyncIterator

from app.memory.checkpoint_store import CheckpointStore
from app.memory.locks import SessionLockManager


class SessionBusyError(RuntimeError):
    pass


class SessionManager:
    def __init__(
        self,
        checkpoint_store: CheckpointStore,
        lock_manager: SessionLockManager,
        *,
        session_timeout_seconds: int,
        lock_timeout_seconds: float = 0,
    ) -> None:
        self._checkpoint_store = checkpoint_store
        self._lock_manager = lock_manager
        self._session_timeout = timedelta(seconds=session_timeout_seconds)
        self._lock_timeout_seconds = lock_timeout_seconds
        self._touched_at: dict[str, datetime] = {}
        self._touch_lock = asyncio.Lock()

    async def get_state(self, thread_id: str) -> dict[str, Any] | None:
        return await self._checkpoint_store.get_state(thread_id)

    async def put_state(self, thread_id: str, state: dict[str, Any]) -> None:
        await self._checkpoint_store.put_state(thread_id, state)
        await self.touch(thread_id)

    async def touch(self, thread_id: str) -> None:
        async with self._touch_lock:
            self._touched_at[thread_id] = datetime.now(UTC)

    @asynccontextmanager
    async def session_lock(self, thread_id: str) -> AsyncIterator[None]:
        acquired = await self._lock_manager.acquire(thread_id, timeout=self._lock_timeout_seconds)
        if not acquired:
            raise SessionBusyError(f"session lock unavailable for {thread_id}")

        try:
            await self.touch(thread_id)
            yield
        finally:
            await self._lock_manager.release(thread_id)

    async def cleanup_expired_sessions(self) -> list[str]:
        cutoff = datetime.now(UTC) - self._session_timeout
        async with self._touch_lock:
            expired = [thread_id for thread_id, touched in self._touched_at.items() if touched < cutoff]
            for thread_id in expired:
                self._touched_at.pop(thread_id, None)

        for thread_id in expired:
            await self._checkpoint_store.delete_state(thread_id)

        return expired
