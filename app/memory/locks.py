from __future__ import annotations

import asyncio
import hashlib
import struct
import time
from typing import Any, Protocol

from app.infra.db import PostgresPoolManager


class SessionLockManager(Protocol):
    async def acquire(self, thread_id: str, timeout: float = 0) -> bool: ...

    async def release(self, thread_id: str) -> None: ...

    async def is_healthy(self) -> bool: ...

    def backend_name(self) -> str: ...


def advisory_lock_key(thread_id: str) -> int:
    digest = hashlib.blake2b(thread_id.encode("utf-8"), digest_size=8).digest()
    return struct.unpack(">q", digest)[0]


class InMemorySessionLockManager:
    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()

    async def _get_lock(self, thread_id: str) -> asyncio.Lock:
        async with self._registry_lock:
            lock = self._locks.get(thread_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[thread_id] = lock
            return lock

    async def acquire(self, thread_id: str, timeout: float = 0) -> bool:
        lock = await self._get_lock(thread_id)
        if timeout <= 0:
            if lock.locked():
                return False
            await lock.acquire()
            return True

        try:
            await asyncio.wait_for(lock.acquire(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def release(self, thread_id: str) -> None:
        lock = await self._get_lock(thread_id)
        if lock.locked():
            lock.release()

    async def is_healthy(self) -> bool:
        return True

    def backend_name(self) -> str:
        return "memory"


class PostgresSessionLockManager:
    def __init__(self, pool_manager: PostgresPoolManager, poll_interval_seconds: float = 0.05) -> None:
        self._pool_manager = pool_manager
        self._poll_interval_seconds = max(0.01, poll_interval_seconds)
        self._held_connections: dict[str, Any] = {}
        self._registry_lock = asyncio.Lock()

    async def acquire(self, thread_id: str, timeout: float = 0) -> bool:
        async with self._registry_lock:
            if thread_id in self._held_connections:
                return False

        connection = await self._pool_manager.acquire()
        key = advisory_lock_key(thread_id)

        try:
            acquired = await self._try_acquire(connection, key=key, timeout=timeout)
            if not acquired:
                await self._pool_manager.release(connection)
                return False

            async with self._registry_lock:
                self._held_connections[thread_id] = connection
            return True
        except Exception:
            await self._pool_manager.release(connection)
            raise

    async def release(self, thread_id: str) -> None:
        async with self._registry_lock:
            connection = self._held_connections.pop(thread_id, None)

        if connection is None:
            return

        key = advisory_lock_key(thread_id)
        try:
            await connection.fetchval("SELECT pg_advisory_unlock($1)", key)
        finally:
            await self._pool_manager.release(connection)

    async def _try_acquire(self, connection: Any, *, key: int, timeout: float) -> bool:
        if timeout <= 0:
            return bool(await connection.fetchval("SELECT pg_try_advisory_lock($1)", key))

        deadline = time.monotonic() + timeout
        while True:
            acquired = bool(await connection.fetchval("SELECT pg_try_advisory_lock($1)", key))
            if acquired:
                return True

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False

            await asyncio.sleep(min(self._poll_interval_seconds, remaining))

    async def is_healthy(self) -> bool:
        return await self._pool_manager.ping()

    def backend_name(self) -> str:
        return "postgres"
