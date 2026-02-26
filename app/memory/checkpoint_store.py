from __future__ import annotations

import asyncio
import copy
import json
from typing import Any, Protocol

from app.infra.db import PostgresPoolManager


class CheckpointStore(Protocol):
    async def setup(self) -> None: ...

    async def get_state(self, thread_id: str) -> dict[str, Any] | None: ...

    async def put_state(self, thread_id: str, state: dict[str, Any]) -> None: ...

    async def delete_state(self, thread_id: str) -> None: ...

    async def is_healthy(self) -> bool: ...

    def backend_name(self) -> str: ...


class InMemoryCheckpointStore:
    def __init__(self) -> None:
        self._states: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def setup(self) -> None:
        return None

    async def get_state(self, thread_id: str) -> dict[str, Any] | None:
        async with self._lock:
            state = self._states.get(thread_id)
            return copy.deepcopy(state) if state is not None else None

    async def put_state(self, thread_id: str, state: dict[str, Any]) -> None:
        async with self._lock:
            self._states[thread_id] = copy.deepcopy(state)

    async def delete_state(self, thread_id: str) -> None:
        async with self._lock:
            self._states.pop(thread_id, None)

    async def is_healthy(self) -> bool:
        return True

    def backend_name(self) -> str:
        return "memory"


class PostgresCheckpointStore:
    def __init__(self, pool_manager: PostgresPoolManager, table_name: str = "agent_checkpoints") -> None:
        self._pool_manager = pool_manager
        self._table_name = table_name
        self._setup_done = False
        self._setup_lock = asyncio.Lock()

    async def setup(self) -> None:
        if self._setup_done:
            return

        async with self._setup_lock:
            if self._setup_done:
                return

            connection = await self._pool_manager.acquire()
            try:
                await connection.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table_name} (
                        thread_id TEXT PRIMARY KEY,
                        state JSONB NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
            finally:
                await self._pool_manager.release(connection)

            self._setup_done = True

    async def get_state(self, thread_id: str) -> dict[str, Any] | None:
        await self.setup()

        connection = await self._pool_manager.acquire()
        try:
            row = await connection.fetchrow(
                f"SELECT state FROM {self._table_name} WHERE thread_id = $1",
                thread_id,
            )
        finally:
            await self._pool_manager.release(connection)

        if row is None:
            return None

        raw_state = row["state"]
        if isinstance(raw_state, str):
            parsed = json.loads(raw_state)
        else:
            parsed = raw_state

        if not isinstance(parsed, dict):
            return None
        return parsed

    async def put_state(self, thread_id: str, state: dict[str, Any]) -> None:
        await self.setup()
        state_json = json.dumps(state, ensure_ascii=True)

        connection = await self._pool_manager.acquire()
        try:
            await connection.execute(
                f"""
                INSERT INTO {self._table_name} (thread_id, state, updated_at)
                VALUES ($1, $2::jsonb, NOW())
                ON CONFLICT (thread_id)
                DO UPDATE SET state = EXCLUDED.state, updated_at = NOW()
                """,
                thread_id,
                state_json,
            )
        finally:
            await self._pool_manager.release(connection)

    async def delete_state(self, thread_id: str) -> None:
        await self.setup()

        connection = await self._pool_manager.acquire()
        try:
            await connection.execute(
                f"DELETE FROM {self._table_name} WHERE thread_id = $1",
                thread_id,
            )
        finally:
            await self._pool_manager.release(connection)

    async def is_healthy(self) -> bool:
        try:
            await self.setup()
        except Exception:
            return False
        return await self._pool_manager.ping()

    def backend_name(self) -> str:
        return "postgres"
