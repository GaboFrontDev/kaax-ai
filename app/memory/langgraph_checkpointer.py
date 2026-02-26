from __future__ import annotations

import asyncio
from typing import Any


class LangGraphCheckpointerManager:
    def __init__(self, conn_string: str) -> None:
        self._conn_string = conn_string
        self._checkpointer_cm: Any | None = None
        self._checkpointer: Any | None = None
        self._lock = asyncio.Lock()

    async def get_checkpointer(self) -> Any:
        if self._checkpointer is not None:
            return self._checkpointer

        async with self._lock:
            if self._checkpointer is not None:
                return self._checkpointer

            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            self._checkpointer_cm = AsyncPostgresSaver.from_conn_string(self._conn_string)
            self._checkpointer = await self._checkpointer_cm.__aenter__()
            await self._checkpointer.setup()
            return self._checkpointer

    async def is_healthy(self) -> bool:
        try:
            await self.get_checkpointer()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        async with self._lock:
            cm = self._checkpointer_cm
            self._checkpointer_cm = None
            self._checkpointer = None

        if cm is not None:
            await cm.__aexit__(None, None, None)

    @staticmethod
    def backend_name() -> str:
        return "langgraph-postgres"
