from __future__ import annotations

import asyncio
import json
from collections import defaultdict, deque
from datetime import UTC, datetime
from typing import Any, Protocol

from app.infra.redis import RedisSentinelManager


class SlackMessageQueue(Protocol):
    async def enqueue(self, thread_id: str, payload: dict[str, Any]) -> None: ...

    async def pop_next(self, thread_id: str) -> dict[str, Any] | None: ...

    async def is_healthy(self) -> bool: ...

    def backend_name(self) -> str: ...


class InMemorySlackMessageQueue:
    def __init__(self, max_size: int = 200) -> None:
        self._max_size = max_size
        self._queues: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def enqueue(self, thread_id: str, payload: dict[str, Any]) -> None:
        item = {
            "payload": payload,
            "created_at": datetime.now(UTC).isoformat(),
        }
        async with self._lock:
            queue = self._queues[thread_id]
            queue.append(item)
            while len(queue) > self._max_size:
                queue.popleft()

    async def pop_next(self, thread_id: str) -> dict[str, Any] | None:
        async with self._lock:
            queue = self._queues.get(thread_id)
            if not queue:
                return None
            return queue.popleft()

    async def is_healthy(self) -> bool:
        return True

    def backend_name(self) -> str:
        return "memory"


class RedisSlackMessageQueue:
    def __init__(self, redis_manager: RedisSentinelManager, *, max_size: int = 200) -> None:
        self._redis_manager = redis_manager
        self._max_size = max_size

    async def enqueue(self, thread_id: str, payload: dict[str, Any]) -> None:
        redis = await self._redis_manager.get_master()
        key = self._queue_key(thread_id)
        active_threads_key = self._active_threads_key()
        item = json.dumps(
            {
                "payload": payload,
                "thread_id": thread_id,
                "created_at": datetime.now(UTC).isoformat(),
            },
            ensure_ascii=True,
        )

        async with redis.pipeline(transaction=True) as pipeline:
            await pipeline.rpush(key, item)
            await pipeline.ltrim(key, -self._max_size, -1)
            await pipeline.expire(key, 86400)
            await pipeline.sadd(active_threads_key, thread_id)
            await pipeline.expire(active_threads_key, 86400)
            await pipeline.execute()

    async def pop_next(self, thread_id: str) -> dict[str, Any] | None:
        redis = await self._redis_manager.get_master()
        key = self._queue_key(thread_id)
        raw = await redis.lpop(key)
        if raw is None:
            return None

        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None

    async def is_healthy(self) -> bool:
        return await self._redis_manager.ping()

    def backend_name(self) -> str:
        return "redis"

    @staticmethod
    def _queue_key(thread_id: str) -> str:
        return f"slack:queue:{thread_id}"

    @staticmethod
    def _active_threads_key() -> str:
        return "slack:threads"
