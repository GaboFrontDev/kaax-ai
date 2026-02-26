from __future__ import annotations

import asyncio
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from app.infra.redis import RedisSentinelManager


class AttachmentStore(Protocol):
    async def setup(self) -> None: ...

    async def put(self, thread_id: str, files: list[dict[str, Any]]) -> None: ...

    async def get_recent(self, thread_id: str, limit: int) -> list[dict[str, Any]]: ...

    async def cleanup_expired(self) -> None: ...

    async def is_healthy(self) -> bool: ...

    def backend_name(self) -> str: ...


@dataclass(slots=True)
class _Attachment:
    filename: str
    content: str
    type: str
    created_at: datetime


class InMemoryAttachmentStore:
    def __init__(self, max_items: int, ttl_minutes: int) -> None:
        self._max_items = max_items
        self._ttl = timedelta(minutes=ttl_minutes)
        self._data: dict[str, deque[_Attachment]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def setup(self) -> None:
        return None

    async def put(self, thread_id: str, files: list[dict[str, Any]]) -> None:
        if not files:
            return

        now = datetime.now(UTC)
        async with self._lock:
            queue = self._data[thread_id]
            for file in files:
                queue.append(
                    _Attachment(
                        filename=file["filename"],
                        content=file.get("content", ""),
                        type=file.get("type", "application/octet-stream"),
                        created_at=now,
                    )
                )

            while len(queue) > self._max_items:
                queue.popleft()

            self._prune_locked(now)

    async def get_recent(self, thread_id: str, limit: int) -> list[dict[str, Any]]:
        now = datetime.now(UTC)
        async with self._lock:
            self._prune_locked(now)
            queue = self._data.get(thread_id, deque())
            sliced = list(queue)[-limit:]
            return [
                {
                    "filename": item.filename,
                    "content": item.content,
                    "type": item.type,
                    "created_at": item.created_at.isoformat(),
                }
                for item in sliced
            ]

    async def cleanup_expired(self) -> None:
        async with self._lock:
            self._prune_locked(datetime.now(UTC))

    async def is_healthy(self) -> bool:
        return True

    def backend_name(self) -> str:
        return "memory"

    def _prune_locked(self, now: datetime) -> None:
        threshold = now - self._ttl
        for thread_id in list(self._data.keys()):
            queue = self._data[thread_id]
            while queue and queue[0].created_at < threshold:
                queue.popleft()
            if not queue:
                self._data.pop(thread_id, None)


class RedisAttachmentStore:
    def __init__(self, redis_manager: RedisSentinelManager, *, max_items: int, ttl_minutes: int) -> None:
        self._redis_manager = redis_manager
        self._max_items = max_items
        self._ttl_seconds = max(60, ttl_minutes * 60)

    async def setup(self) -> None:
        await self._redis_manager.get_master()

    async def put(self, thread_id: str, files: list[dict[str, Any]]) -> None:
        if not files:
            return

        redis = await self._redis_manager.get_master()
        key = self._key(thread_id)
        now = datetime.now(UTC).isoformat()
        payloads = [
            json.dumps(
                {
                    "filename": file["filename"],
                    "content": file.get("content", ""),
                    "type": file.get("type", "application/octet-stream"),
                    "created_at": now,
                },
                ensure_ascii=True,
            )
            for file in files
        ]

        async with redis.pipeline(transaction=True) as pipeline:
            await pipeline.lpush(key, *payloads)
            await pipeline.ltrim(key, 0, self._max_items - 1)
            await pipeline.expire(key, self._ttl_seconds)
            await pipeline.execute()

    async def get_recent(self, thread_id: str, limit: int) -> list[dict[str, Any]]:
        redis = await self._redis_manager.get_master()
        key = self._key(thread_id)
        raw_items: list[str] = await redis.lrange(key, 0, max(0, limit - 1))

        parsed: list[dict[str, Any]] = []
        for raw in raw_items:
            item = json.loads(raw)
            if isinstance(item, dict):
                parsed.append(item)

        parsed.reverse()
        return parsed

    async def cleanup_expired(self) -> None:
        return None

    async def is_healthy(self) -> bool:
        return await self._redis_manager.ping()

    def backend_name(self) -> str:
        return "redis"

    @staticmethod
    def _key(thread_id: str) -> str:
        return f"attachments:{thread_id}"
