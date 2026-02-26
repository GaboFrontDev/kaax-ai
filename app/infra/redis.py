"""Redis Sentinel helpers for optional attachment/message-queue backends."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RedisSentinelConfig:
    sentinel_nodes: tuple[tuple[str, int], ...]
    master_name: str
    password: str | None = None
    db: int = 0
    socket_timeout_seconds: float = 1.0
    master_host_override: str | None = None
    master_port_override: int | None = None


class RedisSentinelManager:
    def __init__(self, config: RedisSentinelConfig) -> None:
        self._config = config
        self._sentinel: Any | None = None
        self._master: Any | None = None
        self._lock = asyncio.Lock()

    async def get_master(self) -> Any:
        if self._master is not None:
            return self._master

        try:
            import redis.asyncio as redis_asyncio
            from redis.asyncio.sentinel import Sentinel
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError("redis package with asyncio support is required") from exc

        async with self._lock:
            if self._master is not None:
                return self._master

            self._sentinel = Sentinel(
                self._config.sentinel_nodes,
                socket_timeout=self._config.socket_timeout_seconds,
                password=self._config.password,
            )
            host, port = await self._discover_master_endpoint()
            self._master = redis_asyncio.Redis(
                host=host,
                port=port,
                password=self._config.password,
                db=self._config.db,
                decode_responses=True,
                socket_timeout=self._config.socket_timeout_seconds,
            )

        return self._master

    async def _discover_master_endpoint(self) -> tuple[str, int]:
        if self._sentinel is None:
            raise RuntimeError("sentinel is not initialized")

        host, port = await self._sentinel.discover_master(self._config.master_name)
        resolved_host = self._config.master_host_override or host
        resolved_port = self._config.master_port_override or int(port)
        return resolved_host, resolved_port

    async def ping(self) -> bool:
        try:
            master = await self.get_master()
            result = await master.ping()
            return bool(result)
        except Exception:
            await self.close()
            return False

    async def close(self) -> None:
        master = self._master
        sentinel = self._sentinel
        self._master = None
        self._sentinel = None

        if master is not None:
            if hasattr(master, "aclose"):
                await master.aclose()
            elif hasattr(master, "close"):
                maybe_awaitable = master.close()
                if asyncio.iscoroutine(maybe_awaitable):
                    await maybe_awaitable

        if sentinel is not None and hasattr(sentinel, "close"):
            maybe_awaitable = sentinel.close()
            if asyncio.iscoroutine(maybe_awaitable):
                await maybe_awaitable


def parse_sentinel_nodes(raw_nodes: tuple[str, ...]) -> tuple[tuple[str, int], ...]:
    parsed: list[tuple[str, int]] = []
    for raw in raw_nodes:
        item = raw.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"invalid sentinel node '{item}', expected host:port")
        host, port_raw = item.rsplit(":", 1)
        parsed.append((host, int(port_raw)))

    if not parsed:
        raise ValueError("at least one redis sentinel node is required")

    return tuple(parsed)
