"""PostgreSQL helpers and optional asyncpg pool manager."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    dsn: str
    min_pool_size: int = 1
    max_pool_size: int = 10
    command_timeout_seconds: float = 30.0


class PostgresPoolManager:
    def __init__(self, config: DatabaseConfig) -> None:
        self._config = config
        self._pool: Any | None = None
        self._lock = asyncio.Lock()

    @property
    def dsn(self) -> str:
        return self._config.dsn

    async def get_pool(self) -> Any:
        try:
            import asyncpg
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("asyncpg is required for postgres checkpoint backend") from exc

        if self._pool is not None:
            return self._pool

        async with self._lock:
            if self._pool is None:
                self._pool = await asyncpg.create_pool(
                    dsn=self._config.dsn,
                    min_size=self._config.min_pool_size,
                    max_size=self._config.max_pool_size,
                    command_timeout=self._config.command_timeout_seconds,
                )
        return self._pool

    async def acquire(self) -> Any:
        pool = await self.get_pool()
        return await pool.acquire()

    async def release(self, connection: Any) -> None:
        pool = await self.get_pool()
        await pool.release(connection)

    async def ping(self) -> bool:
        try:
            connection = await self.acquire()
            try:
                await connection.execute("SELECT 1")
            finally:
                await self.release(connection)
            return True
        except Exception:
            return False

    async def close(self) -> None:
        if self._pool is None:
            return
        async with self._lock:
            if self._pool is None:
                return
            await self._pool.close()
            self._pool = None


def build_postgres_dsn(
    *,
    db_dsn: str | None,
    user: str,
    password: str,
    host: str,
    port: int,
    db_name: str,
    ssl_mode: str | None = None,
) -> str:
    if db_dsn:
        return db_dsn

    encoded_user = quote_plus(user)
    encoded_password = quote_plus(password)
    base = f"postgresql://{encoded_user}:{encoded_password}@{host}:{port}/{db_name}"
    if ssl_mode:
        return f"{base}?sslmode={quote_plus(ssl_mode)}"
    return base
