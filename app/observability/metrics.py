from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import json
from typing import Any, Protocol

from app.infra.db import PostgresPoolManager


@dataclass(slots=True)
class InMemoryMetrics:
    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    events: list[dict[str, Any]] = field(default_factory=list)
    max_events: int = 20_000
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def inc(self, name: str, value: int = 1) -> None:
        self.counters[name] += value

    def snapshot(self) -> dict[str, int]:
        return dict(self.counters)

    async def setup(self) -> None:
        return None

    async def record_event(
        self,
        *,
        channel: str,
        user_id: str | None,
        thread_id: str,
        direction: str,
        event_type: str,
        success: bool,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        event_at = datetime.now(UTC)
        event = {
            "event_at": event_at,
            "channel": channel,
            "user_id": user_id,
            "thread_id": thread_id,
            "direction": direction,
            "event_type": event_type,
            "success": success,
            "run_id": run_id,
            "metadata": metadata or {},
        }
        async with self._lock:
            self.events.append(event)
            if len(self.events) > self.max_events:
                del self.events[: len(self.events) - self.max_events]

        self.inc(f"events.{channel}.{direction}")

    async def summarize(
        self,
        *,
        since_hours: int = 24,
        top_users_limit: int = 20,
    ) -> dict[str, Any]:
        cutoff = datetime.now(UTC) - timedelta(hours=since_hours)
        async with self._lock:
            filtered = [event for event in self.events if event["event_at"] >= cutoff]

        inbound = [event for event in filtered if event.get("direction") == "inbound"]
        outbound = [event for event in filtered if event.get("direction") == "outbound"]
        outbound_failed = [event for event in outbound if not bool(event.get("success"))]

        channels: dict[str, dict[str, Any]] = {}
        user_stats: dict[str, dict[str, Any]] = {}
        threads_seen: set[str] = set()

        for event in filtered:
            channel = str(event.get("channel") or "unknown")
            user_id = event.get("user_id")
            thread_id = str(event.get("thread_id") or "")
            direction = str(event.get("direction") or "")
            success = bool(event.get("success"))
            event_at = event.get("event_at")

            if thread_id:
                threads_seen.add(thread_id)

            channel_item = channels.setdefault(
                channel,
                {
                    "channel": channel,
                    "inbound_messages": 0,
                    "outbound_messages": 0,
                    "failed_outbound_messages": 0,
                    "unique_users": set(),
                },
            )

            if direction == "inbound":
                channel_item["inbound_messages"] += 1
            elif direction == "outbound":
                channel_item["outbound_messages"] += 1
                if not success:
                    channel_item["failed_outbound_messages"] += 1

            if user_id:
                channel_item["unique_users"].add(str(user_id))
                user_item = user_stats.setdefault(
                    str(user_id),
                    {
                        "user_id": str(user_id),
                        "inbound_messages": 0,
                        "last_seen": None,
                        "channels": set(),
                    },
                )
                if direction == "inbound":
                    user_item["inbound_messages"] += 1
                if isinstance(event_at, datetime):
                    if user_item["last_seen"] is None or user_item["last_seen"] < event_at:
                        user_item["last_seen"] = event_at
                user_item["channels"].add(channel)

        channel_list: list[dict[str, Any]] = []
        for item in channels.values():
            channel_list.append(
                {
                    "channel": item["channel"],
                    "inbound_messages": item["inbound_messages"],
                    "outbound_messages": item["outbound_messages"],
                    "failed_outbound_messages": item["failed_outbound_messages"],
                    "unique_users": len(item["unique_users"]),
                }
            )
        channel_list.sort(key=lambda item: item["inbound_messages"], reverse=True)

        top_users = sorted(
            user_stats.values(),
            key=lambda item: (int(item["inbound_messages"]), item["last_seen"] or datetime.fromtimestamp(0, UTC)),
            reverse=True,
        )[:top_users_limit]
        top_users_serialized = [
            {
                "user_id": item["user_id"],
                "inbound_messages": item["inbound_messages"],
                "last_seen": item["last_seen"].isoformat() if isinstance(item["last_seen"], datetime) else None,
                "channels": sorted(item["channels"]),
            }
            for item in top_users
        ]

        unique_users = len({str(event["user_id"]) for event in filtered if event.get("user_id")})
        return {
            "source": "memory",
            "since_hours": since_hours,
            "window_start": cutoff.isoformat(),
            "window_end": datetime.now(UTC).isoformat(),
            "totals": {
                "events": len(filtered),
                "inbound_messages": len(inbound),
                "outbound_messages": len(outbound),
                "failed_outbound_messages": len(outbound_failed),
                "unique_users": unique_users,
                "active_threads": len(threads_seen),
            },
            "channels": channel_list,
            "top_users": top_users_serialized,
            "leads": {
                "source": "unavailable",
                "total": 0,
                "qualified": 0,
                "in_review": 0,
                "disqualified": 0,
                "qualification_rate": None,
            },
        }


class InteractionMetricsStore(Protocol):
    async def setup(self) -> None: ...

    async def record_event(
        self,
        *,
        channel: str,
        user_id: str | None,
        thread_id: str,
        direction: str,
        event_type: str,
        success: bool,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    async def summarize(self, *, since_hours: int = 24, top_users_limit: int = 20) -> dict[str, Any]: ...


class PostgresInteractionMetrics:
    def __init__(
        self,
        pool_manager: PostgresPoolManager,
        *,
        table_name: str = "interaction_events",
        crm_pool_manager: PostgresPoolManager | None = None,
        crm_table_name: str = "crm_leads",
    ) -> None:
        self._pool_manager = pool_manager
        self._table_name = table_name
        self._crm_pool_manager = crm_pool_manager or pool_manager
        self._crm_table_name = crm_table_name
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
                        id BIGSERIAL PRIMARY KEY,
                        event_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        channel TEXT NOT NULL,
                        user_id TEXT NULL,
                        thread_id TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        run_id TEXT NULL,
                        success BOOLEAN NOT NULL DEFAULT TRUE,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
                    )
                    """
                )
                await connection.execute(
                    f"CREATE INDEX IF NOT EXISTS ix_{self._table_name}_event_at ON {self._table_name} (event_at DESC)"
                )
                await connection.execute(
                    f"CREATE INDEX IF NOT EXISTS ix_{self._table_name}_channel_event_at ON {self._table_name} (channel, event_at DESC)"
                )
                await connection.execute(
                    f"CREATE INDEX IF NOT EXISTS ix_{self._table_name}_user_event_at ON {self._table_name} (user_id, event_at DESC)"
                )
            finally:
                await self._pool_manager.release(connection)

            self._setup_done = True

    async def record_event(
        self,
        *,
        channel: str,
        user_id: str | None,
        thread_id: str,
        direction: str,
        event_type: str,
        success: bool,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self.setup()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=True)

        connection = await self._pool_manager.acquire()
        try:
            await connection.execute(
                f"""
                INSERT INTO {self._table_name}
                    (channel, user_id, thread_id, direction, event_type, run_id, success, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
                """,
                channel,
                user_id,
                thread_id,
                direction,
                event_type,
                run_id,
                success,
                metadata_json,
            )
        finally:
            await self._pool_manager.release(connection)

    async def summarize(
        self,
        *,
        since_hours: int = 24,
        top_users_limit: int = 20,
    ) -> dict[str, Any]:
        await self.setup()
        cutoff = datetime.now(UTC) - timedelta(hours=since_hours)

        connection = await self._pool_manager.acquire()
        try:
            totals_row = await connection.fetchrow(
                f"""
                SELECT
                    COUNT(*)::BIGINT AS events,
                    COUNT(*) FILTER (WHERE direction = 'inbound')::BIGINT AS inbound_messages,
                    COUNT(*) FILTER (WHERE direction = 'outbound')::BIGINT AS outbound_messages,
                    COUNT(*) FILTER (WHERE direction = 'outbound' AND success = FALSE)::BIGINT AS failed_outbound_messages,
                    (COUNT(DISTINCT user_id) FILTER (WHERE user_id IS NOT NULL AND user_id <> ''))::BIGINT AS unique_users,
                    COUNT(DISTINCT thread_id)::BIGINT AS active_threads
                FROM {self._table_name}
                WHERE event_at >= $1
                """,
                cutoff,
            )
            channel_rows = await connection.fetch(
                f"""
                SELECT
                    channel,
                    COUNT(*) FILTER (WHERE direction = 'inbound')::BIGINT AS inbound_messages,
                    COUNT(*) FILTER (WHERE direction = 'outbound')::BIGINT AS outbound_messages,
                    COUNT(*) FILTER (WHERE direction = 'outbound' AND success = FALSE)::BIGINT AS failed_outbound_messages,
                    (COUNT(DISTINCT user_id) FILTER (WHERE user_id IS NOT NULL AND user_id <> ''))::BIGINT AS unique_users
                FROM {self._table_name}
                WHERE event_at >= $1
                GROUP BY channel
                ORDER BY inbound_messages DESC
                """,
                cutoff,
            )
            top_user_rows = await connection.fetch(
                f"""
                SELECT
                    user_id,
                    COUNT(*) FILTER (WHERE direction = 'inbound')::BIGINT AS inbound_messages,
                    MAX(event_at) AS last_seen,
                    ARRAY_AGG(DISTINCT channel) AS channels
                FROM {self._table_name}
                WHERE event_at >= $1
                  AND user_id IS NOT NULL
                  AND user_id <> ''
                GROUP BY user_id
                ORDER BY inbound_messages DESC, last_seen DESC
                LIMIT $2
                """,
                cutoff,
                top_users_limit,
            )
        finally:
            await self._pool_manager.release(connection)

        leads_summary = await self._fetch_leads_summary(cutoff)

        return {
            "source": "postgres",
            "since_hours": since_hours,
            "window_start": cutoff.isoformat(),
            "window_end": datetime.now(UTC).isoformat(),
            "totals": {
                "events": int(totals_row["events"]) if totals_row else 0,
                "inbound_messages": int(totals_row["inbound_messages"]) if totals_row else 0,
                "outbound_messages": int(totals_row["outbound_messages"]) if totals_row else 0,
                "failed_outbound_messages": int(totals_row["failed_outbound_messages"]) if totals_row else 0,
                "unique_users": int(totals_row["unique_users"]) if totals_row else 0,
                "active_threads": int(totals_row["active_threads"]) if totals_row else 0,
            },
            "channels": [
                {
                    "channel": str(row["channel"]),
                    "inbound_messages": int(row["inbound_messages"]),
                    "outbound_messages": int(row["outbound_messages"]),
                    "failed_outbound_messages": int(row["failed_outbound_messages"]),
                    "unique_users": int(row["unique_users"]),
                }
                for row in channel_rows
            ],
            "top_users": [
                {
                    "user_id": str(row["user_id"]),
                    "inbound_messages": int(row["inbound_messages"]),
                    "last_seen": row["last_seen"].isoformat() if row["last_seen"] else None,
                    "channels": [str(channel) for channel in (row["channels"] or [])],
                }
                for row in top_user_rows
            ],
            "leads": leads_summary,
        }

    async def _fetch_leads_summary(self, cutoff: datetime) -> dict[str, Any]:
        connection = await self._crm_pool_manager.acquire()
        try:
            row = await connection.fetchrow(
                f"""
                SELECT
                    COUNT(*)::BIGINT AS total,
                    COUNT(*) FILTER (WHERE lead_status = 'calificado')::BIGINT AS qualified,
                    COUNT(*) FILTER (WHERE lead_status = 'en_revision')::BIGINT AS in_review,
                    COUNT(*) FILTER (WHERE lead_status = 'no_calificado')::BIGINT AS disqualified
                FROM {self._crm_table_name}
                WHERE created_at >= $1
                """,
                cutoff,
            )
        except Exception:
            return {
                "source": "unavailable",
                "total": 0,
                "qualified": 0,
                "in_review": 0,
                "disqualified": 0,
                "qualification_rate": None,
            }
        finally:
            await self._crm_pool_manager.release(connection)

        if row is None:
            return {
                "source": "postgres",
                "total": 0,
                "qualified": 0,
                "in_review": 0,
                "disqualified": 0,
                "qualification_rate": None,
            }

        total = int(row["total"])
        qualified = int(row["qualified"])
        return {
            "source": "postgres",
            "total": total,
            "qualified": qualified,
            "in_review": int(row["in_review"]),
            "disqualified": int(row["disqualified"]),
            "qualification_rate": (qualified / total) if total > 0 else None,
        }
