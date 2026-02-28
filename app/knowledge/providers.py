from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4

from app.infra.db import PostgresPoolManager


@dataclass(frozen=True, slots=True)
class KnowledgeMatch:
    knowledge_id: str
    topic: str
    content: str
    score: float
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class KnowledgeWriteResult:
    knowledge_id: str
    topic: str
    version: int
    status: str


class KnowledgeProvider(Protocol):
    async def search(
        self,
        *,
        tenant_id: str,
        agent_id: str,
        query: str,
        limit: int,
    ) -> list[KnowledgeMatch]:
        ...

    async def upsert_topic(
        self,
        *,
        tenant_id: str,
        agent_id: str,
        topic: str,
        content: str,
        source: str,
        author: str | None,
        metadata: dict[str, Any],
    ) -> KnowledgeWriteResult:
        ...

    async def get_active_topic(
        self,
        *,
        tenant_id: str,
        agent_id: str,
        topic: str,
    ) -> KnowledgeMatch | None:
        ...


class InMemoryKnowledgeProvider:
    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    async def search(
        self,
        *,
        tenant_id: str,
        agent_id: str,
        query: str,
        limit: int,
    ) -> list[KnowledgeMatch]:
        query_lower = query.lower().strip()
        if not query_lower:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in self._rows:
            if row["tenant_id"] != tenant_id or row["agent_id"] != agent_id or not row["is_active"]:
                continue
            topic = str(row["topic"]).lower()
            content = str(row["content"]).lower()

            score = 0.0
            if query_lower in topic:
                score += 1.0
            if query_lower in content:
                score += 0.8

            shared_tokens = {
                token
                for token in query_lower.split()
                if token and (token in topic or token in content)
            }
            score += float(len(shared_tokens)) * 0.15
            if score <= 0:
                continue
            scored.append((score, row))

        scored.sort(key=lambda item: (item[0], item[1]["updated_at"]), reverse=True)
        results: list[KnowledgeMatch] = []
        for score, row in scored[: max(1, limit)]:
            results.append(
                KnowledgeMatch(
                    knowledge_id=str(row["knowledge_id"]),
                    topic=str(row["topic"]),
                    content=str(row["content"]),
                    score=float(min(score, 1.0)),
                    updated_at=row["updated_at"],
                )
            )
        return results

    async def upsert_topic(
        self,
        *,
        tenant_id: str,
        agent_id: str,
        topic: str,
        content: str,
        source: str,
        author: str | None,
        metadata: dict[str, Any],
    ) -> KnowledgeWriteResult:
        now = datetime.now(timezone.utc)
        active = None
        for row in self._rows:
            if (
                row["tenant_id"] == tenant_id
                and row["agent_id"] == agent_id
                and row["topic"] == topic
                and row["is_active"]
            ):
                active = row
                break

        if active is not None:
            active["is_active"] = False
            version = int(active["version"]) + 1
        else:
            version = 1

        knowledge_id = str(uuid4())
        row = {
            "knowledge_id": knowledge_id,
            "tenant_id": tenant_id,
            "agent_id": agent_id,
            "topic": topic,
            "content": content,
            "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "version": version,
            "is_active": True,
            "source": source,
            "author_requestor": author,
            "metadata": dict(metadata),
            "created_at": now,
            "updated_at": now,
        }
        self._rows.append(row)
        return KnowledgeWriteResult(
            knowledge_id=knowledge_id,
            topic=topic,
            version=version,
            status="upserted",
        )

    async def get_active_topic(
        self,
        *,
        tenant_id: str,
        agent_id: str,
        topic: str,
    ) -> KnowledgeMatch | None:
        for row in reversed(self._rows):
            if (
                row["tenant_id"] == tenant_id
                and row["agent_id"] == agent_id
                and row["topic"] == topic
                and row["is_active"]
            ):
                return KnowledgeMatch(
                    knowledge_id=str(row["knowledge_id"]),
                    topic=str(row["topic"]),
                    content=str(row["content"]),
                    score=1.0,
                    updated_at=row["updated_at"],
                )
        return None


class PostgresKnowledgeProvider:
    def __init__(self, pool_manager: PostgresPoolManager, *, table_name: str = "agent_knowledge") -> None:
        self._pool_manager = pool_manager
        self._table_name = table_name

    async def search(
        self,
        *,
        tenant_id: str,
        agent_id: str,
        query: str,
        limit: int,
    ) -> list[KnowledgeMatch]:
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        connection = await self._pool_manager.acquire()
        try:
            rows = await connection.fetch(
                f"""
                WITH ranked AS (
                    SELECT
                        knowledge_id,
                        topic,
                        content,
                        updated_at,
                        ts_rank(
                            to_tsvector('simple', topic || ' ' || content),
                            plainto_tsquery('simple', $3)
                        ) AS score
                    FROM {self._table_name}
                    WHERE tenant_id = $1
                      AND agent_id = $2
                      AND is_active = TRUE
                )
                SELECT knowledge_id, topic, content, updated_at, score
                FROM ranked
                WHERE score > 0
                ORDER BY score DESC, updated_at DESC
                LIMIT $4
                """,
                tenant_id,
                agent_id,
                cleaned_query,
                max(1, limit),
            )
            results: list[KnowledgeMatch] = []
            for row in rows:
                results.append(
                    KnowledgeMatch(
                        knowledge_id=str(row["knowledge_id"]),
                        topic=str(row["topic"]),
                        content=str(row["content"]),
                        score=float(row["score"] or 0.0),
                        updated_at=row["updated_at"],
                    )
                )
            return results
        finally:
            await self._pool_manager.release(connection)

    async def upsert_topic(
        self,
        *,
        tenant_id: str,
        agent_id: str,
        topic: str,
        content: str,
        source: str,
        author: str | None,
        metadata: dict[str, Any],
    ) -> KnowledgeWriteResult:
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        metadata_json = json.dumps(metadata, ensure_ascii=True)
        knowledge_id = str(uuid4())

        connection = await self._pool_manager.acquire()
        try:
            async with connection.transaction():
                active = await connection.fetchrow(
                    f"""
                    SELECT knowledge_id, version
                    FROM {self._table_name}
                    WHERE tenant_id = $1
                      AND agent_id = $2
                      AND topic = $3
                      AND is_active = TRUE
                    FOR UPDATE
                    """,
                    tenant_id,
                    agent_id,
                    topic,
                )
                version = int(active["version"]) + 1 if active is not None else 1

                if active is not None:
                    await connection.execute(
                        f"""
                        UPDATE {self._table_name}
                        SET is_active = FALSE, updated_at = NOW()
                        WHERE knowledge_id = $1
                        """,
                        str(active["knowledge_id"]),
                    )

                await connection.execute(
                    f"""
                    INSERT INTO {self._table_name}
                        (knowledge_id, tenant_id, agent_id, topic, content, content_hash, version, is_active, source,
                         author_requestor, metadata, created_at, updated_at)
                    VALUES
                        ($1, $2, $3, $4, $5, $6, $7, TRUE, $8, $9, $10::jsonb, NOW(), NOW())
                    """,
                    knowledge_id,
                    tenant_id,
                    agent_id,
                    topic,
                    content,
                    content_hash,
                    version,
                    source,
                    author,
                    metadata_json,
                )
            return KnowledgeWriteResult(
                knowledge_id=knowledge_id,
                topic=topic,
                version=version,
                status="upserted",
            )
        finally:
            await self._pool_manager.release(connection)

    async def get_active_topic(
        self,
        *,
        tenant_id: str,
        agent_id: str,
        topic: str,
    ) -> KnowledgeMatch | None:
        connection = await self._pool_manager.acquire()
        try:
            row = await connection.fetchrow(
                f"""
                SELECT knowledge_id, topic, content, updated_at
                FROM {self._table_name}
                WHERE tenant_id = $1
                  AND agent_id = $2
                  AND topic = $3
                  AND is_active = TRUE
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                tenant_id,
                agent_id,
                topic,
            )
            if row is None:
                return None
            return KnowledgeMatch(
                knowledge_id=str(row["knowledge_id"]),
                topic=str(row["topic"]),
                content=str(row["content"]),
                score=1.0,
                updated_at=row["updated_at"],
            )
        finally:
            await self._pool_manager.release(connection)
