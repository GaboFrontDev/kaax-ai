from __future__ import annotations

import asyncio

from app.knowledge.providers import InMemoryKnowledgeProvider


def test_in_memory_knowledge_upsert_versions_active_topic() -> None:
    provider = InMemoryKnowledgeProvider()

    async def _run() -> tuple[int, int]:
        first = await provider.upsert_topic(
            tenant_id="tenant-a",
            agent_id="default",
            topic="pricing",
            content="Plan base 99 USD.",
            source="chat",
            author="admin:test",
            metadata={},
        )
        second = await provider.upsert_topic(
            tenant_id="tenant-a",
            agent_id="default",
            topic="pricing",
            content="Plan base 109 USD.",
            source="chat",
            author="admin:test",
            metadata={},
        )
        active = await provider.get_active_topic(tenant_id="tenant-a", agent_id="default", topic="pricing")
        assert active is not None
        assert "109" in active.content
        return first.version, second.version

    first_version, second_version = asyncio.run(_run())
    assert first_version == 1
    assert second_version == 2


def test_in_memory_knowledge_search_isolated_by_tenant_and_agent() -> None:
    provider = InMemoryKnowledgeProvider()

    async def _run() -> tuple[int, int]:
        await provider.upsert_topic(
            tenant_id="tenant-a",
            agent_id="default",
            topic="soporte",
            content="Soporte en horario laboral.",
            source="chat",
            author="admin:test",
            metadata={},
        )
        await provider.upsert_topic(
            tenant_id="tenant-b",
            agent_id="default",
            topic="soporte",
            content="Soporte 24/7.",
            source="chat",
            author="admin:test",
            metadata={},
        )
        await provider.upsert_topic(
            tenant_id="tenant-a",
            agent_id="sales",
            topic="soporte",
            content="Soporte para agente sales.",
            source="chat",
            author="admin:test",
            metadata={},
        )
        tenant_a_default = await provider.search(
            tenant_id="tenant-a",
            agent_id="default",
            query="laboral",
            limit=5,
        )
        tenant_b_default = await provider.search(
            tenant_id="tenant-b",
            agent_id="default",
            query="24/7",
            limit=5,
        )
        return len(tenant_a_default), len(tenant_b_default)

    tenant_a_count, tenant_b_count = asyncio.run(_run())
    assert tenant_a_count == 1
    assert tenant_b_count == 1
