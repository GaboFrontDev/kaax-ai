from __future__ import annotations

import asyncio

from app.agent.tools.knowledge_learn_tool import (
    KnowledgeLearnDetectorOutput,
    KnowledgeLearnTool,
)
from app.agent.tools.knowledge_search_tool import KnowledgeRequestContext
from app.knowledge.providers import KnowledgeWriteResult


class _StubKnowledgeProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def upsert_topic(
        self,
        *,
        tenant_id: str,
        agent_id: str,
        topic: str,
        content: str,
        source: str,
        author: str | None,
        metadata: dict[str, object],
    ) -> KnowledgeWriteResult:
        self.calls.append(
            {
                "tenant_id": tenant_id,
                "agent_id": agent_id,
                "topic": topic,
                "content": content,
                "source": source,
                "author": author,
                "metadata": metadata,
            }
        )
        return KnowledgeWriteResult(
            knowledge_id="k-1",
            topic=topic,
            version=1,
            status="upserted",
        )


class _AlwaysNotLearningDetector:
    async def detect(self, *, source_text: str, topic_hint: str | None) -> KnowledgeLearnDetectorOutput:
        return KnowledgeLearnDetectorOutput(
            is_learning_instruction=False,
            confidence=0.2,
            topic=None,
            normalized_content=None,
            reason="stub_not_learning",
        )


def test_knowledge_learn_uses_memory_intent_update_fallback() -> None:
    provider = _StubKnowledgeProvider()
    context = KnowledgeRequestContext(
        thread_id="t-1",
        requestor="admin:local",
        tenant_id="tenant-1",
        agent_id="default",
        memory_intent="update",
        memory_intent_confidence=0.9,
    )

    tool = KnowledgeLearnTool(
        knowledge_provider=provider,
        get_context=lambda: context,
        is_admin_requestor=lambda _: True,
        detector=_AlwaysNotLearningDetector(),
        confidence_threshold=0.75,
    )

    result = asyncio.run(
        tool.execute(
            {
                "source_text": "Recuerda que soporte atiende de lunes a viernes de 9 a 18.",
                "confirm": False,
            }
        )
    )

    assert result["status"] == "learned"
    assert result["knowledge_id"] == "k-1"
    assert len(provider.calls) == 1
    metadata = provider.calls[0]["metadata"]
    assert isinstance(metadata, dict)
    assert metadata.get("memory_intent") == "update"
