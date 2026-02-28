from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable

from app.knowledge.providers import KnowledgeProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class KnowledgeRequestContext:
    thread_id: str
    requestor: str
    tenant_id: str
    agent_id: str


class KnowledgeSearchTool:
    name = "knowledge_search"

    def __init__(
        self,
        *,
        knowledge_provider: KnowledgeProvider,
        get_context: Callable[[], KnowledgeRequestContext | None],
        default_limit: int = 5,
    ) -> None:
        self._knowledge_provider = knowledge_provider
        self._get_context = get_context
        self._default_limit = max(1, default_limit)

    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        context = self._get_context()
        if context is None:
            return {"error": "knowledge_search_context_unavailable"}

        query = str(payload.get("query") or "").strip()
        if not query:
            return {"matches": []}

        limit = int(payload.get("limit") or self._default_limit)
        matches = await self._knowledge_provider.search(
            tenant_id=context.tenant_id,
            agent_id=context.agent_id,
            query=query,
            limit=limit,
        )
        logger.info(
            "knowledge_search_%s tenant_id=%s agent_id=%s thread_id=%s query=%s matches=%s",
            "hit" if matches else "miss",
            context.tenant_id,
            context.agent_id,
            context.thread_id,
            query[:120],
            len(matches),
        )
        return {
            "matches": [
                {
                    "topic": match.topic,
                    "content": match.content,
                    "score": float(match.score),
                    "updated_at": match.updated_at.isoformat(),
                }
                for match in matches
            ]
        }
