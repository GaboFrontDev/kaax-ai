from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Optional

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class KnowledgeRequestContext:
    thread_id: str
    requestor: str
    tenant_id: str
    agent_id: str
    memory_intent: str | None = None
    memory_intent_confidence: float | None = None


class KnowledgeSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)


class KnowledgeSearchTool(BaseTool):
    name: str = "knowledge_search"
    description: str = (
        "Search business knowledge learned for the current tenant and agent. "
        "Use this for FAQ/product/service questions before drafting final answers."
    )
    args_schema: Optional[ArgsSchema] = KnowledgeSearchArgs
    return_direct: bool = False
    knowledge_provider: Any
    get_context: Any
    default_limit: int = 5
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = str(payload.get("query") or "").strip()
        limit = int(payload.get("limit") or max(1, self.default_limit))

        context = self.get_context()
        if context is None:
            return {"error": "knowledge_search_context_unavailable"}

        if not query:
            return {"matches": []}

        matches = await self.knowledge_provider.search(
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

    async def _arun(self, query: str, limit: int = 5) -> dict[str, Any]:
        return await self.execute({"query": query, "limit": limit})

    def _run(self, query: str, limit: int = 5) -> dict[str, Any]:
        raise NotImplementedError("KnowledgeSearchTool only supports async execution.")
