from __future__ import annotations

from contextlib import asynccontextmanager
import contextvars
from fnmatch import fnmatchcase

from app.agent.tools.knowledge_search_tool import KnowledgeRequestContext


class ToolRequestContextManager:
    def __init__(
        self,
        *,
        agent_id: str = "default",
        knowledge_admin_requestors: set[str] | None = None,
    ) -> None:
        self._agent_id = agent_id.strip() or "default"
        self._knowledge_admin_requestors = set(knowledge_admin_requestors or set())
        self._context: contextvars.ContextVar[KnowledgeRequestContext | None] = contextvars.ContextVar(
            "tool_request_context",
            default=None,
        )

    @asynccontextmanager
    async def request_context(
        self,
        *,
        thread_id: str,
        requestor: str,
        agent_id: str | None = None,
        memory_intent: str | None = None,
        memory_intent_confidence: float | None = None,
    ):
        resolved_agent_id = (agent_id or self._agent_id).strip() or self._agent_id
        resolved_confidence = (
            float(memory_intent_confidence)
            if isinstance(memory_intent_confidence, (int, float))
            else None
        )
        context = KnowledgeRequestContext(
            thread_id=thread_id,
            requestor=requestor,
            tenant_id=self._derive_tenant_id(requestor),
            agent_id=resolved_agent_id,
            memory_intent=memory_intent,
            memory_intent_confidence=resolved_confidence,
        )
        token = self._context.set(context)
        try:
            yield context
        finally:
            self._context.reset(token)

    def get_context(self) -> KnowledgeRequestContext | None:
        return self._context.get()

    def is_admin_requestor(self, requestor: str) -> bool:
        if not self._knowledge_admin_requestors:
            return False
        normalized = (requestor or "").strip()
        for allowed in self._knowledge_admin_requestors:
            if fnmatchcase(normalized, allowed):
                return True
        return False

    @staticmethod
    def _derive_tenant_id(requestor: str) -> str:
        value = (requestor or "").strip()
        return value or "anonymous"
