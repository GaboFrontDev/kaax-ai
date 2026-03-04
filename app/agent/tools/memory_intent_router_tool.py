from __future__ import annotations

import re
from typing import Any, Optional

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, ConfigDict, Field


class MemoryIntentRouterArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_message: str = Field(min_length=1)


class MemoryIntentRouterTool(BaseTool):
    name: str = "memory_intent_router"
    description: str = (
        "Semantic router for conversation memory. "
        "Returns mode=read or mode=update using the raw user_message, never abstract keys."
    )
    args_schema: Optional[ArgsSchema] = MemoryIntentRouterArgs
    return_direct: bool = False
    knowledge_provider: Any
    get_context: Any
    default_limit: int = 3
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        user_message = str(payload.get("user_message") or "").strip()
        if not user_message:
            return {"mode": "read", "payload": {"matches": []}}

        fallback_context = self._fallback_context(payload)
        mode = "update" if self._is_update_instruction(user_message) else "read"
        if mode == "update":
            return await self._update(user_message, fallback_context=fallback_context)
        return await self._read(user_message, fallback_context=fallback_context)

    async def _arun(self, user_message: str) -> dict[str, Any]:
        return await self.execute({"user_message": user_message})

    def _run(self, user_message: str) -> dict[str, Any]:
        raise NotImplementedError("MemoryIntentRouterTool only supports async execution.")

    async def _read(self, user_message: str, *, fallback_context: dict[str, str]) -> dict[str, Any]:
        context = self.get_context()
        tenant_id = (
            context.tenant_id
            if context is not None
            else fallback_context.get("tenant_id", "anonymous")
        )
        agent_id = (
            context.agent_id
            if context is not None
            else fallback_context.get("agent_id", "default")
        )

        matches = await self.knowledge_provider.search(
            tenant_id=tenant_id,
            agent_id=agent_id,
            query=user_message,
            limit=max(1, int(self.default_limit)),
        )
        payload = {
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
        return {"mode": "read", "payload": payload}

    async def _update(self, user_message: str, *, fallback_context: dict[str, str]) -> dict[str, Any]:
        context = self.get_context()
        tenant_id = (
            context.tenant_id
            if context is not None
            else fallback_context.get("tenant_id", "anonymous")
        )
        agent_id = (
            context.agent_id
            if context is not None
            else fallback_context.get("agent_id", "default")
        )
        author = (
            context.requestor
            if context is not None
            else fallback_context.get("requestor")
        )
        thread_id = (
            context.thread_id
            if context is not None
            else fallback_context.get("thread_id", "unknown")
        )

        topic = self._derive_topic(user_message)
        write = await self.knowledge_provider.upsert_topic(
            tenant_id=tenant_id,
            agent_id=agent_id,
            topic=topic,
            content=user_message,
            source="chat",
            author=author,
            metadata={
                "thread_id": thread_id,
                "router": "memory_intent_router",
            },
        )
        return {
            "mode": "update",
            "payload": {
                "status": write.status,
                "topic": write.topic,
                "knowledge_id": write.knowledge_id,
                "version": write.version,
            },
        }

    @staticmethod
    def _is_update_instruction(user_message: str) -> bool:
        normalized = MemoryIntentRouterTool._normalize_text(user_message)
        update_signals = (
            "aprende",
            "recuerda",
            "guarda",
            "actualiza",
            "corrige",
            "a partir de ahora",
            "ten en cuenta",
            "anota",
        )
        return any(token in normalized for token in update_signals)

    @staticmethod
    def _derive_topic(user_message: str) -> str:
        normalized = MemoryIntentRouterTool._normalize_text(user_message)
        factual_hints = (
            "precio",
            "precios",
            "integraciones",
            "soporte",
            "implementacion",
            "politicas",
        )
        for hint in factual_hints:
            if hint in normalized:
                return hint

        tokens = [token for token in re.findall(r"[a-z0-9]+", normalized) if len(token) > 2]
        if not tokens:
            return "general"
        return " ".join(tokens[:6])

    @staticmethod
    def _normalize_text(text: str) -> str:
        lowered = str(text or "").strip().lower()
        return re.sub(r"\s+", " ", lowered)

    @staticmethod
    def _fallback_context(payload: dict[str, Any]) -> dict[str, str]:
        requestor = str(payload.get("requestor") or "").strip()
        tenant_id = str(payload.get("tenant_id") or requestor or "anonymous").strip()
        agent_id = str(payload.get("agent_id") or "default").strip() or "default"
        thread_id = str(payload.get("thread_id") or "").strip() or "unknown"
        return {
            "requestor": requestor or "anonymous",
            "tenant_id": tenant_id or "anonymous",
            "agent_id": agent_id,
            "thread_id": thread_id,
        }
