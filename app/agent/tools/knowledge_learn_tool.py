from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Optional, Protocol

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, ConfigDict, Field

from app.agent.prompt_loader import load_prompt
from app.agent.tools.knowledge_search_tool import KnowledgeRequestContext

logger = logging.getLogger(__name__)


class KnowledgeLearnDetectorOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_learning_instruction: bool
    confidence: float = Field(ge=0, le=1)
    topic: str | None = None
    normalized_content: str | None = None
    reason: str = Field(min_length=1)


class KnowledgeLearnDetector(Protocol):
    async def detect(self, *, source_text: str, topic_hint: str | None) -> KnowledgeLearnDetectorOutput:
        ...


class BedrockKnowledgeLearnDetector:
    def __init__(self, *, model_name: str, aws_region: str) -> None:
        self._model_name = model_name
        self._aws_region = aws_region
        self._model: Any | None = None
        self._system_prompt = load_prompt("knowledge_learn_detector")

    async def detect(self, *, source_text: str, topic_hint: str | None) -> KnowledgeLearnDetectorOutput:
        if self._model is None:
            from langchain_aws import ChatBedrockConverse

            model = ChatBedrockConverse(
                model_id=self._model_name,
                region_name=self._aws_region,
                temperature=0,
                disable_streaming=True,
            )
            self._model = model.with_structured_output(KnowledgeLearnDetectorOutput)

        user_text = source_text.strip()
        hint_block = f"\nTopic hint: {topic_hint.strip()}" if topic_hint else ""
        output = await self._model.ainvoke(
            [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": f"Input text:\n{user_text}{hint_block}"},
            ]
        )
        if isinstance(output, KnowledgeLearnDetectorOutput):
            return output
        return KnowledgeLearnDetectorOutput.model_validate(output)


@dataclass(slots=True)
class PendingKnowledgeLearn:
    topic: str
    content: str
    confidence: float
    reason: str
    expires_at: datetime


class KnowledgeLearnArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_text: str = Field(min_length=1, max_length=3000)
    confirm: bool = False
    topic_hint: str | None = Field(default=None, min_length=1)


class KnowledgeLearnTool(BaseTool):
    name: str = "knowledge_learn"
    description: str = (
        "Detect and learn new business knowledge from user instructions. "
        "Only use when the user is explicitly teaching business/service information."
    )
    args_schema: Optional[ArgsSchema] = KnowledgeLearnArgs
    return_direct: bool = False
    knowledge_provider: Any
    get_context: Any
    is_admin_requestor: Any
    detector: Any
    confidence_threshold: float = 0.75
    pending_ttl_minutes: int = 30
    pending_by_thread: dict[str, PendingKnowledgeLearn] = Field(default_factory=dict)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        context = self.get_context()
        if context is None:
            return {
                "status": "error",
                "confidence": None,
                "topic": None,
                "knowledge_id": None,
                "message": "No se pudo resolver el contexto para guardar conocimiento.",
                "pending": False,
            }

        if not self.is_admin_requestor(context.requestor):
            logger.info(
                "knowledge_learn_attempt status=unauthorized tenant_id=%s agent_id=%s requestor=%s thread_id=%s",
                context.tenant_id,
                context.agent_id,
                context.requestor,
                context.thread_id,
            )
            return {
                "status": "unauthorized",
                "confidence": None,
                "topic": None,
                "knowledge_id": None,
                "message": "No tienes permisos para ensenar conocimiento al agente.",
                "pending": False,
            }

        source_text = str(payload.get("source_text") or "").strip()
        topic_hint = payload.get("topic_hint")
        topic_hint_text = str(topic_hint).strip() if isinstance(topic_hint, str) else None
        confirm = bool(payload.get("confirm"))
        routed_as_update = context.memory_intent == "update"
        routed_confidence = (
            float(context.memory_intent_confidence)
            if isinstance(context.memory_intent_confidence, (int, float))
            else None
        )
        logger.info(
            "knowledge_learn_attempt tenant_id=%s agent_id=%s requestor=%s thread_id=%s confirm=%s routed_intent=%s",
            context.tenant_id,
            context.agent_id,
            context.requestor,
            context.thread_id,
            confirm,
            context.memory_intent,
        )

        if confirm or self._looks_like_confirmation(source_text):
            confirmed = await self._confirm_pending(context=context)
            if confirmed is not None:
                return confirmed

            # If there is explicit confirmation and router marked update,
            # persist directly even when there is no pending draft.
            if confirm and routed_as_update:
                forced_confidence = routed_confidence if routed_confidence is not None else 0.8
                return await self._persist_learning(
                    context=context,
                    topic_hint_text=topic_hint_text,
                    source_text=source_text,
                    confidence=forced_confidence,
                    reason="memory_intent_router_update_with_confirm",
                    metadata_extra={"forced_by_confirm": True},
                )

        self._drop_expired_pending()
        detection = await self.detector.detect(source_text=source_text, topic_hint=topic_hint_text)

        if not detection.is_learning_instruction:
            if routed_as_update:
                forced_confidence = (
                    max(float(detection.confidence), routed_confidence)
                    if routed_confidence is not None
                    else max(float(detection.confidence), 0.7)
                )
                return await self._persist_learning(
                    context=context,
                    topic_hint_text=topic_hint_text,
                    source_text=source_text,
                    confidence=forced_confidence,
                    reason="memory_intent_router_update_fallback",
                    metadata_extra={
                        "detector_is_learning_instruction": False,
                        "detector_reason": detection.reason,
                    },
                )
            return {
                "status": "ignored_not_instruction",
                "confidence": float(detection.confidence),
                "topic": detection.topic,
                "knowledge_id": None,
                "message": "No detecte una instruccion valida de aprendizaje en este mensaje.",
                "pending": False,
            }

        topic = self._normalize_topic(detection.topic or topic_hint_text or source_text)
        content = self._normalize_content(detection.normalized_content or source_text)
        if not topic or not content:
            return {
                "status": "ignored_not_instruction",
                "confidence": float(detection.confidence),
                "topic": topic or None,
                "knowledge_id": None,
                "message": "No pude extraer un tema y contenido validos para aprender.",
                "pending": False,
            }

        if float(detection.confidence) < self.confidence_threshold:
            if routed_as_update:
                forced_confidence = (
                    max(float(detection.confidence), routed_confidence)
                    if routed_confidence is not None
                    else max(float(detection.confidence), 0.7)
                )
                return await self._persist_learning(
                    context=context,
                    topic_hint_text=topic_hint_text,
                    source_text=source_text,
                    confidence=forced_confidence,
                    reason="memory_intent_router_update_low_confidence_bypass",
                    metadata_extra={"detector_reason": detection.reason},
                )
            pending_ttl = timedelta(minutes=max(1, self.pending_ttl_minutes))
            self.pending_by_thread[context.thread_id] = PendingKnowledgeLearn(
                topic=topic,
                content=content,
                confidence=float(detection.confidence),
                reason=detection.reason,
                expires_at=datetime.now(timezone.utc) + pending_ttl,
            )
            logger.info(
                "knowledge_learn_confirm_required tenant_id=%s agent_id=%s thread_id=%s topic=%s confidence=%.3f",
                context.tenant_id,
                context.agent_id,
                context.thread_id,
                topic,
                float(detection.confidence),
            )
            return {
                "status": "needs_confirmation",
                "confidence": float(detection.confidence),
                "topic": topic,
                "knowledge_id": None,
                "message": (
                    "Detecte una instruccion de aprendizaje pero con baja confianza. "
                    "Confirma si quieres que lo guarde."
                ),
                "pending": True,
            }

        return await self._persist_learning(
            context=context,
            topic_hint_text=topic_hint_text,
            source_text=source_text,
            confidence=float(detection.confidence),
            reason=detection.reason,
            normalized_topic=detection.topic,
            normalized_content=detection.normalized_content,
            metadata_extra={
                "detector_confidence": float(detection.confidence),
                "detector_reason": detection.reason,
            },
        )

    async def _arun(self, source_text: str, confirm: bool = False, topic_hint: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"source_text": source_text, "confirm": bool(confirm)}
        if topic_hint:
            payload["topic_hint"] = topic_hint
        return await self.execute(payload)

    def _run(self, source_text: str, confirm: bool = False, topic_hint: str | None = None) -> dict[str, Any]:
        raise NotImplementedError("KnowledgeLearnTool only supports async execution.")

    async def _confirm_pending(self, *, context: KnowledgeRequestContext) -> dict[str, Any] | None:
        pending = self.pending_by_thread.get(context.thread_id)
        if pending is None:
            return None
        if pending.expires_at <= datetime.now(timezone.utc):
            self.pending_by_thread.pop(context.thread_id, None)
            return None

        write_result = await self.knowledge_provider.upsert_topic(
            tenant_id=context.tenant_id,
            agent_id=context.agent_id,
            topic=pending.topic,
            content=pending.content,
            source="chat",
            author=context.requestor,
            metadata={
                "thread_id": context.thread_id,
                "detector_confidence": pending.confidence,
                "detector_reason": pending.reason,
                "confirmed": True,
            },
        )
        self.pending_by_thread.pop(context.thread_id, None)
        logger.info(
            "knowledge_learn_saved tenant_id=%s agent_id=%s thread_id=%s topic=%s knowledge_id=%s version=%s confirmed=true",
            context.tenant_id,
            context.agent_id,
            context.thread_id,
            write_result.topic,
            write_result.knowledge_id,
            write_result.version,
        )
        return {
            "status": "learned",
            "confidence": pending.confidence,
            "topic": write_result.topic,
            "knowledge_id": write_result.knowledge_id,
            "message": "Conocimiento confirmado y guardado correctamente.",
            "pending": False,
        }

    def _drop_expired_pending(self) -> None:
        now = datetime.now(timezone.utc)
        expired = [thread_id for thread_id, pending in self.pending_by_thread.items() if pending.expires_at <= now]
        for thread_id in expired:
            self.pending_by_thread.pop(thread_id, None)

    async def _persist_learning(
        self,
        *,
        context: KnowledgeRequestContext,
        topic_hint_text: str | None,
        source_text: str,
        confidence: float,
        reason: str,
        normalized_topic: str | None = None,
        normalized_content: str | None = None,
        metadata_extra: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        topic = self._normalize_topic(normalized_topic or topic_hint_text or source_text)
        content = self._normalize_content(normalized_content or source_text)
        if not topic or not content:
            return {
                "status": "ignored_not_instruction",
                "confidence": confidence,
                "topic": topic or None,
                "knowledge_id": None,
                "message": "No pude extraer un tema y contenido validos para aprender.",
                "pending": False,
            }

        metadata: dict[str, object] = {
            "thread_id": context.thread_id,
            "memory_intent": context.memory_intent,
            "memory_intent_confidence": context.memory_intent_confidence,
            "reason": reason,
        }
        if metadata_extra:
            metadata.update(metadata_extra)

        write_result = await self.knowledge_provider.upsert_topic(
            tenant_id=context.tenant_id,
            agent_id=context.agent_id,
            topic=topic,
            content=content,
            source="chat",
            author=context.requestor,
            metadata=metadata,
        )
        self.pending_by_thread.pop(context.thread_id, None)
        logger.info(
            "knowledge_learn_saved tenant_id=%s agent_id=%s thread_id=%s topic=%s knowledge_id=%s version=%s",
            context.tenant_id,
            context.agent_id,
            context.thread_id,
            write_result.topic,
            write_result.knowledge_id,
            write_result.version,
        )
        return {
            "status": "learned",
            "confidence": confidence,
            "topic": write_result.topic,
            "knowledge_id": write_result.knowledge_id,
            "message": "Conocimiento guardado correctamente.",
            "pending": False,
        }

    @staticmethod
    def _looks_like_confirmation(text: str) -> bool:
        normalized = KnowledgeLearnTool._normalize_plain(text)
        affirmatives = {
            "si",
            "si confirmo",
            "confirmo",
            "adelante",
            "ok",
            "ok guardalo",
            "guardalo",
            "guardar",
            "yes",
        }
        return normalized in affirmatives or normalized.startswith("si ")

    @staticmethod
    def _normalize_topic(value: str) -> str:
        text = " ".join(value.strip().split())
        if len(text) > 140:
            text = text[:140].rsplit(" ", 1)[0]
        return text

    @staticmethod
    def _normalize_content(value: str) -> str:
        text = " ".join(value.strip().split())
        if len(text) > 3000:
            text = text[:3000].rsplit(" ", 1)[0]
        return text

    @staticmethod
    def _normalize_plain(value: str) -> str:
        lowered = value.lower().strip()
        normalized = unicodedata.normalize("NFD", lowered)
        no_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
        return " ".join(no_marks.split())
