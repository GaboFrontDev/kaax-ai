from __future__ import annotations

import re
from typing import Any, Literal

from app.agent.orchestration.nodes.capture import (
    after_capture,
    ask_missing_fields,
    capture_lead,
    demo_cta,
    lead_capture_gate,
)
from app.agent.orchestration.nodes.discovery import compose_reply as compose_reply_fallback
from app.agent.orchestration.nodes.discovery import discovery_value, repeat_handler
from app.agent.orchestration.nodes.memory import memory_lookup
from app.agent.orchestration.nodes.reply import persist_state, send_reply
from app.agent.orchestration.nodes.routing import classify_route, classify_turn
from app.agent.orchestration.schemas import OrchestrationState
from app.agent.orchestration.subagents import SubagentRunner, invoke_sales_dialogue
from app.agent.tools.capture_lead_if_ready_tool import CaptureLeadIfReadyTool
from app.agent.tools.memory_intent_router_tool import MemoryIntentRouterTool
from app.memory.session_manager import SessionManager


class OrchestrationNodes:
    def __init__(
        self,
        *,
        session_manager: SessionManager,
        memory_router: MemoryIntentRouterTool,
        capture_tool: CaptureLeadIfReadyTool | None,
        notify_owner: bool,
        subagent_runner: SubagentRunner | None,
    ) -> None:
        self._session_manager = session_manager
        self._memory_router = memory_router
        self._capture_tool = capture_tool
        self._notify_owner = bool(notify_owner)
        self._subagent_runner = subagent_runner

    async def classify_turn(self, state: OrchestrationState) -> dict[str, Any]:
        return await classify_turn(state)

    @staticmethod
    def classify_route(state: OrchestrationState) -> Literal["repeat_handler", "memory_lookup", "discovery_value"]:
        return classify_route(state)

    async def repeat_handler(self, state: OrchestrationState) -> dict[str, Any]:
        return await repeat_handler(state)

    async def memory_lookup(self, state: OrchestrationState) -> dict[str, Any]:
        return await memory_lookup(state, memory_router=self._memory_router)

    async def discovery_value(self, state: OrchestrationState) -> dict[str, Any]:
        return await discovery_value(state)

    async def compose_reply(self, state: OrchestrationState) -> dict[str, Any]:
        fallback = await compose_reply_fallback(state)
        draft = str(fallback.get("draft_response") or "").strip()
        if not draft:
            return fallback

        if self._subagent_runner is None:
            return fallback

        stage = state.conversation_state.stage
        if stage == "lead_capture":
            return fallback

        last_assistant = self._last_assistant_message(state.messages)
        llm_response = await invoke_sales_dialogue(
            runner=self._subagent_runner,
            user_message=state.last_user_message,
            context={
                "stage": stage,
                "router": state.router.model_dump() if state.router is not None else None,
                "conversation_state": state.conversation_state.model_dump(),
                "draft_response": draft,
                "last_assistant_message": last_assistant,
                "response_goal": self._response_goal(state),
                "task": (
                    "Reescribe el borrador para que suene consultivo, natural y orientado a demo, "
                    "sin inventar datos, evitando repetición textual y manteniendo una sola pregunta."
                ),
            },
        )
        if not llm_response:
            return fallback

        refined = self._enforce_response_shape(
            llm_response,
            stage=stage,
            last_assistant_message=last_assistant,
        )
        if not refined:
            return fallback
        return {"draft_response": refined}

    @staticmethod
    def lead_capture_gate(state: OrchestrationState) -> Literal["capture_lead", "send_reply"]:
        return lead_capture_gate(state)

    async def capture_lead(self, state: OrchestrationState) -> dict[str, Any]:
        return await capture_lead(
            state,
            capture_tool=self._capture_tool,
            notify_owner=self._notify_owner,
        )

    @staticmethod
    def after_capture(state: OrchestrationState) -> Literal["ask_missing_fields", "demo_cta", "send_reply"]:
        return after_capture(state)

    async def ask_missing_fields(self, state: OrchestrationState) -> dict[str, Any]:
        return await ask_missing_fields(state)

    async def demo_cta(self, state: OrchestrationState) -> dict[str, Any]:
        return await demo_cta(state)

    async def send_reply(self, state: OrchestrationState) -> dict[str, Any]:
        return await send_reply(state)

    async def persist_state(self, state: OrchestrationState) -> dict[str, Any]:
        return await persist_state(state, session_manager=self._session_manager)

    @staticmethod
    def _enforce_response_shape(
        raw: str,
        *,
        stage: str,
        last_assistant_message: str,
    ) -> str:
        text = " ".join(str(raw or "").split())
        if not text:
            return ""
        parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
        if len(parts) > 3:
            parts = parts[:3]
        normalized = " ".join(parts)

        if stage != "greeting":
            normalized = OrchestrationNodes._strip_leading_greeting(normalized)

        question_count = normalized.count("?")
        if question_count > 1:
            # Keep only the first question to preserve one-question policy.
            first_q = normalized.find("?")
            if first_q > 0:
                prefix = normalized[: first_q + 1]
                suffix = normalized[first_q + 1 :].replace("?", ".")
                normalized = f"{prefix} {suffix}".strip()

        if not normalized:
            return normalized
        if last_assistant_message and OrchestrationNodes._similarity(normalized, last_assistant_message) >= 0.9:
            normalized = f"{normalized} ¿Te parece si lo vemos con un ejemplo real de tu flujo?"
            normalized = OrchestrationNodes._enforce_single_question(normalized)
        return normalized

    @staticmethod
    def _enforce_single_question(text: str) -> str:
        question_count = text.count("?")
        if question_count <= 1:
            return text
        first_q = text.find("?")
        if first_q <= 0:
            return text
        prefix = text[: first_q + 1]
        suffix = text[first_q + 1 :].replace("?", ".")
        return f"{prefix} {suffix}".strip()

    @staticmethod
    def _strip_leading_greeting(text: str) -> str:
        patterns = (
            r"^hola[,!.\s]+",
            r"^buenas[,!.\s]+",
            r"^que tal[,!.\s]+",
            r"^hello[,!.\s]+",
            r"^hi[,!.\s]+",
        )
        result = text.strip()
        for pattern in patterns:
            updated = re.sub(pattern, "", result, flags=re.IGNORECASE).strip()
            if updated != result:
                result = updated
        return result

    @staticmethod
    def _last_assistant_message(messages: list[dict[str, str]]) -> str:
        for message in reversed(messages):
            if str(message.get("role") or "") == "assistant":
                content = str(message.get("content") or "").strip()
                if content:
                    return content
        return ""

    @staticmethod
    def _response_goal(state: OrchestrationState) -> str:
        stage = state.conversation_state.stage
        user_text = (state.last_user_message or "").lower()
        if any(token in user_text for token in ("servicio", "servicios", "funcionalidades")):
            return "Responder con servicios concretos y después avanzar con una sola pregunta."
        if stage in {"greeting", "discovery"}:
            return "Descubrir contexto de negocio con una sola pregunta clara."
        if stage in {"opportunity", "value_mapping"}:
            return "Mapear valor de Kaax AI a dolor detectado y acercar demo."
        if stage == "demo_cta":
            return "Cerrar con propuesta de demo concreta."
        return "Responder breve, útil y orientado al siguiente paso."

    @staticmethod
    def _similarity(left: str, right: str) -> float:
        left_tokens = set(str(left or "").lower().split())
        right_tokens = set(str(right or "").lower().split())
        union = left_tokens.union(right_tokens)
        if not union:
            return 0.0
        return float(len(left_tokens.intersection(right_tokens))) / float(len(union))
