from __future__ import annotations

import re
from typing import Any, Literal

from app.agent.orchestration.capture_utils import normalize_tool_result
from app.agent.orchestration.helpers import (
    append_tool,
    build_crm_external_key,
    build_missing_fields_question,
    compact_text,
    lookup_previous_summary,
    normalize_missing_fields,
    normalize_text,
)
from app.agent.orchestration.routing_rules import derive_router_and_state
from app.agent.orchestration.schemas import OrchestrationState, QAItem
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
    ) -> None:
        self._session_manager = session_manager
        self._memory_router = memory_router
        self._capture_tool = capture_tool
        self._notify_owner = bool(notify_owner)

    async def classify_turn(self, state: OrchestrationState) -> dict[str, Any]:
        router, updated_conversation_state = derive_router_and_state(
            user_message=state.last_user_message,
            conversation_state=state.conversation_state,
        )
        return {
            "router": router,
            "conversation_state": updated_conversation_state,
            "tool_result": None,
        }

    @staticmethod
    def classify_route(state: OrchestrationState) -> Literal["repeat_handler", "memory_lookup", "discovery_value"]:
        if state.router is None:
            return "discovery_value"
        return state.router.route

    async def repeat_handler(self, state: OrchestrationState) -> dict[str, Any]:
        summary = lookup_previous_summary(
            user_message=state.last_user_message,
            conversation_state=state.conversation_state.model_dump(),
        )
        if summary:
            draft = (
                f"Te resumo lo anterior: {summary}. "
                "¿Quieres más detalle o prefieres que lo aterricemos a una demo?"
            )
        else:
            draft = (
                "Ya tocamos este punto. "
                "¿Quieres más detalle o prefieres verlo aplicado a tu negocio en una demo?"
            )
        return {"draft_response": draft}

    async def memory_lookup(self, state: OrchestrationState) -> dict[str, Any]:
        result = await self._memory_router.execute(
            {
                "user_message": state.last_user_message,
                "requestor": state.requestor,
                "tenant_id": state.requestor,
                "agent_id": "default",
                "thread_id": state.thread_id,
            }
        )
        mode = str(result.get("mode") or "read").strip().lower()
        payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}

        conversation_state = state.conversation_state.model_copy(deep=True)
        if mode in {"read", "update"}:
            conversation_state.tooling.last_memory_route_mode = mode

        normalized_question = normalize_text(state.last_user_message)

        if mode == "update":
            if payload.get("status") == "error":
                draft = (
                    "No pude actualizar memoria en este momento. "
                    "¿Quieres que continuemos con descubrimiento para preparar una demo?"
                )
            else:
                draft = (
                    "Listo, actualicé ese conocimiento en memoria. "
                    "¿Quieres que ahora lo aterrice a tu operación?"
                )
        else:
            matches = payload.get("matches") if isinstance(payload.get("matches"), list) else []
            if matches:
                first = matches[0] if isinstance(matches[0], dict) else {}
                summary = compact_text(str(first.get("content") or ""), max_len=220)
                if summary:
                    conversation_state.qa_memory.factual_cache[normalized_question] = summary
                    draft = (
                        f"Según la información confirmada: {summary}. "
                        "¿Quieres más detalle o lo aplicamos a tu caso?"
                    )
                else:
                    draft = (
                        "Tengo coincidencias en memoria, pero sin detalle útil en este momento. "
                        "¿Quieres que te lo explique en una demo breve?"
                    )
            else:
                draft = (
                    "No tengo ese dato confirmado en memoria por ahora. "
                    "¿Quieres que lo revisemos en una demo enfocada a tu caso?"
                )

        return {
            "conversation_state": conversation_state,
            "draft_response": draft,
            "tools_used": append_tool(state.tools_used, "memory_intent_router"),
        }

    async def discovery_value(self, state: OrchestrationState) -> dict[str, Any]:
        conversation_state = state.conversation_state.model_copy(deep=True)
        stage = conversation_state.stage

        if stage == "greeting":
            draft = (
                "Hola, soy Kaax AI. "
                "Ayudo a convertir conversaciones en oportunidades y demos. "
                "¿Qué parte de tu proceso comercial quieres mejorar primero?"
            )
            conversation_state.stage = "discovery"
            return {"draft_response": draft, "conversation_state": conversation_state}

        use_case = (conversation_state.business_context.use_case or "").strip()
        pain_points = [point for point in conversation_state.business_context.pain_points if point.strip()]

        if not use_case:
            draft = (
                "Para ubicar encaje rápido: "
                "¿qué tipo de conversaciones quieres automatizar hoy (captación, seguimiento o soporte)?"
            )
            conversation_state.stage = "discovery"
            return {"draft_response": draft, "conversation_state": conversation_state}

        if not pain_points:
            draft = (
                f"Entiendo que te interesa {use_case}. "
                "¿Dónde pierden más oportunidades hoy: respuesta, calificación o seguimiento?"
            )
            conversation_state.stage = "discovery"
            return {"draft_response": draft, "conversation_state": conversation_state}

        main_pain = pain_points[0]
        if stage in {"discovery", "opportunity"}:
            draft = (
                f"Veo una oportunidad clara en {main_pain}. "
                "Kaax AI puede responder al instante, calificar leads y pasar solo los listos al equipo. "
                "¿Quieres que te muestre el flujo en una demo?"
            )
            conversation_state.stage = "value_mapping"
            return {"draft_response": draft, "conversation_state": conversation_state}

        draft = (
            "Podemos conectarlo a tu CRM actual para que ventas reciba leads priorizados y con contexto. "
            "¿Te propongo una demo de 20 minutos para verlo con tu caso?"
        )
        conversation_state.stage = "value_mapping"
        return {"draft_response": draft, "conversation_state": conversation_state}

    async def compose_reply(self, state: OrchestrationState) -> dict[str, Any]:
        draft = (state.draft_response or "").strip()
        if not draft:
            draft = "Perfecto, cuéntame un poco más de tu operación y avanzamos paso a paso."

        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", draft) if part.strip()]
        if len(sentences) > 3:
            draft = " ".join(sentences[:3])

        return {"draft_response": draft}

    @staticmethod
    def lead_capture_gate(state: OrchestrationState) -> Literal["capture_lead", "send_reply"]:
        router = state.router
        if router is not None and router.flags.lead_capture_ready:
            return "capture_lead"
        return "send_reply"

    async def capture_lead(self, state: OrchestrationState) -> dict[str, Any]:
        if self._capture_tool is None:
            return {
                "tool_result": {
                    "status": "error",
                    "error": "capture_tool_unavailable",
                    "missing_fields": [],
                },
                "draft_response": (
                    "Puedo continuar con el registro, pero el conector CRM no está disponible ahora. "
                    "¿Quieres que avancemos con una demo y te contacto luego?"
                ),
                "tools_used": append_tool(state.tools_used, "capture_lead_if_ready"),
            }

        lead_data = {
            "contact_name": state.conversation_state.lead_data.contact_name,
            "phone": state.conversation_state.lead_data.phone,
            "contact_schedule": state.conversation_state.lead_data.contact_schedule,
            "intent": state.conversation_state.sales.intent,
            "qualification": state.conversation_state.sales.qualification,
        }

        payload: dict[str, Any] = {
            "lead_data": lead_data,
            "business_context": state.conversation_state.business_context.model_dump(),
            "crm_context": {
                "external_key": build_crm_external_key(state.thread_id, lead_data),
            },
            "notify_owner": self._notify_owner,
        }

        raw_result = await self._capture_tool.execute(payload)
        tool_result = normalize_tool_result(raw_result)

        conversation_state = state.conversation_state.model_copy(deep=True)
        conversation_state.tooling.last_capture_result = tool_result

        if tool_result.get("status") == "captured":
            conversation_state.stage = "demo_cta"
        elif tool_result.get("status") == "missing":
            conversation_state.stage = "lead_capture"

        return {
            "conversation_state": conversation_state,
            "tool_result": tool_result,
            "tools_used": append_tool(state.tools_used, "capture_lead_if_ready"),
        }

    @staticmethod
    def after_capture(state: OrchestrationState) -> Literal["ask_missing_fields", "demo_cta", "send_reply"]:
        result = state.tool_result if isinstance(state.tool_result, dict) else {}
        status = str(result.get("status") or "").lower()
        if status == "missing":
            return "ask_missing_fields"
        if status == "captured":
            return "demo_cta"
        return "send_reply"

    async def ask_missing_fields(self, state: OrchestrationState) -> dict[str, Any]:
        result = state.tool_result if isinstance(state.tool_result, dict) else {}
        missing = result.get("missing_fields") if isinstance(result.get("missing_fields"), list) else []
        draft = build_missing_fields_question(normalize_missing_fields(missing))

        conversation_state = state.conversation_state.model_copy(deep=True)
        conversation_state.stage = "lead_capture"

        return {
            "draft_response": draft,
            "conversation_state": conversation_state,
        }

    async def demo_cta(self, state: OrchestrationState) -> dict[str, Any]:
        schedule = (state.conversation_state.lead_data.contact_schedule or "").strip()
        if schedule:
            draft = (
                "Perfecto, ya registré tu información para seguimiento comercial. "
                f"¿Te parece si agendamos la demo en el horario que compartiste: {schedule}?"
            )
        else:
            draft = (
                "Perfecto, ya registré tu información para seguimiento comercial. "
                "¿Te parece si agendamos una demo esta semana?"
            )

        conversation_state = state.conversation_state.model_copy(deep=True)
        conversation_state.stage = "demo_cta"

        return {
            "draft_response": draft,
            "conversation_state": conversation_state,
        }

    async def send_reply(self, state: OrchestrationState) -> dict[str, Any]:
        result = state.tool_result if isinstance(state.tool_result, dict) else {}
        status = str(result.get("status") or "").lower()

        final_response = (state.draft_response or "").strip()

        if status == "not_qualified":
            final_response = (
                "Gracias por la información. "
                "Por ahora no puedo avanzar a registro comercial con los datos disponibles."
            )
        elif status == "error" and not final_response:
            final_response = (
                "Tuvimos una falla para registrar el lead. "
                "¿Quieres que continuemos con una demo y damos seguimiento manual?"
            )
        elif status == "missing" and not final_response:
            missing = result.get("missing_fields") if isinstance(result.get("missing_fields"), list) else []
            final_response = build_missing_fields_question(normalize_missing_fields(missing))

        if not final_response:
            final_response = "Cuéntame un poco más y te guío al siguiente paso."

        conversation_state = state.conversation_state.model_copy(deep=True)
        normalized_question = normalize_text(state.last_user_message)
        answer_summary = compact_text(final_response, max_len=180)
        if normalized_question and answer_summary:
            conversation_state.qa_memory.answered_questions.append(
                QAItem(
                    normalized_question=normalized_question,
                    answer_summary=answer_summary,
                )
            )
            conversation_state.qa_memory.answered_questions = conversation_state.qa_memory.answered_questions[-12:]

        return {
            "final_response": final_response,
            "conversation_state": conversation_state,
        }

    async def persist_state(self, state: OrchestrationState) -> dict[str, Any]:
        messages = list(state.messages)
        if state.last_user_message:
            messages.append({"role": "user", "content": state.last_user_message})
        if state.final_response:
            messages.append({"role": "assistant", "content": state.final_response})

        persisted: dict[str, Any] = {
            "conversation_state": state.conversation_state.model_dump(),
            "last_user_message": state.last_user_message,
            "messages": messages,
        }
        if isinstance(state.tool_result, dict):
            persisted["tool_result"] = state.tool_result

        await self._session_manager.put_state(state.thread_id, persisted)
        return {"messages": messages}
