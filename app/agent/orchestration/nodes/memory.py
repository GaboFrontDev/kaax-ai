from __future__ import annotations

from typing import Any

from app.agent.orchestration.helpers import append_tool, compact_text, normalize_text
from app.agent.orchestration.schemas import OrchestrationState
from app.agent.tools.memory_intent_router_tool import MemoryIntentRouterTool


async def memory_lookup(state: OrchestrationState, *, memory_router: MemoryIntentRouterTool) -> dict[str, Any]:
    result = await memory_router.execute(
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
            fallback = _build_factual_fallback(normalized_question)
            if fallback:
                draft = fallback
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


def _build_factual_fallback(normalized_question: str) -> str | None:
    asks_services = any(
        token in normalized_question
        for token in (
            "servicio",
            "servicios",
            "que ofrecen",
            "que ofreces",
            "funcionalidades",
            "soluciones",
        )
    )
    if asks_services:
        return (
            "Ofrecemos 3 frentes principales: agente de IA en WhatsApp para atención y captación, "
            "calificación automática de leads con handoff a ventas, e integración con CRM (HubSpot, Salesforce u otros con API). "
            "¿Quieres que te muestre cuál te conviene según tu proceso actual?"
        )

    asks_integrations = any(
        token in normalized_question
        for token in ("integracion", "integraciones", "crm")
    )
    if asks_integrations:
        return (
            "Podemos integrarnos con CRMs como HubSpot y Salesforce, y también con CRMs que expongan API. "
            "¿Qué CRM usan hoy para validar el mejor flujo?"
        )

    asks_pricing = any(
        token in normalized_question
        for token in ("precio", "precios", "costo", "plan", "planes", "cotizacion")
    )
    if asks_pricing:
        return (
            "Puedo orientarte con los planes, pero prefiero confirmar la versión comercial vigente antes de darte detalle exacto. "
            "¿Quieres que te comparta una propuesta base y la aterrizamos a tu caso?"
        )

    return None
