from __future__ import annotations

import re
from typing import Any

from app.agent.orchestration.helpers import lookup_previous_summary
from app.agent.orchestration.schemas import OrchestrationState


async def repeat_handler(state: OrchestrationState) -> dict[str, Any]:
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


async def discovery_value(state: OrchestrationState) -> dict[str, Any]:
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


async def compose_reply(state: OrchestrationState) -> dict[str, Any]:
    draft = (state.draft_response or "").strip()
    if not draft:
        draft = "Perfecto, cuéntame un poco más de tu operación y avanzamos paso a paso."

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", draft) if part.strip()]
    if len(sentences) > 3:
        draft = " ".join(sentences[:3])

    return {"draft_response": draft}
