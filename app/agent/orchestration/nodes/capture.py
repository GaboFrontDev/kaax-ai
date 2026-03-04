from __future__ import annotations

from typing import Any, Literal

from app.agent.orchestration.capture_utils import normalize_tool_result
from app.agent.orchestration.helpers import (
    append_tool,
    build_crm_external_key,
    build_missing_fields_question,
    normalize_missing_fields,
)
from app.agent.orchestration.schemas import OrchestrationState
from app.agent.tools.capture_lead_if_ready_tool import CaptureLeadIfReadyTool


def lead_capture_gate(state: OrchestrationState) -> Literal["capture_lead", "send_reply"]:
    router = state.router
    if router is not None and router.flags.lead_capture_ready:
        return "capture_lead"
    return "send_reply"


async def capture_lead(
    state: OrchestrationState,
    *,
    capture_tool: CaptureLeadIfReadyTool | None,
    notify_owner: bool,
) -> dict[str, Any]:
    if capture_tool is None:
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
        "notify_owner": bool(notify_owner),
    }

    raw_result = await capture_tool.execute(payload)
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


def after_capture(state: OrchestrationState) -> Literal["ask_missing_fields", "demo_cta", "send_reply"]:
    result = state.tool_result if isinstance(state.tool_result, dict) else {}
    status = str(result.get("status") or "").lower()
    if status == "missing":
        return "ask_missing_fields"
    if status == "captured":
        return "demo_cta"
    return "send_reply"


async def ask_missing_fields(state: OrchestrationState) -> dict[str, Any]:
    result = state.tool_result if isinstance(state.tool_result, dict) else {}
    missing = result.get("missing_fields") if isinstance(result.get("missing_fields"), list) else []
    draft = build_missing_fields_question(normalize_missing_fields(missing))

    conversation_state = state.conversation_state.model_copy(deep=True)
    conversation_state.stage = "lead_capture"

    return {
        "draft_response": draft,
        "conversation_state": conversation_state,
    }


async def demo_cta(state: OrchestrationState) -> dict[str, Any]:
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
