from __future__ import annotations

from typing import Any

from app.agent.orchestration.helpers import (
    build_missing_fields_question,
    compact_text,
    normalize_missing_fields,
    normalize_text,
)
from app.agent.orchestration.schemas import OrchestrationState, QAItem
from app.memory.session_manager import SessionManager


async def send_reply(state: OrchestrationState) -> dict[str, Any]:
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


async def persist_state(state: OrchestrationState, *, session_manager: SessionManager) -> dict[str, Any]:
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

    await session_manager.put_state(state.thread_id, persisted)
    return {"messages": messages}
