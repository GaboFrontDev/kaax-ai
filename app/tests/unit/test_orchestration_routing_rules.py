from __future__ import annotations

from app.agent.orchestration.routing_rules import derive_router_and_state, normalize_conversation_state
from app.agent.orchestration.schemas import ConversationState, QAItem


def test_router_greeting_starts_discovery_flow() -> None:
    router, updated = derive_router_and_state(
        user_message="hola",
        conversation_state=ConversationState(),
    )

    assert router.route == "discovery_value"
    assert router.stage == "greeting"
    assert router.intent in {"exploring", "unknown"}
    assert router.next_action == "ask_question"
    assert updated.stage == "greeting"
    assert updated.flags.needs_factual_lookup is False


def test_router_factual_question_forces_memory_lookup() -> None:
    router, updated = derive_router_and_state(
        user_message="¿Qué integraciones CRM tienen y cuál es el precio?",
        conversation_state=ConversationState(),
    )

    assert router.route == "memory_lookup"
    assert router.flags.needs_factual_lookup is True
    assert updated.flags.needs_factual_lookup is True
    assert router.intent == "interested"


def test_router_buying_signal_moves_to_lead_capture() -> None:
    state = ConversationState()
    state.business_context.use_case = "captacion de leads por whatsapp"
    state.business_context.pain_points = ["tiempo de respuesta"]

    router, updated = derive_router_and_state(
        user_message="quiero agendar demo",
        conversation_state=state,
    )

    assert router.stage == "lead_capture"
    assert router.flags.has_buying_signal is True
    assert router.flags.lead_capture_ready is True
    assert router.intent == "demo_requested"
    assert updated.sales.qualification in {"warm", "hot"}


def test_router_extracts_contact_fields_from_compact_message() -> None:
    router, updated = derive_router_and_state(
        user_message="mi nombre es Ana, telefono +52 1555 000 0001, horario 10 am",
        conversation_state=ConversationState(),
    )

    assert updated.lead_data.contact_name == "Ana"
    assert updated.lead_data.phone == "+5215550000001"
    assert updated.lead_data.contact_schedule == "10 am"
    assert router.missing_fields == []


def test_router_detects_repeated_question() -> None:
    state = ConversationState()
    state.qa_memory.answered_questions.append(
        QAItem(
            normalized_question="que integraciones crm tienen",
            answer_summary="Integramos con CRMs principales y APIs REST.",
        )
    )

    router, updated = derive_router_and_state(
        user_message="Que integraciones CRM tienen?",
        conversation_state=state,
    )

    assert router.route == "repeat_handler"
    assert router.flags.is_repeat_question is True
    assert updated.flags.is_repeat_question is True


def test_normalize_conversation_state_migrates_legacy_payload() -> None:
    state = normalize_conversation_state(
        {
            "mode": "capture_completion",
            "captured": {
                "contact_name": "Omar",
                "phone": "5544332211",
                "contact_schedule": "de 9 a 6",
            },
            "lead": {
                "intent": "purchase_intent",
                "qualification": "hot",
            },
        }
    )

    assert state.stage == "lead_capture"
    assert state.lead_data.contact_name == "Omar"
    assert state.sales.intent == "demo_requested"
    assert state.sales.qualification == "hot"


def test_router_next_action_capture_when_ready_and_complete() -> None:
    state = ConversationState()
    state.business_context.use_case = "captacion y calificacion"
    state.business_context.pain_points = ["seguimiento inconsistente"]
    state.lead_data.contact_name = "Omar"
    state.lead_data.phone = "5544332211"
    state.lead_data.contact_schedule = "de 9 a 6"

    router, _ = derive_router_and_state(
        user_message="quiero demo mañana",
        conversation_state=state,
    )

    assert router.flags.lead_capture_ready is True
    assert router.missing_fields == []
    assert router.next_action == "capture_lead"
