from __future__ import annotations

from app.agent.orchestration.routing_rules import derive_router_and_state, normalize_conversation_state
from app.agent.orchestration.schemas import CapturedContact, ConversationState, LeadState


def test_router_greeting_path() -> None:
    router, updated = derive_router_and_state(
        user_message="hola",
        conversation_state=ConversationState(),
    )

    assert router.mode == "greeting"
    assert router.agent == "greeting"
    assert router.intent == "unknown"
    assert router.qualification == "cold"
    assert router.missing_fields == ["contact_name", "phone", "contact_schedule"]
    assert router.next_action == "ask_question"
    assert updated.mode == "greeting"
    assert updated.lead.intent == "unknown"
    assert updated.lead.status == "en_revision"


def test_router_pricing_path() -> None:
    router, updated = derive_router_and_state(
        user_message="cuanto cuesta",
        conversation_state=ConversationState(),
    )

    assert router.mode == "discovery"
    assert router.agent == "inventory"
    assert router.intent == "pricing"
    assert router.qualification == "warm"
    assert router.missing_fields == ["contact_name", "phone", "contact_schedule"]
    assert router.next_action == "answer"
    assert updated.lead.intent == "pricing"


def test_router_purchase_capture_completion_path() -> None:
    router, updated = derive_router_and_state(
        user_message="quiero demo manana",
        conversation_state=ConversationState(),
    )

    assert router.mode == "capture_completion"
    assert router.agent == "core_capture"
    assert router.intent == "purchase_intent"
    assert router.qualification == "hot"
    assert router.missing_fields == ["contact_name", "phone", "contact_schedule"]
    assert router.next_action == "ask_question"
    assert updated.mode == "capture_completion"
    assert updated.lead.intent == "purchase_intent"


def test_router_handoff_priority() -> None:
    router, updated = derive_router_and_state(
        user_message="quiero hablar con un asesor humano",
        conversation_state=ConversationState(),
    )
    assert router.mode == "handoff"
    assert router.next_action == "handoff"
    assert updated.mode == "handoff"


def test_router_does_not_handoff_for_product_question_with_word_agentes() -> None:
    router, updated = derive_router_and_state(
        user_message="acerca de los agentes como es que esto funciona?",
        conversation_state=ConversationState(),
    )
    assert router.mode != "handoff"
    assert router.agent in {"knowledge", "greeting", "inventory"}
    assert router.next_action != "handoff"
    assert updated.mode != "handoff"


def test_router_support_keyword_sets_support_mode() -> None:
    router, updated = derive_router_and_state(
        user_message="soporte",
        conversation_state=ConversationState(),
    )
    assert router.intent == "support"
    assert router.mode == "support_answer"
    assert router.agent == "knowledge"
    assert updated.mode == "support_answer"


def test_router_implementation_signal_enters_capture_completion() -> None:
    router, updated = derive_router_and_state(
        user_message="quiero evaluar implementacion",
        conversation_state=ConversationState(),
    )
    assert router.intent == "purchase_intent"
    assert router.mode == "capture_completion"
    assert router.agent == "core_capture"
    assert router.next_action == "ask_question"
    assert updated.mode == "capture_completion"


def test_router_extracts_captured_fields_when_present() -> None:
    router, updated = derive_router_and_state(
        user_message="mi nombre es Ana y mi telefono es +52 1555 000 0001, horario 10 am",
        conversation_state=ConversationState(),
    )

    assert updated.captured.contact_name == "Ana"
    assert updated.captured.phone == "+5215550000001"
    assert updated.captured.contact_schedule == "10 am"
    assert router.missing_fields == []
    assert updated.lead.status == "calificado"


def test_router_keeps_capture_completion_with_full_contact_and_procedamos() -> None:
    router, updated = derive_router_and_state(
        user_message="procedamos",
        conversation_state=ConversationState(
            mode="capture_completion",
            captured=CapturedContact(
                contact_name="Ana",
                phone="+5215550000001",
                contact_schedule="manana 10am",
            ),
            lead=LeadState(
                intent="purchase_intent",
                qualification="hot",
                status="en_revision",
            ),
        ),
    )
    assert router.mode == "capture_completion"
    assert router.agent == "core_capture"
    assert router.next_action == "capture_lead"
    assert updated.lead.intent == "purchase_intent"
    assert updated.lead.status == "calificado"


def test_normalize_conversation_state_migrates_legacy_intent_and_qualification() -> None:
    state = normalize_conversation_state(
        {
            "mode": "discovery",
            "captured": {"contact_name": "Ana"},
            "intent": "pricing",
            "qualification": "warm",
        }
    )
    assert state.lead.intent == "pricing"
    assert state.lead.qualification == "warm"
    assert state.lead.status == "en_revision"


def test_router_keeps_pricing_intent_for_short_followup_mensual() -> None:
    router, updated = derive_router_and_state(
        user_message="mensual",
        conversation_state=ConversationState(
            mode="discovery",
            lead=LeadState(intent="pricing", qualification="warm", status="en_revision"),
        ),
    )
    assert router.intent == "pricing"
    assert router.agent == "inventory"
    assert updated.lead.intent == "pricing"


def test_router_extracts_contact_from_short_format_omar_numero_de_9_a_6() -> None:
    router, updated = derive_router_and_state(
        user_message="omar numero 5544332211 de 9 a 6 entre semana",
        conversation_state=ConversationState(mode="capture_completion"),
    )
    assert updated.captured.contact_name == "omar"
    assert updated.captured.phone == "5544332211"
    assert updated.captured.contact_schedule == "de 9 a 6 entre semana"
    assert router.mode == "capture_completion"
    assert router.next_action == "capture_lead"


def test_router_extracts_name_from_nombre_es_format() -> None:
    router, updated = derive_router_and_state(
        user_message="nombre es gabriel",
        conversation_state=ConversationState(mode="capture_completion"),
    )
    assert updated.captured.contact_name == "gabriel"
    assert router.mode == "capture_completion"
    assert router.next_action == "ask_question"
    assert "phone" in router.missing_fields
    assert "contact_schedule" in router.missing_fields


def test_router_keeps_capture_mode_sticky_until_calificado() -> None:
    router, updated = derive_router_and_state(
        user_message="ya te los di",
        conversation_state=ConversationState(
            mode="capture_completion",
            captured=CapturedContact(
                contact_name="Gabriel",
                phone=None,
                contact_schedule="de 9 a 7",
            ),
            lead=LeadState(
                intent="purchase_intent",
                qualification="hot",
                status="en_revision",
            ),
        ),
    )
    assert router.mode == "capture_completion"
    assert router.agent == "core_capture"
    assert router.next_action == "ask_question"
    assert router.missing_fields == ["phone"]
    assert updated.mode == "capture_completion"
    assert updated.lead.status == "en_revision"


def test_router_does_not_stay_in_capture_for_generic_pricing_after_capture_done() -> None:
    router, updated = derive_router_and_state(
        user_message="que costo tiene y que incluye?",
        conversation_state=ConversationState(
            mode="capture_completion",
            captured=CapturedContact(
                contact_name="Omar",
                phone="5544332211",
                contact_schedule="de 9 a 6 entre semana",
            ),
            lead=LeadState(intent="purchase_intent", qualification="hot", status="calificado"),
        ),
    )
    assert router.mode == "discovery"
    assert router.agent == "inventory"
    assert router.intent == "pricing"
    assert router.next_action == "answer"
    assert updated.mode == "discovery"
