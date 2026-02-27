from app.agent.intent_router import build_routing_response, route_intent


def test_intent_router_marks_short_greeting_as_needs_clarification() -> None:
    decision = route_intent("hola")
    assert decision.route == "needs_clarification"
    assert decision.confidence >= 0.9


def test_intent_router_marks_no_task_message_as_needs_clarification() -> None:
    decision = route_intent("soy de seattle")
    assert decision.route == "needs_clarification"
    assert decision.reason == "no_task_detected"


def test_intent_router_marks_generic_help_as_needs_clarification() -> None:
    decision = route_intent("necesito ayuda")
    assert decision.route == "needs_clarification"


def test_intent_router_marks_dev_request_as_out_of_scope() -> None:
    decision = route_intent("quiero programar un componente en react")
    assert decision.route == "out_of_scope"
    assert decision.confidence >= 0.7


def test_intent_router_marks_business_automation_request_as_in_scope() -> None:
    decision = route_intent("quiero automatizar mensajes de whatsapp e integrar hubspot")
    assert decision.route == "in_scope"
    assert decision.confidence >= 0.7


def test_intent_router_marks_goodbye_as_conversation_end() -> None:
    decision = route_intent("gracias, eso es todo")
    assert decision.route == "conversation_end"
    assert decision.confidence >= 0.9


def test_routing_response_out_of_scope_mentions_service_scope() -> None:
    decision = route_intent("programar en react")
    response = build_routing_response(decision)
    assert "kaax" in response.lower()
    assert "automatizar" in response.lower()


def test_routing_response_conversation_end_is_closure() -> None:
    decision = route_intent("adios")
    response = build_routing_response(decision)
    assert "cerramos" in response.lower()


def test_routing_response_first_turn_greeting_requests_lead_fields() -> None:
    decision = route_intent("hola")
    response = build_routing_response(decision, first_turn_greeting=True)
    lowered = response.lower()
    assert "nombre y empresa" in lowered
    assert "crm actual" in lowered
    assert "volumen aproximado" in lowered


def test_routing_response_follow_up_greeting_is_shorter() -> None:
    decision = route_intent("hola")
    response = build_routing_response(decision, first_turn_greeting=False)
    lowered = response.lower()
    assert "nombre y empresa" not in lowered
    assert "proceso a automatizar" in lowered
