from app.agent.lead_capture import (
    build_conversational_lead_payload,
    build_capture_response,
    is_affirmative_capture,
    is_capture_request,
    parse_lead_payload_from_text,
)


def test_parse_lead_payload_from_text_extracts_sections() -> None:
    text = """
    business_context:
    - what_sells: Automatizacion de chats
    - sales_cycle: 30 dias

    whatsapp_context:
    - brand_tone: consultivo
    - flow_type: inbound_outbound

    crm_context:
    - crm_name: HubSpot
    - required_fields: email,company,timeline

    lead_data:
    - company: Acme
    - email: compras@acme.com
    - buying_intent: alta
    """

    payload = parse_lead_payload_from_text(text)
    assert payload is not None
    assert payload["crm_context"]["crm_name"] == "HubSpot"
    assert payload["crm_context"]["required_fields"] == ["email", "company", "timeline"]
    assert payload["lead_data"]["company"] == "Acme"


def test_capture_detection_understands_request_and_affirmation() -> None:
    assert is_capture_request("Quiero registrar un lead y capturalo en CRM")
    assert is_affirmative_capture("Si, registralo ahora en CRM")


def test_build_capture_response_formats_status() -> None:
    captured = build_capture_response({"status": "captured", "crm_result": {"crm_id": "abc-123"}})
    assert "pronto estaremos en contacto" in captured.lower()

    missing = build_capture_response(
        {"status": "missing_fields", "missing_critical_fields": ["lead_data.email", "lead_data.timeline"]}
    )
    assert "lead_data.email" in missing


def test_build_conversational_lead_payload_extracts_qualification_data() -> None:
    messages = [
        {"role": "user", "content": "Hola, soy Juan Perez de Acme SA"},
        {"role": "user", "content": "Me interesa contratar automatizacion de WhatsApp y CRM"},
        {"role": "user", "content": "Mi correo es juan@acme.com y presupuesto de 5000 usd este trimestre"},
    ]

    payload = build_conversational_lead_payload(messages)
    assert payload is not None
    assert payload["lead_data"]["email"] == "juan@acme.com"
    assert payload["lead_data"]["company"] == "Acme SA"
    assert payload["lead_data"]["buying_intent"] == "alta"
