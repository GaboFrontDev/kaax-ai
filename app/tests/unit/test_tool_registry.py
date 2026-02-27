import asyncio

from app.agent.tools.registry import ToolRegistry


def test_tool_registry_returns_structured_error_for_invalid_input() -> None:
    registry = ToolRegistry()

    result = asyncio.run(registry.execute("crm_upsert_quote", {}))

    assert result.tool == "crm_upsert_quote"
    assert "error" in result.output


def test_tool_registry_validates_success_output_shape() -> None:
    registry = ToolRegistry()

    result = asyncio.run(registry.execute("crm_upsert_quote", {"payload": {"quote_id": "quote-123"}}))

    assert result.output["status"] == "upserted"
    assert result.output["crm_id"] == "quote-123"


def test_detect_lead_capture_readiness_returns_ready_when_context_is_complete() -> None:
    registry = ToolRegistry()

    result = asyncio.run(
        registry.execute(
            "detect_lead_capture_readiness",
            {
                "business_context": {
                    "what_sells": "Licencias SaaS B2B",
                    "sales_cycle": "30-45 dias",
                    "qualification_fields": ["company", "budget", "timeline", "decision_role"],
                    "first_call_questions": ["Uso principal", "equipo", "urgencia"],
                },
                "whatsapp_context": {
                    "brand_tone": "consultivo",
                    "service_hours": "Lun-Vie 9-18",
                    "primary_language": "es",
                    "flow_type": "inbound_outbound",
                },
                "crm_context": {
                    "crm_name": "HubSpot",
                    "required_fields": ["email", "company", "timeline"],
                    "qualified_pipeline_stage": "SQL",
                },
                "agent_limits": {
                    "resolves_alone": ["FAQ", "descubrimiento inicial"],
                    "escalation_triggers": ["alto_valor"],
                    "forbidden_statements": ["promesas de entrega sin confirmacion"],
                    "disqualification_closure": "cierre cordial y oferta de recontacto",
                },
                "lead_data": {
                    "email": "compras@acme.com",
                    "company": "Acme",
                    "timeline": "este trimestre",
                    "need": "automatizar onboarding",
                    "buying_intent": "alta",
                    "decision_role": "director de operaciones",
                    "budget": "5000 usd",
                },
            },
        )
    )

    assert result.output["ready_for_capture"] is True
    assert result.output["lead_status"] == "calificado"
    assert result.output["next_action"] == "registro_crm"
    assert result.output["missing_critical_fields"] == []


def test_detect_lead_capture_readiness_requests_missing_critical_data() -> None:
    registry = ToolRegistry()

    result = asyncio.run(
        registry.execute(
            "detect_lead_capture_readiness",
            {
                "business_context": {},
                "whatsapp_context": {},
                "crm_context": {"crm_name": "Zoho", "required_fields": ["email"], "qualified_pipeline_stage": "MQL"},
                "agent_limits": {},
                "lead_data": {"company": "Acme"},
            },
        )
    )

    assert result.output["ready_for_capture"] is False
    assert result.output["lead_status"] == "en_revision"
    assert result.output["next_action"] == "solicitar_datos_faltantes"
    assert "lead_data.email" in result.output["missing_critical_fields"]


def test_capture_lead_if_ready_captures_when_complete() -> None:
    registry = ToolRegistry()

    result = asyncio.run(
        registry.execute(
            "capture_lead_if_ready",
            {
                "business_context": {
                    "what_sells": "Licencias SaaS B2B",
                    "sales_cycle": "30-45 dias",
                    "qualification_fields": ["company", "budget", "timeline", "decision_role"],
                    "first_call_questions": ["Uso principal", "equipo", "urgencia"],
                },
                "whatsapp_context": {
                    "brand_tone": "consultivo",
                    "service_hours": "Lun-Vie 9-18",
                    "primary_language": "es",
                    "flow_type": "inbound_outbound",
                },
                "crm_context": {
                    "crm_name": "HubSpot",
                    "required_fields": ["email", "company", "timeline"],
                    "qualified_pipeline_stage": "SQL",
                },
                "agent_limits": {
                    "resolves_alone": ["FAQ", "descubrimiento inicial"],
                    "escalation_triggers": ["alto_valor"],
                    "forbidden_statements": ["promesas de entrega sin confirmacion"],
                    "disqualification_closure": "cierre cordial y oferta de recontacto",
                },
                "lead_data": {
                    "email": "compras@acme.com",
                    "company": "Acme",
                    "timeline": "este trimestre",
                    "need": "automatizar onboarding",
                    "buying_intent": "alta",
                    "decision_role": "director de operaciones",
                    "budget": "5000 usd",
                },
                "notify_owner": False,
            },
        )
    )

    assert result.output["status"] == "captured"
    assert result.output["lead_status"] == "calificado"
    assert result.output["owner_notification"] == "skipped"
    assert result.output["crm_result"]["status"] == "upserted"


def test_capture_lead_if_ready_returns_missing_fields_without_capture() -> None:
    registry = ToolRegistry()

    result = asyncio.run(
        registry.execute(
            "capture_lead_if_ready",
            {
                "crm_context": {"crm_name": "Zoho", "required_fields": ["email"], "qualified_pipeline_stage": "MQL"},
                "lead_data": {"company": "Acme"},
                "notify_owner": True,
            },
        )
    )

    assert result.output["status"] == "missing_fields"
    assert result.output["lead_status"] == "en_revision"
    assert "crm_result" not in result.output
    assert "lead_data.email" in result.output["missing_critical_fields"]
