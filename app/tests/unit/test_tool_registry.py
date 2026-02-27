import asyncio

from app.agent.tools.registry import ToolRegistry


def test_tool_registry_returns_structured_error_for_invalid_input() -> None:
    registry = ToolRegistry()

    result = asyncio.run(registry.execute("retrieve_markets", {"country_code": "US"}))

    assert result.tool == "retrieve_markets"
    assert "error" in result.output


def test_tool_registry_validates_success_output_shape() -> None:
    registry = ToolRegistry()

    result = asyncio.run(registry.execute("retrieve_markets", {"query": "ret", "country_code": "US", "limit": 2}))

    assert "markets" in result.output
    first = result.output["markets"][0]
    assert set(first.keys()) == {"name", "country_code", "score"}


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
