import asyncio

from app.agent.tools.knowledge_learn_tool import KnowledgeLearnDetectorOutput
from app.agent.tools.registry import ToolRegistry
from app.knowledge.providers import InMemoryKnowledgeProvider


class _StubDetector:
    def __init__(self, output: KnowledgeLearnDetectorOutput) -> None:
        self._output = output

    async def detect(self, *, source_text: str, topic_hint: str | None) -> KnowledgeLearnDetectorOutput:
        return self._output


class _EchoDetector:
    async def detect(self, *, source_text: str, topic_hint: str | None) -> KnowledgeLearnDetectorOutput:
        topic = topic_hint or "knowledge-topic"
        return KnowledgeLearnDetectorOutput(
            is_learning_instruction=True,
            confidence=0.97,
            topic=topic,
            normalized_content=source_text,
            reason="instruccion explicita",
        )


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


def test_detect_lead_capture_readiness_with_minimal_conversational_payload() -> None:
    registry = ToolRegistry()

    result = asyncio.run(
        registry.execute(
            "detect_lead_capture_readiness",
            {
                "crm_context": {"required_fields": ["email", "company", "timeline"]},
                "lead_data": {
                    "email": "ventas@acme.com",
                    "company": "Acme",
                    "timeline": "este trimestre",
                    "need": "automatizar calificacion por WhatsApp",
                    "buying_intent": "alta",
                },
            },
        )
    )

    assert result.output["ready_for_capture"] is True
    assert result.output["lead_status"] == "calificado"


def test_knowledge_learn_returns_unauthorized_for_non_admin_requestor() -> None:
    registry = ToolRegistry(
        knowledge_admin_requestors={"admin:*"},
        knowledge_learn_detector=_StubDetector(
            KnowledgeLearnDetectorOutput(
                is_learning_instruction=True,
                confidence=0.95,
                topic="propuesta de valor",
                normalized_content="Kaax automatiza chats para ventas y soporte.",
                reason="instruccion explicita",
            )
        ),
    )

    async def _run() -> dict[str, object]:
        async with registry.request_context(thread_id="thread-unauth", requestor="user:demo"):
            result = await registry.execute(
                "knowledge_learn",
                {"source_text": "aprende que ofrecemos automatizacion de WhatsApp"},
            )
            return result.output

    output = asyncio.run(_run())
    assert output["status"] == "unauthorized"
    assert output["pending"] is False


def test_knowledge_learn_requires_confirmation_for_low_confidence_and_then_saves() -> None:
    provider = InMemoryKnowledgeProvider()
    registry = ToolRegistry(
        knowledge_provider=provider,
        knowledge_admin_requestors={"admin:*"},
        knowledge_learn_confidence_threshold=0.75,
        knowledge_learn_detector=_StubDetector(
            KnowledgeLearnDetectorOutput(
                is_learning_instruction=True,
                confidence=0.61,
                topic="horario de soporte",
                normalized_content="Soporte humano de lunes a viernes de 9 a 18 horas.",
                reason="instruccion probable con ambiguedad",
            )
        ),
    )

    async def _run() -> tuple[dict[str, object], dict[str, object]]:
        async with registry.request_context(thread_id="thread-confirm", requestor="admin:owner"):
            first = await registry.execute(
                "knowledge_learn",
                {"source_text": "aprende que nuestro horario es de 9 a 18"},
            )
            second = await registry.execute(
                "knowledge_learn",
                {"source_text": "si confirmo"},
            )
            return first.output, second.output

    first_output, second_output = asyncio.run(_run())
    assert first_output["status"] == "needs_confirmation"
    assert first_output["pending"] is True
    assert second_output["status"] == "learned"
    assert isinstance(second_output.get("knowledge_id"), str)


def test_knowledge_search_is_isolated_by_tenant() -> None:
    provider = InMemoryKnowledgeProvider()
    detector = _EchoDetector()
    registry = ToolRegistry(
        knowledge_provider=provider,
        knowledge_admin_requestors={"tenant-*"},
        knowledge_learn_detector=detector,
    )

    async def _run() -> tuple[dict[str, object], dict[str, object]]:
        async with registry.request_context(thread_id="t-a", requestor="tenant-a"):
            await registry.execute(
                "knowledge_learn",
                {"source_text": "aprende que tenant-a ofrece automatizacion para ventas"},
            )
        async with registry.request_context(thread_id="t-b", requestor="tenant-b"):
            await registry.execute(
                "knowledge_learn",
                {"source_text": "aprende que tenant-b ofrece soporte tecnico 24/7"},
            )
        async with registry.request_context(thread_id="t-a-2", requestor="tenant-a"):
            result_a = await registry.execute("knowledge_search", {"query": "ventas", "limit": 5})
        async with registry.request_context(thread_id="t-b-2", requestor="tenant-b"):
            result_b = await registry.execute("knowledge_search", {"query": "soporte", "limit": 5})
        return result_a.output, result_b.output

    output_a, output_b = asyncio.run(_run())
    assert output_a["matches"]
    assert output_b["matches"]
    assert all("tenant-b" not in item["content"] for item in output_a["matches"])
    assert all("tenant-a" not in item["content"] for item in output_b["matches"])
