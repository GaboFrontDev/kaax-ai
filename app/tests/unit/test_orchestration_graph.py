from __future__ import annotations

import asyncio

from app.agent.orchestration.graph import build_mvp_orchestration_graph
from app.agent.orchestration.schemas import CapturedContact, ConversationState, LeadState, PricingContext
from app.agent.tools.context import ToolRequestContextManager
from app.knowledge.providers import InMemoryKnowledgeProvider
from app.memory.checkpoint_store import InMemoryCheckpointStore
from app.memory.locks import InMemorySessionLockManager
from app.memory.session_manager import SessionManager


def _build_session_manager() -> SessionManager:
    return SessionManager(
        checkpoint_store=InMemoryCheckpointStore(),
        lock_manager=InMemorySessionLockManager(),
        session_timeout_seconds=1800,
    )


def test_graph_support_message_requests_technical_details() -> None:
    graph = build_mvp_orchestration_graph(
        session_manager=_build_session_manager(),
        knowledge_provider=InMemoryKnowledgeProvider(),
        tool_context_manager=ToolRequestContextManager(),
        capture_tool=None,
    )
    result = asyncio.run(
        graph.ainvoke(
            {
                "thread_id": "t-support-1",
                "requestor": "tenant-1",
                "last_user_message": "soporte",
                "conversation_state": ConversationState().model_dump(),
            }
        )
    )

    response = str(result.get("final_response") or "")
    assert "mensaje de error exacto" in response


def test_graph_product_inquiry_ignores_irrelevant_kb_content() -> None:
    knowledge_provider = InMemoryKnowledgeProvider()
    asyncio.run(
        knowledge_provider.upsert_topic(
            tenant_id="tenant-1",
            agent_id="default",
            topic="servicios",
            content="We do not sell pots",
            source="unit_test",
            author="tester",
            metadata={},
        )
    )

    graph = build_mvp_orchestration_graph(
        session_manager=_build_session_manager(),
        knowledge_provider=knowledge_provider,
        tool_context_manager=ToolRequestContextManager(),
        capture_tool=None,
    )
    result = asyncio.run(
        graph.ainvoke(
            {
                "thread_id": "t-product-1",
                "requestor": "tenant-1",
                "last_user_message": "que servicios tienes?",
                "conversation_state": ConversationState().model_dump(),
            }
        )
    )

    response = str(result.get("final_response") or "")
    assert "automatiza atencion inicial" in response
    assert "we do not sell pots" not in response.lower()


def test_graph_uses_llm_subagent_when_runner_is_provided() -> None:
    class _StubRunner:
        async def run(self, *, agent_name, user_message, context):  # noqa: ANN001
            assert agent_name == "knowledge"
            assert "servicios" in user_message
            assert isinstance(context, dict)
            return "Respuesta dinamica de subagente"

    graph = build_mvp_orchestration_graph(
        session_manager=_build_session_manager(),
        knowledge_provider=InMemoryKnowledgeProvider(),
        tool_context_manager=ToolRequestContextManager(),
        capture_tool=None,
        subagent_runner=_StubRunner(),
    )
    result = asyncio.run(
        graph.ainvoke(
            {
                "thread_id": "t-llm-1",
                "requestor": "tenant-1",
                "last_user_message": "que servicios tienes?",
                "conversation_state": ConversationState().model_dump(),
            }
        )
    )
    assert result.get("final_response") == "Respuesta dinamica de subagente"


def test_graph_greeting_uses_greeting_agent_and_avoids_contact_capture_prompt() -> None:
    graph = build_mvp_orchestration_graph(
        session_manager=_build_session_manager(),
        knowledge_provider=InMemoryKnowledgeProvider(),
        tool_context_manager=ToolRequestContextManager(),
        capture_tool=None,
    )
    result = asyncio.run(
        graph.ainvoke(
            {
                "thread_id": "t-greeting-1",
                "requestor": "tenant-1",
                "last_user_message": "hola",
                "conversation_state": ConversationState().model_dump(),
            }
        )
    )

    router = result.get("router")
    assert router is not None
    assert router.agent == "greeting"
    response = str(result.get("final_response") or "").lower()
    assert "nombre" not in response
    assert "telefono" not in response
    assert "horario" not in response


def test_graph_supervisor_fallback_safe_when_supervisor_returns_invalid_payload() -> None:
    async def _invalid_supervisor(_: str, __: dict[str, object]):  # noqa: ANN001
        return {"foo": "bar"}

    graph = build_mvp_orchestration_graph(
        session_manager=_build_session_manager(),
        knowledge_provider=InMemoryKnowledgeProvider(),
        tool_context_manager=ToolRequestContextManager(),
        capture_tool=None,
        supervisor_override=_invalid_supervisor,
    )
    result = asyncio.run(
        graph.ainvoke(
            {
                "thread_id": "t-supervisor-fallback-1",
                "requestor": "tenant-1",
                "last_user_message": "mhm",
                "conversation_state": ConversationState().model_dump(),
            }
        )
    )

    router = result.get("router")
    assert router is not None
    assert router.agent == "knowledge"
    assert router.next_action == "ask_question"


def test_graph_capture_completion_with_full_contact_triggers_capture_tool() -> None:
    class _CaptureTool:
        async def execute(self, payload):  # noqa: ANN001
            assert payload["lead_data"]["contact_name"] == "Ana"
            return {
                "status": "captured",
                "lead_status": "calificado",
                "missing_critical_fields": [],
                "qualification_evidence": ["contact_name: Ana"],
                "crm_result": {"crm_id": "crm-123", "status": "upserted"},
                "structured_payload": {"lead_data": payload["lead_data"]},
            }

    session_manager = _build_session_manager()
    graph = build_mvp_orchestration_graph(
        session_manager=session_manager,
        knowledge_provider=InMemoryKnowledgeProvider(),
        tool_context_manager=ToolRequestContextManager(),
        capture_tool=_CaptureTool(),
    )
    result = asyncio.run(
        graph.ainvoke(
            {
                "thread_id": "t-capture-full-1",
                "requestor": "tenant-1",
                "last_user_message": "procedamos",
                "conversation_state": ConversationState(
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
                ).model_dump(),
            }
        )
    )

    assert "Registro completado" in str(result.get("final_response") or "")
    persisted = asyncio.run(session_manager.get_state("t-capture-full-1"))
    assert isinstance(persisted, dict)
    assert persisted["conversation_state"]["lead"]["status"] == "calificado"
    assert persisted["tool_result"]["status"] == "captured"


def test_graph_inventory_pricing_turn_uses_canonical_offer_even_with_snapshot() -> None:
    session_manager = _build_session_manager()
    graph = build_mvp_orchestration_graph(
        session_manager=session_manager,
        knowledge_provider=InMemoryKnowledgeProvider(),
        tool_context_manager=ToolRequestContextManager(),
        capture_tool=None,
    )
    result = asyncio.run(
        graph.ainvoke(
            {
                "thread_id": "t-inventory-snapshot-1",
                "requestor": "tenant-1",
                "last_user_message": "cuanto cuesta",
                "conversation_state": ConversationState(
                    pricing_context=PricingContext(
                        verified_summary="Plan base desde 100 USD mensuales.",
                        source="snapshot",
                        query="cuanto cuesta",
                    )
                ).model_dump(),
            }
        )
    )

    response = str(result.get("final_response") or "").lower()
    assert "18,000 mxn" in response
    assert "no tenemos tarifa anual publicada" in response
    assert "100 usd" not in response


def test_graph_inventory_monthly_followup_returns_explicit_monthly_price() -> None:
    session_manager = _build_session_manager()
    graph = build_mvp_orchestration_graph(
        session_manager=session_manager,
        knowledge_provider=InMemoryKnowledgeProvider(),
        tool_context_manager=ToolRequestContextManager(),
        capture_tool=None,
    )
    result = asyncio.run(
        graph.ainvoke(
            {
                "thread_id": "t-inventory-monthly-1",
                "requestor": "tenant-1",
                "last_user_message": "mensual",
                "conversation_state": ConversationState(
                    mode="discovery",
                    lead=LeadState(intent="pricing", qualification="warm", status="en_revision"),
                ).model_dump(),
            }
        )
    )

    response = str(result.get("final_response") or "").lower()
    assert "18,000 mxn" in response
    assert "iva" in response


def test_graph_inventory_pricing_request_ignores_llm_hallucinated_annual_amount() -> None:
    class _HallucinatingRunner:
        async def run(self, *, agent_name, user_message, context):  # noqa: ANN001
            raise AssertionError(f"inventory runner should not be called for pricing request: {agent_name}")

    graph = build_mvp_orchestration_graph(
        session_manager=_build_session_manager(),
        knowledge_provider=InMemoryKnowledgeProvider(),
        tool_context_manager=ToolRequestContextManager(),
        capture_tool=None,
        subagent_runner=_HallucinatingRunner(),
    )
    result = asyncio.run(
        graph.ainvoke(
            {
                "thread_id": "t-inventory-no-hallucination-1",
                "requestor": "tenant-1",
                "last_user_message": "precios",
                "conversation_state": ConversationState().model_dump(),
            }
        )
    )
    response = str(result.get("final_response") or "").lower()
    assert "18,000 mxn" in response
    assert "no tenemos tarifa anual publicada" in response
    assert "216,000" not in response
    assert "216000" not in response


def test_graph_capture_flow_with_contact_request_and_compact_contact_message() -> None:
    class _CaptureTool:
        async def execute(self, payload):  # noqa: ANN001
            assert payload["lead_data"]["contact_name"] == "omar"
            assert payload["lead_data"]["phone"] == "5544332211"
            assert payload["lead_data"]["contact_schedule"] == "de 9 a 6 entre semana"
            return {
                "status": "captured",
                "lead_status": "calificado",
                "missing_critical_fields": [],
                "qualification_evidence": ["contact_name: omar"],
                "crm_result": {"crm_id": "crm-om", "status": "upserted"},
                "structured_payload": {"lead_data": payload["lead_data"]},
            }

    session_manager = _build_session_manager()
    graph = build_mvp_orchestration_graph(
        session_manager=session_manager,
        knowledge_provider=InMemoryKnowledgeProvider(),
        tool_context_manager=ToolRequestContextManager(),
        capture_tool=_CaptureTool(),
    )

    first = asyncio.run(
        graph.ainvoke(
            {
                "thread_id": "t-capture-contact-1",
                "requestor": "tenant-1",
                "last_user_message": "quisiera que me contacten",
                "conversation_state": ConversationState().model_dump(),
            }
        )
    )
    assert "nombre" in str(first.get("final_response") or "").lower()

    second = asyncio.run(
        graph.ainvoke(
            {
                "thread_id": "t-capture-contact-1",
                "requestor": "tenant-1",
                "last_user_message": "omar numero 5544332211 de 9 a 6 entre semana",
                "conversation_state": first["conversation_state"].model_dump(),
            }
        )
    )
    assert "registro completado" in str(second.get("final_response") or "").lower()
    persisted = asyncio.run(session_manager.get_state("t-capture-contact-1"))
    assert isinstance(persisted, dict)
    assert persisted["conversation_state"]["lead"]["status"] == "calificado"


def test_graph_can_report_captured_user_info() -> None:
    graph = build_mvp_orchestration_graph(
        session_manager=_build_session_manager(),
        knowledge_provider=InMemoryKnowledgeProvider(),
        tool_context_manager=ToolRequestContextManager(),
        capture_tool=None,
    )
    result = asyncio.run(
        graph.ainvoke(
            {
                "thread_id": "t-user-info-1",
                "requestor": "tenant-1",
                "last_user_message": "puedes decirme mi informacion?",
                "conversation_state": ConversationState(
                    mode="discovery",
                    captured=CapturedContact(
                        contact_name="Eduardo",
                        phone="5544332211",
                        contact_schedule="de 9 a 5 entre semana",
                    ),
                    lead=LeadState(intent="purchase_intent", qualification="hot", status="en_revision"),
                ).model_dump(),
            }
        )
    )

    response = str(result.get("final_response") or "").lower()
    assert "informacion que tengo registrada" in response
    assert "eduardo" in response
    assert "5544332211" in response


def test_graph_after_capture_can_answer_pricing_without_reentering_capture_loop() -> None:
    class _CaptureTool:
        async def execute(self, payload):  # noqa: ANN001
            return {
                "status": "captured",
                "lead_status": "calificado",
                "missing_critical_fields": [],
                "qualification_evidence": ["ok"],
                "crm_result": {"crm_id": "crm-1", "status": "upserted"},
                "structured_payload": {"lead_data": payload["lead_data"]},
            }

    session_manager = _build_session_manager()
    graph = build_mvp_orchestration_graph(
        session_manager=session_manager,
        knowledge_provider=InMemoryKnowledgeProvider(),
        tool_context_manager=ToolRequestContextManager(),
        capture_tool=_CaptureTool(),
    )

    first = asyncio.run(
        graph.ainvoke(
            {
                "thread_id": "t-post-capture-pricing-1",
                "requestor": "tenant-1",
                "last_user_message": "omar numero 5544332211 de 9 a 6 entre semana",
                "conversation_state": ConversationState(mode="capture_completion").model_dump(),
            }
        )
    )
    assert "registro completado" in str(first.get("final_response") or "").lower()

    second = asyncio.run(
        graph.ainvoke(
            {
                "thread_id": "t-post-capture-pricing-1",
                "requestor": "tenant-1",
                "last_user_message": "que costo tiene y que incluye?",
                "conversation_state": first["conversation_state"].model_dump(),
            }
        )
    )
    response = str(second.get("final_response") or "").lower()
    assert "18,000 mxn" in response
    assert "registro completado" not in response


def test_graph_can_answer_contact_timeline_after_lead_is_captured() -> None:
    graph = build_mvp_orchestration_graph(
        session_manager=_build_session_manager(),
        knowledge_provider=InMemoryKnowledgeProvider(),
        tool_context_manager=ToolRequestContextManager(),
        capture_tool=None,
    )
    result = asyncio.run(
        graph.ainvoke(
            {
                "thread_id": "t-contact-timeline-1",
                "requestor": "tenant-1",
                "last_user_message": "cuando se pondran en contacto conmigo?",
                "conversation_state": ConversationState(
                    mode="discovery",
                    captured=CapturedContact(
                        contact_name="Omar",
                        phone="5544332211",
                        contact_schedule="de 9 a 6 entre semana",
                    ),
                    lead=LeadState(intent="purchase_intent", qualification="hot", status="calificado"),
                ).model_dump(),
            }
        )
    )
    response = str(result.get("final_response") or "").lower()
    assert "ya esta registrada" in response
    assert "de 9 a 6 entre semana" in response


def test_graph_inventory_flags_conflict_between_snapshot_and_kb() -> None:
    session_manager = _build_session_manager()
    knowledge_provider = InMemoryKnowledgeProvider()
    asyncio.run(
        knowledge_provider.upsert_topic(
            tenant_id="tenant-1",
            agent_id="default",
            topic="pricing",
            content="Plan enterprise desde 900 USD al mes.",
            source="unit_test",
            author="tester",
            metadata={},
        )
    )

    graph = build_mvp_orchestration_graph(
        session_manager=session_manager,
        knowledge_provider=knowledge_provider,
        tool_context_manager=ToolRequestContextManager(),
        capture_tool=None,
    )
    result = asyncio.run(
        graph.ainvoke(
            {
                "thread_id": "t-inventory-conflict-1",
                "requestor": "tenant-1",
                "last_user_message": "precio plan enterprise",
                "conversation_state": ConversationState(
                    pricing_context=PricingContext(
                        verified_summary="Plan starter desde 50 MXN al mes.",
                        source="snapshot",
                        query="precio plan enterprise",
                    )
                ).model_dump(),
            }
        )
    )

    response = str(result.get("final_response") or "")
    assert "diferencia" in response.lower()


def test_graph_supervisor_llm_cannot_break_capture_completion_when_contact_is_complete() -> None:
    async def _misrouting_supervisor(_: str, __: dict[str, object]):  # noqa: ANN001
        return {
            "mode": "discovery",
            "agent": "knowledge",
            "intent": "unknown",
            "qualification": "cold",
            "missing_fields": ["contact_name"],
            "next_action": "answer",
        }

    class _CaptureTool:
        async def execute(self, payload):  # noqa: ANN001
            assert payload["lead_data"]["contact_name"] == "Omar"
            return {
                "status": "captured",
                "lead_status": "calificado",
                "missing_critical_fields": [],
                "qualification_evidence": ["contact_name: Omar"],
                "crm_result": {"crm_id": "crm-2", "status": "upserted"},
                "structured_payload": {"lead_data": payload["lead_data"]},
            }

    session_manager = _build_session_manager()
    graph = build_mvp_orchestration_graph(
        session_manager=session_manager,
        knowledge_provider=InMemoryKnowledgeProvider(),
        tool_context_manager=ToolRequestContextManager(),
        capture_tool=_CaptureTool(),
        supervisor_override=_misrouting_supervisor,
    )
    result = asyncio.run(
        graph.ainvoke(
            {
                "thread_id": "t-supervisor-capture-priority-1",
                "requestor": "tenant-1",
                "last_user_message": "procedamos",
                "conversation_state": ConversationState(
                    mode="capture_completion",
                    captured=CapturedContact(
                        contact_name="Omar",
                        phone="5544332211",
                        contact_schedule="de 9 a 6",
                    ),
                    lead=LeadState(
                        intent="purchase_intent",
                        qualification="hot",
                        status="en_revision",
                    ),
                ).model_dump(),
            }
        )
    )
    assert "registro completado" in str(result.get("final_response") or "").lower()
    persisted = asyncio.run(session_manager.get_state("t-supervisor-capture-priority-1"))
    assert isinstance(persisted, dict)
    assert persisted["conversation_state"]["lead"]["status"] == "calificado"
