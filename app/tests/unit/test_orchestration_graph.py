from __future__ import annotations

import asyncio

from app.agent.orchestration.graph import build_mvp_orchestration_graph
from app.agent.orchestration.schemas import ConversationState, QAItem
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


def test_graph_greeting_starts_sales_discovery() -> None:
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
                "messages": [],
                "last_user_message": "hola",
                "conversation_state": ConversationState().model_dump(),
            }
        )
    )

    response = str(result.get("final_response") or "").lower()
    assert "proceso comercial" in response
    assert "telefono" not in response
    assert "nombre" not in response


def test_graph_uses_sales_dialogue_subagent_when_runner_is_provided() -> None:
    class _StubRunner:
        async def run(self, *, agent_name, user_message, context):  # noqa: ANN001
            assert agent_name == "sales_dialogue"
            assert "draft_response" in context
            assert isinstance(user_message, str)
            return "Claro. Kaax AI puede ayudarte a automatizar esto con enfoque comercial. ¿Te muestro un flujo en demo?"

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
                "thread_id": "t-llm-polish-1",
                "requestor": "tenant-1",
                "messages": [],
                "last_user_message": "hola",
                "conversation_state": ConversationState().model_dump(),
            }
        )
    )

    assert "automatizar" in str(result.get("final_response") or "").lower()


def test_graph_factual_question_uses_memory_router_read() -> None:
    knowledge_provider = InMemoryKnowledgeProvider()
    asyncio.run(
        knowledge_provider.upsert_topic(
            tenant_id="tenant-1",
            agent_id="default",
            topic="integraciones",
            content="Integramos con HubSpot, Salesforce y CRMs con API REST.",
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
                "thread_id": "t-factual-1",
                "requestor": "tenant-1",
                "messages": [],
                "last_user_message": "Que integraciones CRM tienen?",
                "conversation_state": ConversationState().model_dump(),
            }
        )
    )

    response = str(result.get("final_response") or "").lower()
    tools_used = result.get("tools_used") or []
    assert "confirmada" in response
    assert "hubspot" in response
    assert "memory_intent_router" in tools_used


def test_graph_services_question_returns_concrete_offer_when_kb_is_empty() -> None:
    graph = build_mvp_orchestration_graph(
        session_manager=_build_session_manager(),
        knowledge_provider=InMemoryKnowledgeProvider(),
        tool_context_manager=ToolRequestContextManager(),
        capture_tool=None,
    )

    result = asyncio.run(
        graph.ainvoke(
            {
                "thread_id": "t-services-1",
                "requestor": "tenant-1",
                "messages": [],
                "last_user_message": "que servicios tienen?",
                "conversation_state": ConversationState().model_dump(),
            }
        )
    )

    response = str(result.get("final_response") or "").lower()
    assert "whatsapp" in response
    assert "crm" in response
    assert "calific" in response


def test_graph_repeat_question_returns_short_recap() -> None:
    state = ConversationState()
    state.qa_memory.answered_questions.append(
        QAItem(
            normalized_question="que integraciones crm tienen",
            answer_summary="Integramos con CRMs principales.",
        )
    )

    graph = build_mvp_orchestration_graph(
        session_manager=_build_session_manager(),
        knowledge_provider=InMemoryKnowledgeProvider(),
        tool_context_manager=ToolRequestContextManager(),
        capture_tool=None,
    )

    result = asyncio.run(
        graph.ainvoke(
            {
                "thread_id": "t-repeat-1",
                "requestor": "tenant-1",
                "messages": [],
                "last_user_message": "Que integraciones CRM tienen?",
                "conversation_state": state.model_dump(),
            }
        )
    )

    response = str(result.get("final_response") or "").lower()
    assert "te resumo" in response
    assert "demo" in response


def test_graph_capture_missing_fields_asks_only_missing() -> None:
    class _CaptureTool:
        async def execute(self, payload):  # noqa: ANN001
            assert payload["lead_data"]["phone"] == "5544332211"
            return {
                "status": "missing_fields",
                "lead_status": "en_revision",
                "missing_critical_fields": ["lead_data.contact_name", "lead_data.contact_schedule"],
                "qualification_evidence": ["phone: 5544332211"],
                "structured_payload": {},
            }

    state = ConversationState()
    state.business_context.use_case = "captacion de leads"
    state.business_context.pain_points = ["seguimiento inconsistente"]
    state.lead_data.phone = "5544332211"

    graph = build_mvp_orchestration_graph(
        session_manager=_build_session_manager(),
        knowledge_provider=InMemoryKnowledgeProvider(),
        tool_context_manager=ToolRequestContextManager(),
        capture_tool=_CaptureTool(),
    )

    result = asyncio.run(
        graph.ainvoke(
            {
                "thread_id": "t-capture-missing-1",
                "requestor": "tenant-1",
                "messages": [],
                "last_user_message": "quiero agendar demo",
                "conversation_state": state.model_dump(),
            }
        )
    )

    response = str(result.get("final_response") or "").lower()
    assert "nombre de contacto" in response
    assert "horario preferido" in response
    assert "telefono" not in response


def test_graph_capture_success_returns_demo_cta_and_persists() -> None:
    class _CaptureTool:
        async def execute(self, payload):  # noqa: ANN001
            return {
                "status": "captured",
                "lead_status": "calificado",
                "missing_critical_fields": [],
                "qualification_evidence": ["ready"],
                "crm_result": {"crm_id": "crm-123", "status": "upserted"},
                "structured_payload": {"lead_data": payload["lead_data"]},
            }

    session_manager = _build_session_manager()
    state = ConversationState()
    state.business_context.use_case = "captacion de leads"
    state.business_context.pain_points = ["tiempo de respuesta"]
    state.lead_data.contact_name = "Ana"
    state.lead_data.phone = "+5215550000001"
    state.lead_data.contact_schedule = "lunes 10am"

    graph = build_mvp_orchestration_graph(
        session_manager=session_manager,
        knowledge_provider=InMemoryKnowledgeProvider(),
        tool_context_manager=ToolRequestContextManager(),
        capture_tool=_CaptureTool(),
    )

    result = asyncio.run(
        graph.ainvoke(
            {
                "thread_id": "t-capture-success-1",
                "requestor": "tenant-1",
                "messages": [],
                "last_user_message": "quiero demo",
                "conversation_state": state.model_dump(),
            }
        )
    )

    response = str(result.get("final_response") or "").lower()
    assert "agendamos la demo" in response

    persisted = asyncio.run(session_manager.get_state("t-capture-success-1"))
    assert isinstance(persisted, dict)
    assert len(persisted.get("messages") or []) == 2
    tool_result = persisted.get("tool_result") or {}
    assert tool_result.get("status") == "captured"


def test_graph_memory_update_instruction_routes_update_mode() -> None:
    graph = build_mvp_orchestration_graph(
        session_manager=_build_session_manager(),
        knowledge_provider=InMemoryKnowledgeProvider(),
        tool_context_manager=ToolRequestContextManager(),
        capture_tool=None,
    )

    result = asyncio.run(
        graph.ainvoke(
            {
                "thread_id": "t-update-1",
                "requestor": "tenant-1",
                "messages": [],
                "last_user_message": "Recuerda que soporte atiende de lunes a viernes de 9 a 18.",
                "conversation_state": ConversationState().model_dump(),
            }
        )
    )

    response = str(result.get("final_response") or "").lower()
    assert "actualicé" in response
    assert "memoria" in response


def test_graph_persists_messages_across_turns() -> None:
    session_manager = _build_session_manager()
    graph = build_mvp_orchestration_graph(
        session_manager=session_manager,
        knowledge_provider=InMemoryKnowledgeProvider(),
        tool_context_manager=ToolRequestContextManager(),
        capture_tool=None,
    )

    first = asyncio.run(
        graph.ainvoke(
            {
                "thread_id": "t-msg-1",
                "requestor": "tenant-1",
                "messages": [],
                "last_user_message": "hola",
                "conversation_state": ConversationState().model_dump(),
            }
        )
    )
    second = asyncio.run(
        graph.ainvoke(
            {
                "thread_id": "t-msg-1",
                "requestor": "tenant-1",
                "messages": first.get("messages") or [],
                "last_user_message": "quiero automatizar seguimiento",
                "conversation_state": first["conversation_state"].model_dump(),
            }
        )
    )

    persisted = asyncio.run(session_manager.get_state("t-msg-1"))
    assert isinstance(persisted, dict)
    assert len(persisted.get("messages") or []) >= 4
    assert isinstance(second.get("final_response"), str)
