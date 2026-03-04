from __future__ import annotations

from typing import Any, Awaitable, Callable

from langgraph.graph import END, START, StateGraph

from app.agent.orchestration.nodes import OrchestrationNodes
from app.agent.orchestration.schemas import OrchestrationState, RouterDecision
from app.agent.tools.capture_lead_if_ready_tool import CaptureLeadIfReadyTool
from app.agent.tools.context import ToolRequestContextManager
from app.agent.tools.memory_intent_router_tool import MemoryIntentRouterTool
from app.knowledge.providers import KnowledgeProvider
from app.memory.session_manager import SessionManager


def build_mvp_orchestration_graph(
    *,
    session_manager: SessionManager,
    knowledge_provider: KnowledgeProvider,
    tool_context_manager: ToolRequestContextManager,
    capture_tool: CaptureLeadIfReadyTool | None,
    subagent_runner: Any | None = None,
    knowledge_search_limit: int = 3,
    notify_owner: bool = False,
    supervisor_model_name: str = "anthropic.claude-3-haiku-20240307-v1:0",
    aws_region: str = "us-east-1",
    supervisor_temperature: float = 0,
    supervisor_override: Callable[[str, dict[str, Any]], Awaitable[RouterDecision | dict[str, Any] | None]]
    | None = None,
) -> Any:
    # Legacy args are kept in the signature for backward compatibility.
    _ = (supervisor_model_name, aws_region, supervisor_temperature, supervisor_override)

    memory_router = MemoryIntentRouterTool(
        knowledge_provider=knowledge_provider,
        get_context=tool_context_manager.get_context,
        default_limit=max(1, int(knowledge_search_limit or 1)),
    )
    nodes = OrchestrationNodes(
        session_manager=session_manager,
        memory_router=memory_router,
        capture_tool=capture_tool,
        notify_owner=notify_owner,
        subagent_runner=subagent_runner,
    )

    builder = StateGraph(OrchestrationState)
    builder.add_node("classify_turn", nodes.classify_turn)
    builder.add_node("repeat_handler", nodes.repeat_handler)
    builder.add_node("memory_lookup", nodes.memory_lookup)
    builder.add_node("discovery_value", nodes.discovery_value)
    builder.add_node("compose_reply", nodes.compose_reply)
    builder.add_node("capture_lead", nodes.capture_lead)
    builder.add_node("ask_missing_fields", nodes.ask_missing_fields)
    builder.add_node("demo_cta", nodes.demo_cta)
    builder.add_node("send_reply", nodes.send_reply)
    builder.add_node("persist_state", nodes.persist_state)

    builder.add_edge(START, "classify_turn")
    builder.add_conditional_edges(
        "classify_turn",
        nodes.classify_route,
        {
            "repeat_handler": "repeat_handler",
            "memory_lookup": "memory_lookup",
            "discovery_value": "discovery_value",
        },
    )
    builder.add_edge("repeat_handler", "compose_reply")
    builder.add_edge("memory_lookup", "compose_reply")
    builder.add_edge("discovery_value", "compose_reply")

    builder.add_conditional_edges(
        "compose_reply",
        nodes.lead_capture_gate,
        {
            "capture_lead": "capture_lead",
            "send_reply": "send_reply",
        },
    )

    builder.add_conditional_edges(
        "capture_lead",
        nodes.after_capture,
        {
            "ask_missing_fields": "ask_missing_fields",
            "demo_cta": "demo_cta",
            "send_reply": "send_reply",
        },
    )
    builder.add_edge("ask_missing_fields", "send_reply")
    builder.add_edge("demo_cta", "send_reply")
    builder.add_edge("send_reply", "persist_state")
    builder.add_edge("persist_state", END)

    return builder.compile()
