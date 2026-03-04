from __future__ import annotations

from typing import Any, Literal

from app.agent.orchestration.routing_rules import derive_router_and_state
from app.agent.orchestration.schemas import OrchestrationState


async def classify_turn(state: OrchestrationState) -> dict[str, Any]:
    router, updated_conversation_state = derive_router_and_state(
        user_message=state.last_user_message,
        conversation_state=state.conversation_state,
    )
    return {
        "router": router,
        "conversation_state": updated_conversation_state,
        "tool_result": None,
    }


def classify_route(state: OrchestrationState) -> Literal["repeat_handler", "memory_lookup", "discovery_value"]:
    if state.router is None:
        return "discovery_value"
    return state.router.route
