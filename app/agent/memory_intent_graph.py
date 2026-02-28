from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from app.agent.prompt_loader import load_prompt

logger = logging.getLogger(__name__)


class MemoryIntentInput(BaseModel):
    user_message: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)
    confirm: bool = False
    topic_hint: str | None = Field(default=None, min_length=1)


class MemoryIntentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Literal["read", "update"]
    confidence: float = Field(default=0.5, ge=0, le=1)
    reason: str = Field(min_length=1)


class MemoryIntentState(BaseModel):
    user_message: str = ""
    limit: int = 5
    confirm: bool = False
    topic_hint: str | None = None
    decision: MemoryIntentDecision | None = None
    read_result: dict[str, Any] = Field(default_factory=dict)
    update_result: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)


def build_memory_intent_graph(
    *,
    model_name: str,
    aws_region: str,
    classifier: Callable[[str], Awaitable[MemoryIntentDecision]] | None = None,
    read_handler: Callable[[MemoryIntentState], Awaitable[dict[str, Any]]] | None = None,
    update_handler: Callable[[MemoryIntentState], Awaitable[dict[str, Any]]] | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """Build a semantic graph that routes each message as memory read/update."""

    prompt = _load_prompt()
    model_lock = asyncio.Lock()
    structured_model: Any | None = None

    async def get_structured_model() -> Any:
        nonlocal structured_model
        if structured_model is not None:
            return structured_model
        async with model_lock:
            if structured_model is not None:
                return structured_model
            from langchain_aws import ChatBedrockConverse

            llm = ChatBedrockConverse(
                model_id=model_name,
                region_name=aws_region,
                temperature=0,
                disable_streaming=True,
            )
            structured_model = llm.with_structured_output(MemoryIntentDecision)
            return structured_model

    async def classify_memory_intent(state: MemoryIntentState) -> dict[str, Any]:
        message = state.user_message.strip()
        if not message:
            return {
                "decision": MemoryIntentDecision(
                    operation="read",
                    confidence=1.0,
                    reason="Mensaje vacio; se clasifica como read.",
                )
            }

        try:
            if classifier is not None:
                decision = await classifier(message)
            else:
                model = await get_structured_model()
                raw_decision = await model.ainvoke(
                    [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"Mensaje del usuario:\n{message}"},
                    ]
                )
                decision = _normalize_decision(raw_decision)
            if not isinstance(decision, MemoryIntentDecision):
                decision = _normalize_decision(decision)
            return {"decision": decision}
        except Exception as exc:  # pragma: no cover - provider/network dependent
            logger.warning("memory_intent_classification_failed: %s", exc)
            return {
                "decision": MemoryIntentDecision(
                    operation="read",
                    confidence=0.0,
                    reason=f"Fallo clasificador ({type(exc).__name__}); fallback a read.",
                )
            }

    async def run_read_memory(state: MemoryIntentState) -> dict[str, Any]:
        if read_handler is None:
            return {"read_result": {}}
        return {"read_result": await read_handler(state)}

    async def run_update_memory(state: MemoryIntentState) -> dict[str, Any]:
        if update_handler is None:
            return {"update_result": {}}
        return {"update_result": await update_handler(state)}

    def finalize_memory_result(state: MemoryIntentState) -> dict[str, Any]:
        decision = state.decision or MemoryIntentDecision(
            operation="read",
            confidence=0.0,
            reason="No hubo decision valida; fallback a read.",
        )
        payload = state.read_result if decision.operation == "read" else state.update_result
        return {
            "result": {
                "operation": decision.operation,
                "confidence": decision.confidence,
                "reason": decision.reason,
                "payload": payload,
            }
        }

    def continue_by_intent(state: MemoryIntentState) -> Literal["run_read_memory", "run_update_memory"]:
        decision = state.decision
        if decision is not None and decision.operation == "update":
            return "run_update_memory"
        return "run_read_memory"

    builder = StateGraph(MemoryIntentState)
    builder.add_node("classify_memory_intent", classify_memory_intent)
    builder.add_node("run_read_memory", run_read_memory)
    builder.add_node("run_update_memory", run_update_memory)
    builder.add_node("finalize_memory_result", finalize_memory_result)

    builder.add_edge(START, "classify_memory_intent")
    builder.add_conditional_edges(
        "classify_memory_intent",
        continue_by_intent,
        ["run_read_memory", "run_update_memory"],
    )
    builder.add_edge("run_read_memory", "finalize_memory_result")
    builder.add_edge("run_update_memory", "finalize_memory_result")
    builder.add_edge("finalize_memory_result", END)

    graph = builder.compile(checkpointer=checkpointer)
    return graph.with_config(run_name="Memory Intent Router")


def build_memory_intent_tool(
    *,
    model_name: str,
    aws_region: str,
    classifier: Callable[[str], Awaitable[MemoryIntentDecision]] | None = None,
    read_handler: Callable[[MemoryIntentState], Awaitable[dict[str, Any]]] | None = None,
    update_handler: Callable[[MemoryIntentState], Awaitable[dict[str, Any]]] | None = None,
    checkpointer: Any | None = None,
) -> Any:
    graph = build_memory_intent_graph(
        model_name=model_name,
        aws_region=aws_region,
        classifier=classifier,
        read_handler=read_handler,
        update_handler=update_handler,
        checkpointer=checkpointer,
    )
    return graph.as_tool(
        name="memory_intent_router",
        description=(
            "Semantic memory router for conversational workflows. "
            "Use this tool when you must decide if a user message should READ existing memory or UPDATE memory with new knowledge. "
            "Decision rules are semantic (LLM-based), not keyword-only. "
            "Input: user_message (required), plus optional limit for read operations, and optional confirm/topic_hint for update operations. "
            "If routed to read, it executes memory retrieval behavior (knowledge_search path). "
            "If routed to update, it executes memory learning behavior (knowledge_learn path), including confirmation context when provided. "
            "Output always includes: operation ('read' or 'update'), confidence (0..1), reason, and payload (result from the executed path). "
            "When a user makes a question, ask yourself: is the user likely trying to retrieve/consult existing knowledge (read), or are they trying to teach/update knowledge (update)?. "
            "Use this tool before memory actions to keep routing explicit, auditable, and consistent."
        ),
        args_schema=MemoryIntentInput,
    )


class MemoryIntentGraph:
    """Wrapper used by runtime to classify each inbound message with the graph."""

    def __init__(
        self,
        *,
        model_name: str,
        aws_region: str,
        classifier: Callable[[str], Awaitable[MemoryIntentDecision]] | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        self._graph = build_memory_intent_graph(
            model_name=model_name,
            aws_region=aws_region,
            classifier=classifier,
            checkpointer=checkpointer,
        )

    async def classify(self, message: str) -> MemoryIntentDecision:
        result = await self._graph.ainvoke({"user_message": str(message or "").strip()})
        if isinstance(result, dict):
            decision = result.get("decision")
            if isinstance(decision, MemoryIntentDecision):
                return decision
            if decision is not None:
                return _normalize_decision(decision)
        return MemoryIntentDecision(
            operation="read",
            confidence=0.0,
            reason="No se pudo clasificar; fallback a read.",
        )


def _load_prompt() -> str:
    try:
        return load_prompt("memory_intent_router")
    except Exception:
        return (
            "Clasifica cada mensaje como 'read' o 'update'. "
            "Usa 'update' solo cuando el usuario esta ensenando/actualizando memoria. "
            "Para preguntas normales, consultas o recuperacion de datos, usa 'read'."
        )


def _normalize_decision(raw: Any) -> MemoryIntentDecision:
    if isinstance(raw, MemoryIntentDecision):
        return raw

    if isinstance(raw, dict):
        operation_value = str(raw.get("operation", "read")).strip().lower()
        operation: Literal["read", "update"] = "update" if operation_value == "update" else "read"

        confidence_raw = raw.get("confidence")
        if isinstance(confidence_raw, (int, float)):
            confidence = float(min(1.0, max(0.0, confidence_raw)))
        else:
            confidence = 0.5

        reason_raw = raw.get("reason")
        reason = str(reason_raw).strip() if isinstance(reason_raw, str) and reason_raw.strip() else (
            "Clasificacion semantica sin razon detallada."
        )
        return MemoryIntentDecision(
            operation=operation,
            confidence=confidence,
            reason=reason,
        )

    return MemoryIntentDecision(
        operation="read",
        confidence=0.0,
        reason="Respuesta invalida del clasificador; fallback a read.",
    )
