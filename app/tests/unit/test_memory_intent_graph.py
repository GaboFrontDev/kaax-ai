from __future__ import annotations

import asyncio

from app.agent.memory_intent_graph import (
    MemoryIntentDecision,
    MemoryIntentGraph,
    build_memory_intent_graph,
)


def test_memory_intent_graph_classifies_update_with_stub() -> None:
    async def classifier(_: str) -> MemoryIntentDecision:
        return MemoryIntentDecision(
            operation="update",
            confidence=0.92,
            reason="El usuario esta ensenando conocimiento nuevo.",
        )

    graph = MemoryIntentGraph(
        model_name="stub-model",
        aws_region="us-east-1",
        classifier=classifier,
    )
    decision = asyncio.run(graph.classify("aprende que nuestro horario es de 9 a 18"))
    assert decision.operation == "update"
    assert decision.confidence == 0.92


def test_memory_intent_graph_falls_back_to_read_on_error() -> None:
    async def classifier(_: str) -> MemoryIntentDecision:
        raise RuntimeError("boom")

    graph = MemoryIntentGraph(
        model_name="stub-model",
        aws_region="us-east-1",
        classifier=classifier,
    )
    decision = asyncio.run(graph.classify("hola"))
    assert decision.operation == "read"
    assert decision.confidence == 0.0


def test_memory_intent_graph_routes_and_executes_handlers() -> None:
    async def read_classifier(_: str) -> MemoryIntentDecision:
        return MemoryIntentDecision(operation="read", confidence=0.88, reason="consulta")

    async def update_classifier(_: str) -> MemoryIntentDecision:
        return MemoryIntentDecision(operation="update", confidence=0.91, reason="aprendizaje")

    async def read_handler(state) -> dict[str, object]:
        return {"path": "read", "query": state.user_message, "limit": state.limit}

    async def update_handler(state) -> dict[str, object]:
        return {
            "path": "update",
            "source_text": state.user_message,
            "confirm": state.confirm,
            "topic_hint": state.topic_hint,
        }

    read_graph = build_memory_intent_graph(
        model_name="stub-model",
        aws_region="us-east-1",
        classifier=read_classifier,
        read_handler=read_handler,
        update_handler=update_handler,
    )
    read_result = asyncio.run(read_graph.ainvoke({"user_message": "cual es el horario", "limit": 7}))
    assert read_result["result"]["operation"] == "read"
    assert read_result["result"]["payload"]["path"] == "read"
    assert read_result["result"]["payload"]["limit"] == 7

    update_graph = build_memory_intent_graph(
        model_name="stub-model",
        aws_region="us-east-1",
        classifier=update_classifier,
        read_handler=read_handler,
        update_handler=update_handler,
    )
    update_result = asyncio.run(
        update_graph.ainvoke(
            {"user_message": "aprende esto", "confirm": True, "topic_hint": "horario"}
        )
    )
    assert update_result["result"]["operation"] == "update"
    assert update_result["result"]["payload"]["path"] == "update"
    assert update_result["result"]["payload"]["confirm"] is True


def test_memory_intent_graph_accepts_missing_confidence_from_classifier_dict() -> None:
    async def classifier(_: str):
        return {"operation": "update", "reason": "El usuario esta ensenando memoria."}

    graph = MemoryIntentGraph(
        model_name="stub-model",
        aws_region="us-east-1",
        classifier=classifier,
    )
    decision = asyncio.run(graph.classify("Aprende esto: horario de soporte..."))
    assert decision.operation == "update"
    assert decision.confidence == 0.5
