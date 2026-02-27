import asyncio
from typing import Any

from app.agent.intent_router_llm import LLMIntentRouter


class _FakeStructuredModel:
    def __init__(self, output: Any, *, raises: bool = False) -> None:
        self._output = output
        self._raises = raises

    async def ainvoke(self, input: Any, config: Any | None = None) -> Any:
        if self._raises:
            raise RuntimeError("boom")
        return self._output


def test_llm_intent_router_accepts_high_confidence_route() -> None:
    router = LLMIntentRouter(
        model=_FakeStructuredModel(
            {"route": "out_of_scope", "confidence": 0.91, "reason": "request_is_coding_tutorial"}
        ),
        system_prompt="router",
        confidence_threshold=0.7,
    )

    decision = asyncio.run(router.route("quiero programar en react"))

    assert decision.route == "out_of_scope"
    assert decision.confidence == 0.91
    assert decision.reason.startswith("llm:")


def test_llm_intent_router_accepts_conversation_end_route() -> None:
    router = LLMIntentRouter(
        model=_FakeStructuredModel(
            {"route": "conversation_end", "confidence": 0.95, "reason": "user_said_goodbye"}
        ),
        system_prompt="router",
        confidence_threshold=0.7,
    )

    decision = asyncio.run(router.route("gracias, adios"))

    assert decision.route == "conversation_end"
    assert decision.confidence == 0.95


def test_llm_intent_router_forces_clarification_when_confidence_is_low() -> None:
    router = LLMIntentRouter(
        model=_FakeStructuredModel({"route": "in_scope", "confidence": 0.45, "reason": "ambiguous"}),
        system_prompt="router",
        confidence_threshold=0.7,
    )

    decision = asyncio.run(router.route("hola"))

    assert decision.route == "needs_clarification"
    assert decision.reason.startswith("llm_low_confidence:")


def test_llm_intent_router_falls_back_to_keyword_router_on_failure() -> None:
    router = LLMIntentRouter(
        model=_FakeStructuredModel({}, raises=True),
        system_prompt="router",
        confidence_threshold=0.7,
    )

    decision = asyncio.run(router.route("soy de seattle"))

    assert decision.route == "needs_clarification"
    assert decision.reason == "llm_router_failed"
