from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.agent.intent_router import IntentDecision


class IntentRoutingStructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: Literal["in_scope", "needs_clarification", "out_of_scope", "conversation_end"]
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1)


class StructuredIntentModel(Protocol):
    async def ainvoke(self, input: Any, config: Any | None = None) -> Any: ...


class LLMIntentRouter:
    def __init__(
        self,
        *,
        model: StructuredIntentModel,
        system_prompt: str,
        confidence_threshold: float = 0.7,
    ) -> None:
        self._model = model
        self._system_prompt = system_prompt
        self._confidence_threshold = confidence_threshold

    async def route(self, user_text: str) -> IntentDecision:
        try:
            raw = await self._model.ainvoke(
                {
                    "messages": [
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user", "content": user_text},
                    ]
                }
            )
            parsed = IntentRoutingStructuredOutput.model_validate(raw)
            if parsed.confidence < self._confidence_threshold:
                return IntentDecision(
                    route="needs_clarification",
                    confidence=parsed.confidence,
                    reason=f"llm_low_confidence:{parsed.reason}",
                )
            return IntentDecision(
                route=parsed.route,
                confidence=parsed.confidence,
                reason=f"llm:{parsed.reason}",
            )
        except Exception:
            return IntentDecision(
                route="needs_clarification",
                confidence=0.0,
                reason="llm_router_failed",
            )
