from __future__ import annotations

from app.agent.runtime import AssistRequest, StreamingEvent


class ChainlitAdapter:
    async def normalize_inbound(self, raw: dict[str, object]) -> AssistRequest:
        return AssistRequest(
            user_text=str(raw.get("message", "")),
            requestor=str(raw.get("user", "chainlit:user")),
            thread_id=str(raw.get("thread_id", "chainlit:thread")),
            stream=bool(raw.get("stream", True)),
        )

    async def denormalize_outbound(self, event: StreamingEvent) -> dict[str, object]:
        return event.model_dump(exclude_none=True)
