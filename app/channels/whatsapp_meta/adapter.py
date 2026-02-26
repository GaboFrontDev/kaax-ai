from __future__ import annotations

from app.agent.runtime import AssistRequest, StreamingEvent


class WhatsAppMetaAdapter:
    async def normalize_inbound(self, raw: dict[str, object]) -> AssistRequest:
        from_number = str(raw.get("from", "unknown"))
        to_number = str(raw.get("to", "unknown"))
        text = str(raw.get("text", ""))
        return AssistRequest(
            user_text=text,
            requestor=f"wa-meta:{from_number}",
            # Keep stable memory context for the same user and business number.
            thread_id=f"wa-meta:{to_number}:{from_number}",
            stream=False,
        )

    async def denormalize_outbound(self, event: StreamingEvent) -> dict[str, object]:
        return event.model_dump(exclude_none=True)
