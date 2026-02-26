from __future__ import annotations

from app.agent.runtime import AssistRequest, StreamingEvent


class WhatsAppMetaAdapter:
    async def normalize_inbound(self, raw: dict[str, object]) -> AssistRequest:
        from_number = str(raw.get("from", "unknown"))
        text = str(raw.get("text", ""))
        message_id = str(raw.get("wa_message_id", "msg"))
        return AssistRequest(
            user_text=text,
            requestor=f"wa-meta:{from_number}",
            thread_id=f"wa-meta:{from_number}:{message_id}",
            stream=False,
        )

    async def denormalize_outbound(self, event: StreamingEvent) -> dict[str, object]:
        return event.model_dump(exclude_none=True)
