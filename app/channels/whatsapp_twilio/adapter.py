from __future__ import annotations

from app.agent.runtime import AssistRequest, StreamingEvent


class WhatsAppTwilioAdapter:
    async def normalize_inbound(self, raw: dict[str, object]) -> AssistRequest:
        from_number = str(raw.get("From", "unknown"))
        to_number = str(raw.get("To", "unknown"))
        text = str(raw.get("Body", ""))
        return AssistRequest(
            user_text=text,
            requestor=f"wa-twilio:{from_number}",
            # Keep a stable thread per WhatsApp conversation pair.
            thread_id=f"wa-twilio:{to_number}:{from_number}",
            stream=False,
        )

    async def denormalize_outbound(self, event: StreamingEvent) -> dict[str, object]:
        return event.model_dump(exclude_none=True)
