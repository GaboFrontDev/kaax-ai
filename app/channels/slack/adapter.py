from __future__ import annotations

from app.agent.runtime import AssistRequest, StreamingEvent


class SlackAdapter:
    async def normalize_inbound(self, raw: dict[str, object]) -> AssistRequest:
        event = dict(raw.get("event", {}))
        channel = str(event.get("channel", "slack"))
        user = str(event.get("user", "unknown"))
        thread_ts = str(event.get("thread_ts") or event.get("ts") or "new-thread")
        text = str(event.get("text", ""))
        return AssistRequest(
            user_text=text,
            requestor=f"slack:{user}",
            thread_id=f"slack:{channel}:{thread_ts}",
            stream=False,
        )

    async def denormalize_outbound(self, event: StreamingEvent) -> dict[str, object]:
        return event.model_dump(exclude_none=True)
