from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from app.agent.runtime import StreamingEvent


def encode_sse(payload: dict[str, Any], *, event: str = "message") -> str:
    data = json.dumps(payload, ensure_ascii=True)
    return f"event: {event}\ndata: {data}\n\n"


async def with_heartbeat(
    events: AsyncIterator[StreamingEvent],
    *,
    heartbeat_seconds: float = 15.0,
) -> AsyncIterator[str]:
    iterator = events.__aiter__()
    while True:
        try:
            event = await asyncio.wait_for(iterator.__anext__(), timeout=heartbeat_seconds)
            yield encode_sse(event.model_dump(exclude_none=True))
        except asyncio.TimeoutError:
            yield encode_sse({"type": "heartbeat"}, event="heartbeat")
        except StopAsyncIteration:
            break
