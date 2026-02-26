from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StreamingPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    content: str | None = None
    tool: str | None = None
    payload: dict[str, object] | None = None
    thread_id: str
    run_id: str | None = None
