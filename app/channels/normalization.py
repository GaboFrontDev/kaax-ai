from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class NormalizedInbound(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel: str
    thread_id: str
    user_text: str
    requestor: str
