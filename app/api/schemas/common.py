from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AttachmentPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1)
    content: str = ""
    type: str = "application/octet-stream"
    created_at: str | None = None
