from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    run_id: str = Field(alias="runId", min_length=1)
    score: int = Field(ge=0, le=1)
    comment: str | None = None
    conversation_id: str | None = Field(alias="conversationId", default=None)
    source: str | None = None


class FeedbackResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "ok"
