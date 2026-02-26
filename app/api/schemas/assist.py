from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.agent.runtime import AssistRequest
from app.api.schemas.common import AttachmentPayload


class AgentAssistRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    user_text: str = Field(alias="userText", min_length=1)
    requestor: str = Field(min_length=1)
    stream_response: bool = Field(alias="streamResponse", default=False)
    session_id: str | None = Field(alias="sessionId", default=None)
    temperature: float | None = None
    formatter: str = "basic"
    prompt_name: str | None = Field(alias="promptName", default=None)
    tool_choice: str = Field(alias="toolChoice", default="auto")
    attachments: list[AttachmentPayload] = Field(default_factory=list)

    @property
    def thread_id(self) -> str:
        if self.session_id:
            return self.session_id
        return f"{self.requestor}:default"

    def to_runtime_request(self) -> AssistRequest:
        return AssistRequest(
            user_text=self.user_text,
            requestor=self.requestor,
            thread_id=self.thread_id,
            stream=self.stream_response,
            formatter=self.formatter,
            tool_choice=self.tool_choice,
            attachments=[attachment.model_dump() for attachment in self.attachments],
        )


class AgentAssistResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: str
    tools_used: list[str]
    completion_time: float
    conversation_id: str | None = None
    run_id: str | None = None
    attachments: list[AttachmentPayload] | None = None
