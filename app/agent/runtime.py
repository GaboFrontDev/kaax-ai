from __future__ import annotations

from typing import Any, AsyncIterator, Protocol

from pydantic import BaseModel, ConfigDict, Field


class AssistRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_text: str
    requestor: str
    thread_id: str
    stream: bool = False
    formatter: str = "basic"
    tool_choice: str = "auto"
    attachments: list[dict[str, Any]] = Field(default_factory=list)


class StreamingEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    content: str | None = None
    tool: str | None = None
    payload: dict[str, Any] | None = None
    thread_id: str
    run_id: str | None = None


class AgentRuntime(Protocol):
    async def invoke(self, req: AssistRequest) -> dict[str, Any]: ...

    async def stream(self, req: AssistRequest) -> AsyncIterator[StreamingEvent]: ...
