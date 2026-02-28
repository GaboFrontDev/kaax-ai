from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, ConfigDict, Field

from app.agent.tools.registry import ToolRegistry


class KnowledgeLearnArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_text: str = Field(min_length=1, max_length=3000)
    confirm: bool = False
    topic_hint: str | None = Field(default=None, min_length=1)


class KnowledgeLearnLangChainTool(BaseTool):
    name: str = "knowledge_learn"
    description: str = (
        "Detect and learn new business knowledge from user instructions. "
        "Only use when the user is explicitly teaching business/service information."
    )
    args_schema: Optional[ArgsSchema] = KnowledgeLearnArgs
    return_direct: bool = False
    tool_registry: ToolRegistry
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def _arun(self, source_text: str, confirm: bool = False, topic_hint: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"source_text": source_text, "confirm": bool(confirm)}
        if topic_hint:
            payload["topic_hint"] = topic_hint
        result = await self.tool_registry.execute(self.name, payload)
        return result.output

    def _run(self, source_text: str, confirm: bool = False, topic_hint: str | None = None) -> dict[str, Any]:
        raise NotImplementedError("KnowledgeLearnLangChainTool only supports async execution.")

