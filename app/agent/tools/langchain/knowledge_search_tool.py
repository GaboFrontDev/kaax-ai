from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, ConfigDict, Field

from app.agent.tools.registry import ToolRegistry


class KnowledgeSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)


class KnowledgeSearchLangChainTool(BaseTool):
    name: str = "knowledge_search"
    description: str = (
        "Search business knowledge learned for the current tenant and agent. "
        "Use this for FAQ/product/service questions before drafting final answers."
    )
    args_schema: Optional[ArgsSchema] = KnowledgeSearchArgs
    return_direct: bool = False
    tool_registry: ToolRegistry
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def _arun(self, query: str, limit: int = 5) -> dict[str, Any]:
        result = await self.tool_registry.execute(self.name, {"query": query, "limit": limit})
        return result.output

    def _run(self, query: str, limit: int = 5) -> dict[str, Any]:
        raise NotImplementedError("KnowledgeSearchLangChainTool only supports async execution.")

