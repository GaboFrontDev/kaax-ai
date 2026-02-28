from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, ConfigDict, Field

from app.agent.tools.registry import ToolRegistry


class CrmUpsertQuoteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    payload: dict[str, Any] = Field(default_factory=dict)


class CrmUpsertQuoteLangChainTool(BaseTool):
    name: str = "crm_upsert_quote"
    description: str = "Store quote/lead data in kaax internal CRM registry using a structured payload."
    args_schema: Optional[ArgsSchema] = CrmUpsertQuoteArgs
    return_direct: bool = False
    tool_registry: ToolRegistry
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def _arun(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = await self.tool_registry.execute(self.name, {"payload": payload})
        return result.output

    def _run(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("CrmUpsertQuoteLangChainTool only supports async execution.")
