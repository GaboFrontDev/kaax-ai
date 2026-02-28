from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, ConfigDict, Field


class CrmUpsertQuoteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    payload: dict[str, Any] = Field(default_factory=dict)


class CrmUpsertQuoteTool(BaseTool):
    name: str = "crm_upsert_quote"
    description: str = "Store quote/lead data in kaax internal CRM registry using a structured payload."
    args_schema: Optional[ArgsSchema] = CrmUpsertQuoteArgs
    return_direct: bool = False
    crm_provider: Any
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        quote = dict(payload["payload"])
        return await self.crm_provider.upsert_quote(quote)

    async def _arun(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self.execute({"payload": payload})

    def _run(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("CrmUpsertQuoteTool only supports async execution.")
