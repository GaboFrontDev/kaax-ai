from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, ConfigDict, Field

from app.agent.tools.registry import ToolRegistry


class DetectLeadCaptureReadinessArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    business_context: dict[str, Any] = Field(default_factory=dict)
    whatsapp_context: dict[str, Any] = Field(default_factory=dict)
    crm_context: dict[str, Any] = Field(default_factory=dict)
    agent_limits: dict[str, Any] = Field(default_factory=dict)
    lead_data: dict[str, Any] = Field(default_factory=dict)


class DetectLeadCaptureReadinessLangChainTool(BaseTool):
    name: str = "detect_lead_capture_readiness"
    description: str = "Detect if a lead has enough context and evidence to be registered in kaax internal CRM."
    args_schema: Optional[ArgsSchema] = DetectLeadCaptureReadinessArgs
    return_direct: bool = False
    tool_registry: ToolRegistry
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def _arun(
        self,
        business_context: dict[str, Any] | None = None,
        whatsapp_context: dict[str, Any] | None = None,
        crm_context: dict[str, Any] | None = None,
        agent_limits: dict[str, Any] | None = None,
        lead_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if business_context is not None:
            payload["business_context"] = business_context
        if whatsapp_context is not None:
            payload["whatsapp_context"] = whatsapp_context
        if crm_context is not None:
            payload["crm_context"] = crm_context
        if agent_limits is not None:
            payload["agent_limits"] = agent_limits
        if lead_data is not None:
            payload["lead_data"] = lead_data
        result = await self.tool_registry.execute(self.name, payload)
        return result.output

    def _run(
        self,
        business_context: dict[str, Any] | None = None,
        whatsapp_context: dict[str, Any] | None = None,
        crm_context: dict[str, Any] | None = None,
        agent_limits: dict[str, Any] | None = None,
        lead_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError("DetectLeadCaptureReadinessLangChainTool only supports async execution.")
