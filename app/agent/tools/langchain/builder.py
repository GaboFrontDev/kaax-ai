from __future__ import annotations

from typing import Any

from app.agent.tools.langchain.capture_lead_if_ready_tool import CaptureLeadIfReadyLangChainTool
from app.agent.tools.langchain.crm_upsert_quote_tool import CrmUpsertQuoteLangChainTool
from app.agent.tools.langchain.detect_lead_capture_readiness_tool import (
    DetectLeadCaptureReadinessLangChainTool,
)
from app.agent.tools.registry import ToolRegistry


def build_langchain_tools(tool_registry: ToolRegistry) -> list[Any]:
    return [
        CrmUpsertQuoteLangChainTool(tool_registry=tool_registry),
        DetectLeadCaptureReadinessLangChainTool(tool_registry=tool_registry),
        CaptureLeadIfReadyLangChainTool(tool_registry=tool_registry),
    ]
