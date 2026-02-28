from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Protocol

from app.agent.tools.capture_lead_if_ready_tool import CaptureLeadIfReadyTool
from app.agent.tools.crm_upsert_quote_tool import CrmUpsertQuoteTool
from app.agent.tools.detect_lead_capture_readiness_tool import DetectLeadCaptureReadinessTool
from app.agent.tools.validator import ToolValidationError, validate_tool_input, validate_tool_output
from app.crm.providers import CRMProvider, InMemoryCRMProvider

logger = logging.getLogger(__name__)


class ToolExecutor(Protocol):
    name: str

    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass(slots=True)
class ToolExecutionResult:
    tool: str
    input: dict[str, Any]
    output: dict[str, Any]


class ToolRegistry:
    def __init__(
        self,
        crm_provider: CRMProvider | None = None,
        *,
        owner_notify_enabled: bool = False,
        owner_whatsapp_number: str | None = None,
        owner_phone_number_id: str | None = None,
        whatsapp_meta_access_token: str | None = None,
        whatsapp_meta_api_version: str = "v21.0",
    ) -> None:
        self._crm_provider: CRMProvider = crm_provider or InMemoryCRMProvider()

        crm_upsert_tool = CrmUpsertQuoteTool(self._crm_provider)
        readiness_tool = DetectLeadCaptureReadinessTool()
        capture_tool = CaptureLeadIfReadyTool(
            crm_upsert_tool=crm_upsert_tool,
            readiness_tool=readiness_tool,
            owner_notify_enabled=owner_notify_enabled,
            owner_whatsapp_number=owner_whatsapp_number,
            owner_phone_number_id=owner_phone_number_id,
            whatsapp_meta_access_token=whatsapp_meta_access_token,
            whatsapp_meta_api_version=whatsapp_meta_api_version,
        )

        self._tools: dict[str, ToolExecutor] = {
            crm_upsert_tool.name: crm_upsert_tool,
            readiness_tool.name: readiness_tool,
            capture_tool.name: capture_tool,
        }

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        return tuple(self._tools.keys())

    async def execute(self, tool_name: str, payload: dict[str, Any]) -> ToolExecutionResult:
        logger.info("tool_execute_start tool=%s", tool_name)
        try:
            validated = validate_tool_input(tool_name, payload)
        except ToolValidationError as exc:
            output = validate_tool_output(tool_name, {"error": str(exc)})
            logger.warning("tool_execute_validation_error tool=%s error=%s", tool_name, str(exc))
            return ToolExecutionResult(tool=tool_name, input={}, output=output)

        executor = self._tools.get(tool_name)
        if executor is None:
            output = validate_tool_output(tool_name, {"error": f"unsupported tool {tool_name}"})
            return ToolExecutionResult(tool=tool_name, input=validated, output=output)

        try:
            output = await executor.execute(validated)
        except Exception as exc:
            output = {"error": f"tool execution failed: {type(exc).__name__}: {exc}"}

        normalized_output = validate_tool_output(tool_name, output)
        logger.info("tool_execute_done tool=%s status=%s", tool_name, normalized_output.get("status"))
        return ToolExecutionResult(tool=tool_name, input=validated, output=normalized_output)
