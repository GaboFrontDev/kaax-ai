from __future__ import annotations

from contextlib import asynccontextmanager
import contextvars
from dataclasses import dataclass
from fnmatch import fnmatchcase
import logging
from typing import Any, Protocol

from app.agent.tools.capture_lead_if_ready_tool import CaptureLeadIfReadyTool
from app.agent.tools.crm_upsert_quote_tool import CrmUpsertQuoteTool
from app.agent.tools.detect_lead_capture_readiness_tool import DetectLeadCaptureReadinessTool
from app.agent.tools.knowledge_learn_tool import (
    BedrockKnowledgeLearnDetector,
    KnowledgeLearnDetector,
    KnowledgeLearnTool,
)
from app.agent.tools.knowledge_search_tool import KnowledgeRequestContext, KnowledgeSearchTool
from app.agent.tools.validator import ToolValidationError, validate_tool_input, validate_tool_output
from app.crm.providers import CRMProvider, InMemoryCRMProvider
from app.knowledge.providers import InMemoryKnowledgeProvider, KnowledgeProvider

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
        knowledge_provider: KnowledgeProvider | None = None,
        *,
        owner_notify_enabled: bool = False,
        owner_whatsapp_number: str | None = None,
        owner_phone_number_id: str | None = None,
        whatsapp_meta_access_token: str | None = None,
        whatsapp_meta_api_version: str = "v21.0",
        knowledge_admin_requestors: set[str] | None = None,
        knowledge_search_default_limit: int = 5,
        knowledge_learn_confidence_threshold: float = 0.75,
        knowledge_learn_detector_model_name: str = "anthropic.claude-3-haiku-20240307-v1:0",
        knowledge_learn_detector_region: str = "us-east-1",
        knowledge_learn_detector: KnowledgeLearnDetector | None = None,
        agent_id: str = "default",
    ) -> None:
        self._crm_provider: CRMProvider = crm_provider or InMemoryCRMProvider()
        self._knowledge_provider: KnowledgeProvider = knowledge_provider or InMemoryKnowledgeProvider()
        self._knowledge_admin_requestors = set(knowledge_admin_requestors or set())
        self._agent_id = agent_id or "default"
        self._request_context: contextvars.ContextVar[KnowledgeRequestContext | None] = contextvars.ContextVar(
            "tool_registry_request_context",
            default=None,
        )

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
        detector = knowledge_learn_detector or BedrockKnowledgeLearnDetector(
            model_name=knowledge_learn_detector_model_name,
            aws_region=knowledge_learn_detector_region,
        )
        knowledge_search_tool = KnowledgeSearchTool(
            knowledge_provider=self._knowledge_provider,
            get_context=self._current_request_context,
            default_limit=knowledge_search_default_limit,
        )
        knowledge_learn_tool = KnowledgeLearnTool(
            knowledge_provider=self._knowledge_provider,
            get_context=self._current_request_context,
            is_admin_requestor=self._is_admin_requestor,
            detector=detector,
            confidence_threshold=knowledge_learn_confidence_threshold,
        )

        self._tools: dict[str, ToolExecutor] = {
            crm_upsert_tool.name: crm_upsert_tool,
            readiness_tool.name: readiness_tool,
            capture_tool.name: capture_tool,
            knowledge_search_tool.name: knowledge_search_tool,
            knowledge_learn_tool.name: knowledge_learn_tool,
        }

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        return tuple(self._tools.keys())

    @asynccontextmanager
    async def request_context(
        self,
        *,
        thread_id: str,
        requestor: str,
        agent_id: str | None = None,
    ):
        tenant_id = self._derive_tenant_id(requestor)
        context = KnowledgeRequestContext(
            thread_id=thread_id,
            requestor=requestor,
            tenant_id=tenant_id,
            agent_id=(agent_id or self._agent_id).strip() or self._agent_id,
        )
        token = self._request_context.set(context)
        try:
            yield context
        finally:
            self._request_context.reset(token)

    def _current_request_context(self) -> KnowledgeRequestContext | None:
        return self._request_context.get()

    def _derive_tenant_id(self, requestor: str) -> str:
        raw = (requestor or "").strip()
        return raw or "anonymous"

    def _is_admin_requestor(self, requestor: str) -> bool:
        if not self._knowledge_admin_requestors:
            return False
        normalized = (requestor or "").strip()
        for allowed in self._knowledge_admin_requestors:
            if fnmatchcase(normalized, allowed):
                return True
        return False

    async def execute(self, tool_name: str, payload: dict[str, Any]) -> ToolExecutionResult:
        context = self._current_request_context()
        logger.info(
            "tool_execute_start tool=%s thread_id=%s requestor=%s",
            tool_name,
            context.thread_id if context else "n/a",
            context.requestor if context else "n/a",
        )
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
        logger.info(
            "tool_execute_done tool=%s status=%s thread_id=%s",
            tool_name,
            normalized_output.get("status"),
            context.thread_id if context else "n/a",
        )
        return ToolExecutionResult(tool=tool_name, input=validated, output=normalized_output)
