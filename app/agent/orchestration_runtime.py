from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator
from uuid import uuid4

from app.agent.middleware.prompt_sanitizer import PromptSanitizerMiddleware
from app.agent.orchestration.graph import build_mvp_orchestration_graph
from app.agent.orchestration.routing_rules import normalize_conversation_state
from app.agent.orchestration.schemas import OrchestrationState
from app.agent.orchestration.subagents import LangChainSubagentRunner
from app.agent.result_parser import dedupe_tools
from app.agent.runtime import AssistRequest, StreamingEvent
from app.agent.tools.capture_lead_if_ready_tool import CaptureLeadIfReadyTool
from app.agent.tools.context import ToolRequestContextManager
from app.agent.tools.crm_upsert_quote_tool import CrmUpsertQuoteTool
from app.agent.tools.detect_lead_capture_readiness_tool import DetectLeadCaptureReadinessTool
from app.crm.providers import CRMProvider
from app.knowledge.providers import KnowledgeProvider
from app.memory.attachments_store import AttachmentStore
from app.memory.session_manager import SessionBusyError, SessionManager
from app.observability.logging import set_correlation

logger = logging.getLogger(__name__)


class LangGraphMvpRuntime:
    def __init__(
        self,
        *,
        session_manager: SessionManager,
        attachment_store: AttachmentStore,
        tool_context_manager: ToolRequestContextManager,
        sanitizer: PromptSanitizerMiddleware,
        crm_provider: CRMProvider,
        knowledge_provider: KnowledgeProvider,
        owner_notify_enabled: bool = False,
        owner_whatsapp_number: str | None = None,
        owner_phone_number_id: str | None = None,
        whatsapp_meta_access_token: str | None = None,
        whatsapp_meta_api_version: str = "v21.0",
        knowledge_search_default_limit: int = 3,
        model_name: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        supervisor_model_name: str | None = None,
        aws_region: str = "us-east-1",
        model_temperature: float = 0.2,
    ) -> None:
        self._session_manager = session_manager
        self._attachment_store = attachment_store
        self._tool_context_manager = tool_context_manager
        self._sanitizer = sanitizer

        capture_tool = CaptureLeadIfReadyTool(
            crm_upsert_tool=CrmUpsertQuoteTool(crm_provider=crm_provider),
            readiness_tool=DetectLeadCaptureReadinessTool(),
            owner_notify_enabled=owner_notify_enabled,
            owner_whatsapp_number=owner_whatsapp_number,
            owner_phone_number_id=owner_phone_number_id,
            whatsapp_meta_access_token=whatsapp_meta_access_token,
            whatsapp_meta_api_version=whatsapp_meta_api_version,
        )
        subagent_runner = LangChainSubagentRunner(
            model_name=model_name,
            aws_region=aws_region,
            temperature=model_temperature,
        )

        self._graph = build_mvp_orchestration_graph(
            session_manager=session_manager,
            knowledge_provider=knowledge_provider,
            tool_context_manager=tool_context_manager,
            capture_tool=capture_tool,
            subagent_runner=subagent_runner,
            knowledge_search_limit=max(1, int(knowledge_search_default_limit)),
            notify_owner=owner_notify_enabled,
            supervisor_model_name=supervisor_model_name or model_name,
            aws_region=aws_region,
            supervisor_temperature=0,
        )

    async def invoke(self, req: AssistRequest) -> dict[str, Any]:
        started_at = time.perf_counter()
        run_id = str(uuid4())
        set_correlation(thread_id=req.thread_id, run_id=run_id)

        try:
            async with self._session_manager.session_lock(req.thread_id):
                response, tools_used, attachments = await self._run_turn(req)
            return {
                "response": response,
                "tools_used": tools_used,
                "memory_intent": None,
                "completion_time": round(time.perf_counter() - started_at, 3),
                "conversation_id": req.thread_id,
                "run_id": run_id,
                "attachments": attachments,
            }
        finally:
            set_correlation(None, None)

    async def stream(self, req: AssistRequest) -> AsyncIterator[StreamingEvent]:
        run_id = str(uuid4())
        set_correlation(thread_id=req.thread_id, run_id=run_id)

        try:
            async with self._session_manager.session_lock(req.thread_id):
                response, tools_used, attachments = await self._run_turn(req)
                yield StreamingEvent(
                    type="message",
                    content=response,
                    thread_id=req.thread_id,
                    run_id=run_id,
                )
                yield StreamingEvent(
                    type="complete",
                    payload={
                        "tools_used": tools_used,
                        "attachments": attachments,
                        "memory_intent": None,
                    },
                    thread_id=req.thread_id,
                    run_id=run_id,
                )
        except SessionBusyError as exc:
            yield StreamingEvent(
                type="error",
                content=str(exc),
                thread_id=req.thread_id,
                run_id=run_id,
            )
        finally:
            set_correlation(None, None)

    async def _run_turn(self, req: AssistRequest) -> tuple[str, list[str], list[dict[str, Any]]]:
        user_text = self._sanitizer.sanitize(req.user_text)
        logger.info("langgraph_mvp_inbound thread_id=%s text=%s", req.thread_id, user_text[:300])

        async with self._tool_context_manager.request_context(
            thread_id=req.thread_id,
            requestor=req.requestor,
        ):
            if req.attachments:
                await self._attachment_store.put(req.thread_id, req.attachments)

            previous_state = await self._session_manager.get_state(req.thread_id)
            raw_conversation_state = (
                previous_state.get("conversation_state")
                if isinstance(previous_state, dict)
                else None
            )
            conversation_state = normalize_conversation_state(raw_conversation_state)

            result = await self._graph.ainvoke(
                {
                    "thread_id": req.thread_id,
                    "requestor": req.requestor,
                    "last_user_message": user_text,
                    "conversation_state": conversation_state.model_dump(),
                }
            )
            state = self._coerce_state(result, thread_id=req.thread_id, requestor=req.requestor)
            response = (state.final_response or "").strip() or "No se genero respuesta."
            tools_used = dedupe_tools(list(state.tools_used))
            attachments = await self._attachment_store.get_recent(req.thread_id, limit=20)
            return response, tools_used, attachments

    @staticmethod
    def _coerce_state(result: Any, *, thread_id: str, requestor: str) -> OrchestrationState:
        if isinstance(result, OrchestrationState):
            return result
        if isinstance(result, dict):
            try:
                return OrchestrationState.model_validate(result)
            except Exception:
                pass
        return OrchestrationState(
            thread_id=thread_id,
            requestor=requestor,
            final_response="No se genero respuesta.",
        )
