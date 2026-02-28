from __future__ import annotations

import logging
from typing import Any

from langchain.agents import create_agent
from langchain_aws import ChatBedrockConverse

from app.agent.memory_intent_graph import MemoryIntentState, build_memory_intent_tool
from app.agent.tools.capture_lead_if_ready_tool import CaptureLeadIfReadyTool
from app.agent.tools.context import ToolRequestContextManager
from app.agent.tools.crm_upsert_quote_tool import CrmUpsertQuoteTool
from app.agent.tools.detect_lead_capture_readiness_tool import DetectLeadCaptureReadinessTool
from app.agent.tools.knowledge_learn_tool import BedrockKnowledgeLearnDetector, KnowledgeLearnTool
from app.agent.tools.knowledge_search_tool import KnowledgeSearchTool
from app.crm.providers import CRMProvider
from app.knowledge.providers import KnowledgeProvider

logger = logging.getLogger(__name__)


def build_tools(
    *,
    crm_provider: CRMProvider,
    knowledge_provider: KnowledgeProvider,
    tool_context_manager: ToolRequestContextManager,
    owner_notify_enabled: bool = False,
    owner_whatsapp_number: str | None = None,
    owner_phone_number_id: str | None = None,
    whatsapp_meta_access_token: str | None = None,
    whatsapp_meta_api_version: str = "v21.0",
    knowledge_search_default_limit: int = 5,
    knowledge_learn_confidence_threshold: float = 0.75,
    knowledge_learn_detector_model_name: str = "anthropic.claude-3-haiku-20240307-v1:0",
    knowledge_learn_detector_region: str = "us-east-1",
    knowledge_learn_detector: Any | None = None,
    checkpointer: Any | None = None,
) -> list[Any]:
    crm_upsert_tool = CrmUpsertQuoteTool(crm_provider=crm_provider)
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
        knowledge_provider=knowledge_provider,
        get_context=tool_context_manager.get_context,
        default_limit=knowledge_search_default_limit,
    )
    knowledge_learn_tool = KnowledgeLearnTool(
        knowledge_provider=knowledge_provider,
        get_context=tool_context_manager.get_context,
        is_admin_requestor=tool_context_manager.is_admin_requestor,
        detector=detector,
        confidence_threshold=knowledge_learn_confidence_threshold,
    )

    async def _read_handler(state: MemoryIntentState) -> dict[str, Any]:
        return await knowledge_search_tool.execute(
            {
                "query": state.user_message,
                "limit": max(1, int(state.limit or knowledge_search_default_limit)),
            }
        )

    async def _update_handler(state: MemoryIntentState) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_text": state.user_message,
            "confirm": bool(state.confirm),
        }
        if state.topic_hint:
            payload["topic_hint"] = state.topic_hint
        return await knowledge_learn_tool.execute(payload)

    memory_intent_tool = build_memory_intent_tool(
        model_name=knowledge_learn_detector_model_name,
        aws_region=knowledge_learn_detector_region,
        read_handler=_read_handler,
        update_handler=_update_handler,
        checkpointer=checkpointer,
    )
    return [
        crm_upsert_tool,
        readiness_tool,
        capture_tool,
        memory_intent_tool,
        knowledge_search_tool,
        knowledge_learn_tool,
    ]


def build_agent_graph(
    *,
    model_name: str,
    small_model_name: str,
    aws_region: str,
    temperature: float,
    system_prompt: str,
    crm_provider: CRMProvider,
    knowledge_provider: KnowledgeProvider,
    tool_context_manager: ToolRequestContextManager,
    owner_notify_enabled: bool = False,
    owner_whatsapp_number: str | None = None,
    owner_phone_number_id: str | None = None,
    whatsapp_meta_access_token: str | None = None,
    whatsapp_meta_api_version: str = "v21.0",
    knowledge_search_default_limit: int = 5,
    knowledge_learn_confidence_threshold: float = 0.75,
    knowledge_learn_detector_model_name: str = "anthropic.claude-3-haiku-20240307-v1:0",
    knowledge_learn_detector_region: str = "us-east-1",
    knowledge_learn_detector: Any | None = None,
    enable_summarization: bool = True,
    checkpointer: Any | None = None,
    tools: list[Any] | None = None,
    middleware: list[Any] | None = None,
) -> Any:
    model = ChatBedrockConverse(
        model_id=model_name,
        region_name=aws_region,
        temperature=temperature,
        disable_streaming=False,
    )

    configured_tools = (
        tools
        if tools is not None
        else build_tools(
            crm_provider=crm_provider,
            knowledge_provider=knowledge_provider,
            tool_context_manager=tool_context_manager,
            owner_notify_enabled=owner_notify_enabled,
            owner_whatsapp_number=owner_whatsapp_number,
            owner_phone_number_id=owner_phone_number_id,
            whatsapp_meta_access_token=whatsapp_meta_access_token,
            whatsapp_meta_api_version=whatsapp_meta_api_version,
            knowledge_search_default_limit=knowledge_search_default_limit,
            knowledge_learn_confidence_threshold=knowledge_learn_confidence_threshold,
            knowledge_learn_detector_model_name=knowledge_learn_detector_model_name,
            knowledge_learn_detector_region=knowledge_learn_detector_region,
            knowledge_learn_detector=knowledge_learn_detector,
            checkpointer=checkpointer,
        )
    )
    configured_middleware = list(middleware or [])

    if enable_summarization:
        try:
            from langchain.agents.middleware import SummarizationMiddleware

            summary_model = ChatBedrockConverse(
                model_id=small_model_name,
                region_name=aws_region,
                temperature=0,
                disable_streaming=True,
            )
            configured_middleware.append(
                SummarizationMiddleware(
                    model=summary_model,
                    max_tokens_before_summary=150_000,
                    messages_to_keep=20,
                )
            )
        except Exception as exc:  # pragma: no cover - optional path
            logger.warning("langchain_summarization_middleware_unavailable: %s", exc)

    return create_agent(
        model=model,
        tools=configured_tools,
        checkpointer=checkpointer,
        system_prompt=system_prompt,
        middleware=configured_middleware,
    )
