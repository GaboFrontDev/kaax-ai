from __future__ import annotations

import logging
from typing import Any

from app.agent.builder import build_agent_graph
from app.agent.middleware.prompt_sanitizer import PromptSanitizerMiddleware
from app.agent.prompt_loader import load_prompt
from app.agent.runtime import AgentRuntime, LangChainAgentRuntime
from app.agent.tools.context import ToolRequestContextManager
from app.crm.providers import CRMProvider
from app.knowledge.providers import KnowledgeProvider
from app.memory.attachments_store import AttachmentStore
from app.memory.session_manager import SessionManager

logger = logging.getLogger(__name__)


def build_agent(
    *,
    session_manager: SessionManager,
    attachment_store: AttachmentStore,
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
    runtime_backend: str = "langchain",
    runtime_strict: bool = False,
    model_name: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
    small_model_name: str = "anthropic.claude-3-haiku-20240307-v1:0",
    aws_region: str = "us-east-1",
    model_temperature: float = 0.5,
    prompt_name: str = "default",
    langchain_summarization_enabled: bool = True,
    checkpointer_manager: Any | None = None,
) -> AgentRuntime:
    sanitizer = PromptSanitizerMiddleware()
    system_prompt = load_prompt(prompt_name)

    if runtime_backend.lower() != "langchain":
        raise ValueError("Only LangChain runtime is supported. Set AGENT_RUNTIME_BACKEND=langchain.")

    def graph_factory(checkpointer: Any | None) -> Any:
        return build_agent_graph(
            model_name=model_name,
            small_model_name=small_model_name,
            aws_region=aws_region,
            temperature=model_temperature,
            system_prompt=system_prompt,
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
            enable_summarization=langchain_summarization_enabled,
            checkpointer=checkpointer,
        )

    try:
        return LangChainAgentRuntime(
            session_manager=session_manager,
            attachment_store=attachment_store,
            tool_context_manager=tool_context_manager,
            sanitizer=sanitizer,
            graph_factory=graph_factory,
            checkpointer_manager=checkpointer_manager,
        )
    except Exception as exc:  # pragma: no cover - optional path
        logger.exception("langchain_runtime_init_failed")
        message = "langchain runtime initialization failed"
        if runtime_strict:
            message = "langchain runtime requested but initialization failed"
        raise RuntimeError(message) from exc
