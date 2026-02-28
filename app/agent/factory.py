from __future__ import annotations

from typing import Any

from app.agent.middleware.prompt_sanitizer import PromptSanitizerMiddleware
from app.agent.orchestration_runtime import LangGraphMvpRuntime
from app.agent.runtime import AgentRuntime
from app.agent.tools.context import ToolRequestContextManager
from app.crm.providers import CRMProvider
from app.knowledge.providers import KnowledgeProvider
from app.memory.attachments_store import AttachmentStore
from app.memory.session_manager import SessionManager


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
    runtime_backend: str = "langgraph_mvp",
    runtime_strict: bool = False,
    model_name: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
    small_model_name: str = "anthropic.claude-3-haiku-20240307-v1:0",
    aws_region: str = "us-east-1",
    model_temperature: float = 0.5,
    prompt_name: str = "default",
    langchain_summarization_enabled: bool = True,
    memory_intent_enabled: bool = True,
    memory_intent_model_name: str | None = None,
    checkpointer_manager: Any | None = None,
) -> AgentRuntime:
    # Legacy arguments remain in signature for API compatibility.

    backend = runtime_backend.strip().lower()
    if backend != "langgraph_mvp":
        raise ValueError(
            "Unsupported runtime backend. AGENT_RUNTIME_BACKEND must be 'langgraph_mvp'."
        )

    sanitizer = PromptSanitizerMiddleware()
    return LangGraphMvpRuntime(
        session_manager=session_manager,
        attachment_store=attachment_store,
        tool_context_manager=tool_context_manager,
        sanitizer=sanitizer,
        crm_provider=crm_provider,
        knowledge_provider=knowledge_provider,
        owner_notify_enabled=owner_notify_enabled,
        owner_whatsapp_number=owner_whatsapp_number,
        owner_phone_number_id=owner_phone_number_id,
        whatsapp_meta_access_token=whatsapp_meta_access_token,
        whatsapp_meta_api_version=whatsapp_meta_api_version,
        knowledge_search_default_limit=knowledge_search_default_limit,
        model_name=model_name,
        supervisor_model_name=small_model_name,
        aws_region=aws_region,
        model_temperature=model_temperature,
    )
