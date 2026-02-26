from __future__ import annotations

import logging
from typing import Any

from app.agent.langchain_runtime import LangChainAgentRuntime
from app.agent.middleware.prompt_sanitizer import PromptSanitizerMiddleware
from app.agent.middleware.summarization import SummarizationMiddleware
from app.agent.prompt_loader import load_prompt
from app.agent.runtime import AgentRuntime, DefaultAgentRuntime
from app.agent.tools.registry import ToolRegistry
from app.memory.attachments_store import AttachmentStore
from app.memory.session_manager import SessionManager

logger = logging.getLogger(__name__)


def build_agent(
    *,
    session_manager: SessionManager,
    attachment_store: AttachmentStore,
    tool_registry: ToolRegistry,
    tool_retry_attempts: int = 2,
    tool_retry_backoff_ms: int = 200,
    runtime_backend: str = "stub",
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

    if runtime_backend.lower() == "langchain":
        try:
            return LangChainAgentRuntime(
                session_manager=session_manager,
                attachment_store=attachment_store,
                tool_registry=tool_registry,
                sanitizer=sanitizer,
                model_name=model_name,
                small_model_name=small_model_name,
                aws_region=aws_region,
                temperature=model_temperature,
                system_prompt=load_prompt(prompt_name),
                enable_summarization=langchain_summarization_enabled,
                checkpointer_manager=checkpointer_manager,
            )
        except Exception as exc:  # pragma: no cover - optional path
            logger.exception("langchain_runtime_init_failed")
            if runtime_strict:
                raise RuntimeError("langchain runtime requested but initialization failed") from exc
            logger.warning("falling_back_to_stub_runtime")

    summarizer = SummarizationMiddleware()
    return DefaultAgentRuntime(
        session_manager=session_manager,
        attachment_store=attachment_store,
        tool_registry=tool_registry,
        sanitizer=sanitizer,
        summarizer=summarizer,
        tool_retry_attempts=tool_retry_attempts,
        tool_retry_backoff_ms=tool_retry_backoff_ms,
    )
