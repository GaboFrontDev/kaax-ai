import asyncio

from app.agent.middleware.prompt_sanitizer import PromptSanitizerMiddleware
from app.agent.middleware.summarization import SummarizationMiddleware
from app.agent.runtime import AssistRequest, DefaultAgentRuntime
from app.agent.tools.registry import ToolRegistry
from app.memory.attachments_store import InMemoryAttachmentStore
from app.memory.checkpoint_store import InMemoryCheckpointStore
from app.memory.locks import InMemorySessionLockManager
from app.memory.session_manager import SessionManager


def _build_runtime() -> DefaultAgentRuntime:
    session_manager = SessionManager(
        checkpoint_store=InMemoryCheckpointStore(),
        lock_manager=InMemorySessionLockManager(),
        session_timeout_seconds=1800,
    )
    attachments = InMemoryAttachmentStore(max_items=20, ttl_minutes=120)
    tools = ToolRegistry()
    sanitizer = PromptSanitizerMiddleware()
    summarizer = SummarizationMiddleware()
    return DefaultAgentRuntime(
        session_manager=session_manager,
        attachment_store=attachments,
        tool_registry=tools,
        sanitizer=sanitizer,
        summarizer=summarizer,
    )


def test_default_runtime_out_of_scope_request_is_guardrailed() -> None:
    runtime = _build_runtime()
    req = AssistRequest(
        user_text="quiero programar un componente en react",
        requestor="test",
        thread_id="thread-1",
    )

    result = asyncio.run(runtime.invoke(req))

    assert result["tools_used"] == []
    assert "fuera de ese alcance" in result["response"].lower()


def test_default_runtime_ambiguous_request_requests_clarification() -> None:
    runtime = _build_runtime()
    req = AssistRequest(
        user_text="soy de seattle",
        requestor="test",
        thread_id="thread-2",
    )

    result = asyncio.run(runtime.invoke(req))

    assert result["tools_used"] == []
    assert "automatizar ventas" in result["response"].lower()


def test_default_runtime_conversation_end_returns_closure_message() -> None:
    runtime = _build_runtime()
    req = AssistRequest(
        user_text="gracias, eso es todo",
        requestor="test",
        thread_id="thread-3",
    )

    result = asyncio.run(runtime.invoke(req))

    assert result["tools_used"] == []
    assert "cerramos por ahora" in result["response"].lower()
