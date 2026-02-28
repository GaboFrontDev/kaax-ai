from __future__ import annotations

import pytest

import app.agent.factory as factory_module
from app.agent.factory import build_agent
from app.agent.tools.context import ToolRequestContextManager
from app.crm.providers import InMemoryCRMProvider
from app.knowledge.providers import InMemoryKnowledgeProvider
from app.memory.attachments_store import InMemoryAttachmentStore
from app.memory.checkpoint_store import InMemoryCheckpointStore
from app.memory.locks import InMemorySessionLockManager
from app.memory.session_manager import SessionManager


def _build_deps() -> tuple[
    SessionManager,
    InMemoryAttachmentStore,
    InMemoryCRMProvider,
    InMemoryKnowledgeProvider,
    ToolRequestContextManager,
]:
    session_manager = SessionManager(
        checkpoint_store=InMemoryCheckpointStore(),
        lock_manager=InMemorySessionLockManager(),
        session_timeout_seconds=1800,
    )
    attachments = InMemoryAttachmentStore(max_items=20, ttl_minutes=120)
    crm_provider = InMemoryCRMProvider()
    knowledge_provider = InMemoryKnowledgeProvider()
    context_manager = ToolRequestContextManager()
    return session_manager, attachments, crm_provider, knowledge_provider, context_manager


def test_build_agent_rejects_legacy_backend() -> None:
    session_manager, attachments, crm_provider, knowledge_provider, context_manager = _build_deps()

    with pytest.raises(ValueError):
        build_agent(
            session_manager=session_manager,
            attachment_store=attachments,
            crm_provider=crm_provider,
            knowledge_provider=knowledge_provider,
            tool_context_manager=context_manager,
            runtime_backend="langchain",
        )


def test_build_agent_uses_langgraph_mvp_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    session_manager, attachments, crm_provider, knowledge_provider, context_manager = _build_deps()
    sentinel_runtime = object()
    monkeypatch.setattr(factory_module, "LangGraphMvpRuntime", lambda **_: sentinel_runtime)

    runtime = build_agent(
        session_manager=session_manager,
        attachment_store=attachments,
        crm_provider=crm_provider,
        knowledge_provider=knowledge_provider,
        tool_context_manager=context_manager,
        runtime_backend="langgraph_mvp",
    )

    assert runtime is sentinel_runtime


def test_build_agent_default_backend_is_langgraph_mvp(monkeypatch: pytest.MonkeyPatch) -> None:
    session_manager, attachments, crm_provider, knowledge_provider, context_manager = _build_deps()
    sentinel_runtime = object()
    monkeypatch.setattr(factory_module, "LangGraphMvpRuntime", lambda **_: sentinel_runtime)

    runtime = build_agent(
        session_manager=session_manager,
        attachment_store=attachments,
        crm_provider=crm_provider,
        knowledge_provider=knowledge_provider,
        tool_context_manager=context_manager,
    )

    assert runtime is sentinel_runtime
