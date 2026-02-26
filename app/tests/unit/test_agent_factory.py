from __future__ import annotations

import pytest

import app.agent.factory as factory_module
from app.agent.factory import build_agent
from app.agent.runtime import DefaultAgentRuntime
from app.agent.tools.registry import ToolRegistry
from app.memory.attachments_store import InMemoryAttachmentStore
from app.memory.checkpoint_store import InMemoryCheckpointStore
from app.memory.locks import InMemorySessionLockManager
from app.memory.session_manager import SessionManager


def _build_deps() -> tuple[SessionManager, InMemoryAttachmentStore, ToolRegistry]:
    session_manager = SessionManager(
        checkpoint_store=InMemoryCheckpointStore(),
        lock_manager=InMemorySessionLockManager(),
        session_timeout_seconds=1800,
    )
    attachments = InMemoryAttachmentStore(max_items=20, ttl_minutes=120)
    tools = ToolRegistry()
    return session_manager, attachments, tools


def test_build_agent_falls_back_to_stub_when_langchain_init_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    session_manager, attachments, tools = _build_deps()
    monkeypatch.setattr(
        factory_module,
        "LangChainAgentRuntime",
        lambda **_: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    runtime = build_agent(
        session_manager=session_manager,
        attachment_store=attachments,
        tool_registry=tools,
        runtime_backend="langchain",
        runtime_strict=False,
    )

    assert isinstance(runtime, DefaultAgentRuntime)


def test_build_agent_langchain_strict_raises_when_init_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    session_manager, attachments, tools = _build_deps()
    monkeypatch.setattr(
        factory_module,
        "LangChainAgentRuntime",
        lambda **_: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError):
        build_agent(
            session_manager=session_manager,
            attachment_store=attachments,
            tool_registry=tools,
            runtime_backend="langchain",
            runtime_strict=True,
        )
