from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Callable, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.agent.event_mapper import LangChainStreamEventMapper
from app.agent.middleware.prompt_sanitizer import PromptSanitizerMiddleware
from app.agent.result_parser import dedupe_tools, extract_response_text, extract_tools_used
from app.agent.tools.context import ToolRequestContextManager
from app.memory.attachments_store import AttachmentStore
from app.memory.langgraph_checkpointer import LangGraphCheckpointerManager
from app.memory.session_manager import SessionBusyError, SessionManager
from app.observability.logging import set_correlation

logger = logging.getLogger(__name__)


class AssistRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_text: str
    requestor: str
    thread_id: str
    stream: bool = False
    formatter: str = "basic"
    tool_choice: str = "auto"
    attachments: list[dict[str, Any]] = Field(default_factory=list)


class StreamingEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    content: str | None = None
    tool: str | None = None
    payload: dict[str, Any] | None = None
    thread_id: str
    run_id: str | None = None


class AgentRuntime(Protocol):
    async def invoke(self, req: AssistRequest) -> dict[str, Any]: ...

    async def stream(self, req: AssistRequest) -> AsyncIterator[StreamingEvent]: ...


class LangChainAgentRuntime:
    def __init__(
        self,
        *,
        session_manager: SessionManager,
        attachment_store: AttachmentStore,
        tool_context_manager: ToolRequestContextManager,
        sanitizer: PromptSanitizerMiddleware,
        graph_factory: Callable[[Any | None], Any],
        checkpointer_manager: LangGraphCheckpointerManager | None,
    ) -> None:
        self._session_manager = session_manager
        self._attachment_store = attachment_store
        self._tool_context_manager = tool_context_manager
        self._sanitizer = sanitizer
        self._graph_factory = graph_factory
        self._checkpointer_manager = checkpointer_manager

        self._graph: Any | None = None
        self._graph_lock = asyncio.Lock()

    async def _get_graph(self) -> Any:
        if self._graph is not None:
            return self._graph

        async with self._graph_lock:
            if self._graph is not None:
                return self._graph

            checkpointer = None
            if self._checkpointer_manager is not None:
                checkpointer = await self._checkpointer_manager.get_checkpointer()

            self._graph = self._graph_factory(checkpointer)
            return self._graph

    async def invoke(self, req: AssistRequest) -> dict[str, Any]:
        started_at = time.perf_counter()
        run_id = str(uuid4())
        set_correlation(thread_id=req.thread_id, run_id=run_id)

        try:
            async with self._session_manager.session_lock(req.thread_id):
                async with self._tool_context_manager.request_context(
                    thread_id=req.thread_id,
                    requestor=req.requestor,
                ):
                    if req.attachments:
                        await self._attachment_store.put(req.thread_id, req.attachments)

                    user_text = self._sanitizer.sanitize(req.user_text)
                    logger.info("langchain_runtime_inbound thread_id=%s text=%s", req.thread_id, user_text[:300])

                    graph = await self._get_graph()

                    result = await graph.ainvoke(
                        {"messages": [{"role": "user", "content": user_text}]},
                        config={"configurable": {"thread_id": req.thread_id}},
                    )
                    answer = extract_response_text(result)
                    if not answer:
                        answer = "No se generó respuesta del modelo."

                    attachments = await self._attachment_store.get_recent(req.thread_id, limit=20)

            return {
                "response": answer,
                "tools_used": dedupe_tools(extract_tools_used(result)),
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
                async with self._tool_context_manager.request_context(
                    thread_id=req.thread_id,
                    requestor=req.requestor,
                ):
                    if req.attachments:
                        await self._attachment_store.put(req.thread_id, req.attachments)

                    user_text = self._sanitizer.sanitize(req.user_text)
                    logger.info("langchain_runtime_inbound_stream thread_id=%s text=%s", req.thread_id, user_text[:300])

                    graph = await self._get_graph()

                    mapper = LangChainStreamEventMapper(thread_id=req.thread_id, run_id=run_id)

                    async for event in graph.astream_events(
                        {"messages": [{"role": "user", "content": user_text}]},
                        config={"configurable": {"thread_id": req.thread_id}},
                        version="v2",
                    ):
                        for mapped in mapper.map_event(event):
                            yield StreamingEvent(**mapped)

                    answer = ""
                    if mapper.final_output is not None:
                        answer = extract_response_text(mapper.final_output)
                    if not answer:
                        answer = "No se generó respuesta del modelo."

                    attachments = await self._attachment_store.get_recent(req.thread_id, limit=20)
                    yield StreamingEvent(
                        type="complete",
                        payload={
                            "tools_used": dedupe_tools(mapper.tools_used),
                            "attachments": attachments,
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
