from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator
from uuid import uuid4

from app.agent.middleware.prompt_sanitizer import PromptSanitizerMiddleware
from app.agent.runtime import AssistRequest, StreamingEvent, _content_to_text
from app.agent.tools.registry import ToolRegistry
from app.memory.attachments_store import AttachmentStore
from app.memory.langgraph_checkpointer import LangGraphCheckpointerManager
from app.memory.session_manager import SessionBusyError, SessionManager
from app.observability.logging import set_correlation

logger = logging.getLogger(__name__)


def _extract_response_text(result: Any) -> str:
    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        messages = result.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                role: str | None = None
                content: Any = None

                if isinstance(message, dict):
                    role = str(message.get("role") or message.get("type") or "")
                    content = message.get("content")
                else:
                    role = str(getattr(message, "type", ""))
                    content = getattr(message, "content", None)

                if role in {"ai", "assistant"}:
                    text = _content_to_text(content)
                    if text:
                        return text

        for key in ("output", "response", "answer"):
            if key in result:
                text = _content_to_text(result.get(key))
                if text:
                    return text

        return json.dumps(result, ensure_ascii=True)

    return _content_to_text(result)


def build_langchain_tools(tool_registry: ToolRegistry) -> list[Any]:
    from langchain_core.tools import tool

    @tool
    async def get_iso_country_code(country_name: str) -> dict[str, Any]:
        """Return ISO alpha-2 country code for a country name."""

        return (await tool_registry.execute("get_iso_country_code", {"country_name": country_name})).output

    @tool
    async def retrieve_markets(query: str, country_code: str | None = None, limit: int = 10) -> dict[str, Any]:
        """Retrieve candidate markets for a query and optional country code."""

        payload: dict[str, Any] = {"query": query, "limit": limit}
        if country_code:
            payload["country_code"] = country_code
        return (await tool_registry.execute("retrieve_markets", payload)).output

    @tool
    async def retrieve_segments(
        query: str,
        country_code: str | None = None,
        taxonomy: str = "default",
        limit: int = 10,
    ) -> dict[str, Any]:
        """Retrieve audience segments for a query."""

        payload: dict[str, Any] = {
            "query": query,
            "taxonomy": taxonomy,
            "limit": limit,
        }
        if country_code:
            payload["country_code"] = country_code
        return (await tool_registry.execute("retrieve_segments", payload)).output

    @tool
    async def retrieve_formats(
        market_name: str,
        format_query: str,
        country_code: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Retrieve available media formats for a market."""

        payload: dict[str, Any] = {
            "market_name": market_name,
            "format_query": format_query,
            "limit": limit,
        }
        if country_code:
            payload["country_code"] = country_code
        return (await tool_registry.execute("retrieve_formats", payload)).output

    @tool
    async def find_units(
        segment_ids: list[str],
        markets: list[str],
        media_formats: list[str] | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Find ranked inventory units for selected segments and markets."""

        payload: dict[str, Any] = {
            "segment_ids": segment_ids,
            "markets": markets,
            "limit": limit,
        }
        if media_formats is not None:
            payload["media_formats"] = media_formats
        return (await tool_registry.execute("find_units", payload)).output

    @tool
    async def update_user_preferences(email: str, preferences: dict[str, str]) -> dict[str, Any]:
        """Persist user preferences by email."""

        return (
            await tool_registry.execute(
                "update_user_preferences",
                {
                    "email": email,
                    "preferences": preferences,
                },
            )
        ).output

    @tool
    async def crm_upsert_quote(payload: dict[str, Any]) -> dict[str, Any]:
        """Upsert quote data in CRM using a structured payload."""

        return (await tool_registry.execute("crm_upsert_quote", {"payload": payload})).output

    return [
        get_iso_country_code,
        retrieve_markets,
        retrieve_segments,
        retrieve_formats,
        find_units,
        update_user_preferences,
        crm_upsert_quote,
    ]


class LangChainAgentRuntime:
    def __init__(
        self,
        *,
        session_manager: SessionManager,
        attachment_store: AttachmentStore,
        tool_registry: ToolRegistry,
        sanitizer: PromptSanitizerMiddleware,
        model_name: str,
        small_model_name: str,
        aws_region: str,
        temperature: float,
        system_prompt: str,
        enable_summarization: bool,
        checkpointer_manager: LangGraphCheckpointerManager | None,
    ) -> None:
        self._session_manager = session_manager
        self._attachment_store = attachment_store
        self._tool_registry = tool_registry
        self._sanitizer = sanitizer
        self._checkpointer_manager = checkpointer_manager

        self._model_name = model_name
        self._small_model_name = small_model_name
        self._aws_region = aws_region
        self._temperature = temperature
        self._system_prompt = system_prompt
        self._enable_summarization = enable_summarization

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

            self._graph = self._build_graph(checkpointer=checkpointer)
            return self._graph

    def _build_graph(self, *, checkpointer: Any | None) -> Any:
        from langchain.agents import create_agent
        from langchain_aws import ChatBedrockConverse

        model = ChatBedrockConverse(
            model_id=self._model_name,
            region_name=self._aws_region,
            temperature=self._temperature,
            disable_streaming=False,
        )
        tools = build_langchain_tools(self._tool_registry)
        middleware: list[Any] = []

        if self._enable_summarization:
            try:
                from langchain.agents.middleware import SummarizationMiddleware

                summary_model = ChatBedrockConverse(
                    model_id=self._small_model_name,
                    region_name=self._aws_region,
                    temperature=0,
                    disable_streaming=True,
                )
                middleware.append(
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
            tools=tools,
            checkpointer=checkpointer,
            system_prompt=self._system_prompt,
            middleware=middleware,
        )

    async def invoke(self, req: AssistRequest) -> dict[str, Any]:
        started_at = time.perf_counter()
        run_id = str(uuid4())
        set_correlation(thread_id=req.thread_id, run_id=run_id)

        try:
            async with self._session_manager.session_lock(req.thread_id):
                if req.attachments:
                    await self._attachment_store.put(req.thread_id, req.attachments)

                graph = await self._get_graph()
                user_text = self._sanitizer.sanitize(req.user_text)

                result = await graph.ainvoke(
                    {"messages": [{"role": "user", "content": user_text}]},
                    config={"configurable": {"thread_id": req.thread_id}},
                )
                answer = _extract_response_text(result)
                if not answer:
                    answer = "No se generó respuesta del modelo."

                attachments = await self._attachment_store.get_recent(req.thread_id, limit=20)

            return {
                "response": answer,
                "tools_used": self._extract_tools_used(result),
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
                if req.attachments:
                    await self._attachment_store.put(req.thread_id, req.attachments)

                graph = await self._get_graph()
                user_text = self._sanitizer.sanitize(req.user_text)

                chunks: list[str] = []
                tools_used: list[str] = []
                final_output: Any = None

                async for event in graph.astream_events(
                    {"messages": [{"role": "user", "content": user_text}]},
                    config={"configurable": {"thread_id": req.thread_id}},
                    version="v2",
                ):
                    event_name = str(event.get("event", ""))
                    data = event.get("data", {}) if isinstance(event.get("data"), dict) else {}

                    if event_name == "on_chat_model_stream":
                        text = _content_to_text(data.get("chunk"))
                        if text:
                            chunks.append(text)
                            yield StreamingEvent(
                                type="content",
                                content=text,
                                thread_id=req.thread_id,
                                run_id=run_id,
                            )
                    elif event_name == "on_tool_start":
                        tool_name = str(event.get("name", "tool"))
                        if tool_name not in tools_used:
                            tools_used.append(tool_name)
                        yield StreamingEvent(
                            type="tool_start",
                            tool=tool_name,
                            payload=data.get("input") if isinstance(data.get("input"), dict) else {"input": data.get("input")},
                            thread_id=req.thread_id,
                            run_id=run_id,
                        )
                    elif event_name == "on_tool_end":
                        tool_name = str(event.get("name", "tool"))
                        output = data.get("output")
                        payload = output if isinstance(output, dict) else {"output": output}
                        yield StreamingEvent(
                            type="tool_result",
                            tool=tool_name,
                            payload=payload,
                            thread_id=req.thread_id,
                            run_id=run_id,
                        )
                    elif event_name == "on_chain_end" and "output" in data:
                        final_output = data.get("output")

                answer = "".join(chunks).strip()
                if not answer and final_output is not None:
                    answer = _extract_response_text(final_output)
                if not answer:
                    answer = "No se generó respuesta del modelo."

                attachments = await self._attachment_store.get_recent(req.thread_id, limit=20)
                yield StreamingEvent(
                    type="complete",
                    payload={
                        "tools_used": tools_used,
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

    @staticmethod
    def _extract_tools_used(result: Any) -> list[str]:
        tools_used: list[str] = []
        if not isinstance(result, dict):
            return tools_used

        messages = result.get("messages")
        if not isinstance(messages, list):
            return tools_used

        for message in messages:
            tool_calls = None
            if isinstance(message, dict):
                tool_calls = message.get("tool_calls")
            else:
                tool_calls = getattr(message, "tool_calls", None)

            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    name = None
                    if isinstance(tool_call, dict):
                        name = tool_call.get("name")
                    else:
                        name = getattr(tool_call, "name", None)
                    if isinstance(name, str) and name and name not in tools_used:
                        tools_used.append(name)

        return tools_used
