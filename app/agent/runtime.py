from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, AsyncIterator, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.agent.middleware.prompt_sanitizer import PromptSanitizerMiddleware
from app.agent.middleware.summarization import SummarizationMiddleware
from app.agent.tools.registry import ToolExecutionResult, ToolRegistry
from app.memory.attachments_store import AttachmentStore
from app.memory.session_manager import SessionBusyError, SessionManager
from app.observability.logging import set_correlation


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        for key in ("text", "content", "output", "answer"):
            value = content.get(key)
            if isinstance(value, str):
                return value
        return json.dumps(content, ensure_ascii=True)

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    text_attr = getattr(content, "content", None)
    if text_attr is not None and text_attr is not content:
        return _content_to_text(text_attr)

    return str(content)


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


class DefaultAgentRuntime:
    def __init__(
        self,
        session_manager: SessionManager,
        attachment_store: AttachmentStore,
        tool_registry: ToolRegistry,
        sanitizer: PromptSanitizerMiddleware,
        summarizer: SummarizationMiddleware,
        *,
        tool_retry_attempts: int = 2,
        tool_retry_backoff_ms: int = 200,
    ) -> None:
        self._session_manager = session_manager
        self._attachment_store = attachment_store
        self._tool_registry = tool_registry
        self._sanitizer = sanitizer
        self._summarizer = summarizer
        self._tool_retry_attempts = max(1, tool_retry_attempts)
        self._tool_retry_backoff_ms = max(0, tool_retry_backoff_ms)

    async def invoke(self, req: AssistRequest) -> dict[str, Any]:
        started_at = time.perf_counter()
        run_id = str(uuid4())
        set_correlation(thread_id=req.thread_id, run_id=run_id)

        try:
            async with self._session_manager.session_lock(req.thread_id):
                if req.attachments:
                    await self._attachment_store.put(req.thread_id, req.attachments)

                state = await self._load_state(req.thread_id)
                self._repair_dangling_tool_calls(state)

                user_text = self._sanitizer.sanitize(req.user_text)
                state["messages"].append({"role": "user", "content": user_text})

                tool_results = await self._run_selected_tools(user_text)
                answer = self._compose_answer(user_text, tool_results)

                state["messages"].append({"role": "assistant", "content": answer})
                state["pending_tool_calls"] = []
                await self._summarizer.maybe_summarize(state)
                await self._session_manager.put_state(req.thread_id, state)

                attachments = await self._attachment_store.get_recent(req.thread_id, limit=20)

            return {
                "response": answer,
                "tools_used": [result.tool for result in tool_results],
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

                state = await self._load_state(req.thread_id)
                self._repair_dangling_tool_calls(state)

                user_text = self._sanitizer.sanitize(req.user_text)
                state["messages"].append({"role": "user", "content": user_text})

                tool_plan = self._select_tools(user_text)
                tool_results: list[ToolExecutionResult] = []

                for tool_name, payload in tool_plan:
                    state["pending_tool_calls"].append({"tool": tool_name, "input": payload})
                    yield StreamingEvent(
                        type="tool_start",
                        tool=tool_name,
                        payload=payload,
                        thread_id=req.thread_id,
                        run_id=run_id,
                    )
                    result = await self._execute_tool_with_retry(tool_name, payload)
                    tool_results.append(result)
                    yield StreamingEvent(
                        type="tool_result",
                        tool=tool_name,
                        payload=result.output,
                        thread_id=req.thread_id,
                        run_id=run_id,
                    )

                answer = self._compose_answer(user_text, tool_results)
                state["messages"].append({"role": "assistant", "content": answer})
                state["pending_tool_calls"] = []

                for chunk in self._chunk_text(answer):
                    yield StreamingEvent(
                        type="content",
                        content=chunk,
                        thread_id=req.thread_id,
                        run_id=run_id,
                    )

                await self._summarizer.maybe_summarize(state)
                await self._session_manager.put_state(req.thread_id, state)
                attachments = await self._attachment_store.get_recent(req.thread_id, limit=20)

                yield StreamingEvent(
                    type="complete",
                    payload={
                        "tools_used": [result.tool for result in tool_results],
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

    async def _load_state(self, thread_id: str) -> dict[str, Any]:
        state = await self._session_manager.get_state(thread_id)
        if state is not None:
            return state
        return {"messages": [], "pending_tool_calls": [], "summary": None}

    def _repair_dangling_tool_calls(self, state: dict[str, Any]) -> None:
        dangling = list(state.get("pending_tool_calls", []))
        if not dangling:
            return
        for call in dangling:
            tool = call.get("tool", "unknown")
            payload = json.dumps(call.get("input", {}), ensure_ascii=True)
            state["messages"].append(
                {
                    "role": "tool",
                    "content": f"Synthetic tool completion injected for dangling call: {tool} {payload}",
                }
            )
        state["pending_tool_calls"] = []

    async def _run_selected_tools(self, user_text: str) -> list[ToolExecutionResult]:
        results: list[ToolExecutionResult] = []
        for tool_name, payload in self._select_tools(user_text):
            results.append(await self._execute_tool_with_retry(tool_name, payload))
        return results

    async def _execute_tool_with_retry(self, tool_name: str, payload: dict[str, Any]) -> ToolExecutionResult:
        last_exc: Exception | None = None
        for attempt in range(1, self._tool_retry_attempts + 1):
            try:
                return await self._tool_registry.execute(tool_name, payload)
            except Exception as exc:  # pragma: no cover - safeguard path
                last_exc = exc
                if attempt >= self._tool_retry_attempts:
                    break
                delay = (self._tool_retry_backoff_ms / 1000.0) * (2 ** (attempt - 1))
                await asyncio.sleep(delay)

        raise RuntimeError(f"tool failed after retries: {tool_name}: {last_exc}") from last_exc

    def _select_tools(self, user_text: str) -> list[tuple[str, dict[str, Any]]]:
        text = user_text.lower()
        calls: list[tuple[str, dict[str, Any]]] = []

        if any(token in text for token in ("iso", "country code", "codigo pais", "código país")):
            country = self._extract_country_name(user_text)
            calls.append(("get_iso_country_code", {"country_name": country}))

        if any(token in text for token in ("market", "mercado")):
            payload: dict[str, Any] = {"query": user_text, "limit": 10}
            country_code = self._extract_country_code(user_text)
            if country_code:
                payload["country_code"] = country_code
            calls.append(("retrieve_markets", payload))

        if any(token in text for token in ("segment", "segmento")):
            calls.append(("retrieve_segments", {"query": user_text, "taxonomy": "default", "limit": 10}))

        return calls

    @staticmethod
    def _extract_country_name(user_text: str) -> str:
        match = re.search(r"(?:de|of)\s+([A-Za-z\s]+)$", user_text.strip(), flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return user_text.strip().split()[-1]

    @staticmethod
    def _extract_country_code(user_text: str) -> str | None:
        match = re.search(r"\b([A-Z]{2})\b", user_text)
        if not match:
            return None
        return match.group(1)

    @staticmethod
    def _compose_answer(user_text: str, tool_results: list[ToolExecutionResult]) -> str:
        if not tool_results:
            return f"Entendido. Respuesta base sin tools: {user_text}"

        snippets = []
        for result in tool_results:
            snippets.append(f"{result.tool}: {json.dumps(result.output, ensure_ascii=True)}")
        joined = " | ".join(snippets)
        return f"Resultado validado de tools -> {joined}"

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 80) -> list[str]:
        return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)] or [""]
