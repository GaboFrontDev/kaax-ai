from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, AsyncIterator
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.agent.lead_capture import (
    build_conversational_lead_payload,
    build_capture_response,
    is_affirmative_capture,
    is_capture_request,
    parse_lead_payload_from_text,
)
from app.agent.intent_router import IntentDecision, build_routing_response, is_greeting_message
from app.agent.intent_router_llm import LLMIntentRouter, IntentRoutingStructuredOutput
from app.agent.middleware.prompt_sanitizer import PromptSanitizerMiddleware
from app.agent.prompt_loader import load_prompt
from app.agent.runtime import AssistRequest, StreamingEvent, _content_to_text
from app.agent.tools.registry import ToolRegistry
from app.memory.attachments_store import AttachmentStore
from app.memory.langgraph_checkpointer import LangGraphCheckpointerManager
from app.memory.session_manager import SessionBusyError, SessionManager
from app.observability.logging import set_correlation

logger = logging.getLogger(__name__)


_THINKING_BLOCK_RE = re.compile(r"(?is)<thinking\b[^>]*>.*?</thinking\s*>")
_THINKING_TAG_RE = re.compile(r"(?is)</?thinking\b[^>]*>")


class DetectLeadCaptureReadinessArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    business_context: dict[str, Any] = Field(default_factory=dict)
    whatsapp_context: dict[str, Any] = Field(default_factory=dict)
    crm_context: dict[str, Any] = Field(default_factory=dict)
    agent_limits: dict[str, Any] = Field(default_factory=dict)
    lead_data: dict[str, Any] = Field(default_factory=dict)


class CaptureLeadIfReadyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    business_context: dict[str, Any] = Field(default_factory=dict)
    whatsapp_context: dict[str, Any] = Field(default_factory=dict)
    crm_context: dict[str, Any] = Field(default_factory=dict)
    agent_limits: dict[str, Any] = Field(default_factory=dict)
    lead_data: dict[str, Any] = Field(default_factory=dict)
    notify_owner: bool = False


def _strip_thinking_tags(text: str) -> str:
    if not text:
        return ""

    without_blocks = _THINKING_BLOCK_RE.sub("", text)
    without_tags = _THINKING_TAG_RE.sub("", without_blocks)
    return without_tags.strip()


class _ThinkingStreamFilter:
    _OPEN_TAG = "<thinking"
    _CLOSE_TAG = "</thinking>"

    def __init__(self) -> None:
        self._inside_thinking = False
        self._pending = ""

    @staticmethod
    def _prefix_overlap_len(value: str, prefix: str) -> int:
        max_len = min(len(value), len(prefix))
        for size in range(max_len, 0, -1):
            if value.endswith(prefix[:size]):
                return size
        return 0

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""

        self._pending += chunk
        output_parts: list[str] = []

        while self._pending:
            pending_lower = self._pending.lower()

            if self._inside_thinking:
                close_index = pending_lower.find(self._CLOSE_TAG)
                if close_index == -1:
                    overlap = self._prefix_overlap_len(pending_lower, self._CLOSE_TAG)
                    self._pending = self._pending[-overlap:] if overlap else ""
                    break

                self._pending = self._pending[close_index + len(self._CLOSE_TAG) :]
                self._inside_thinking = False
                continue

            open_index = pending_lower.find(self._OPEN_TAG)
            if open_index == -1:
                overlap = self._prefix_overlap_len(pending_lower, self._OPEN_TAG)
                if overlap:
                    output_parts.append(self._pending[:-overlap])
                    self._pending = self._pending[-overlap:]
                else:
                    output_parts.append(self._pending)
                    self._pending = ""
                break

            output_parts.append(self._pending[:open_index])
            self._pending = self._pending[open_index:]
            gt_index = self._pending.find(">")
            if gt_index == -1:
                break

            self._pending = self._pending[gt_index + 1 :]
            self._inside_thinking = True

        return "".join(output_parts)

    def finish(self) -> str:
        if self._inside_thinking:
            self._pending = ""
            return ""
        if self._OPEN_TAG.startswith(self._pending.lower()):
            self._pending = ""
            return ""
        tail = self._pending
        self._pending = ""
        return tail


def _extract_response_text(result: Any) -> str:
    text = ""

    if isinstance(result, str):
        text = result
    elif isinstance(result, dict):
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
                        break
            if text:
                return _strip_thinking_tags(text)

        for key in ("output", "response", "answer"):
            if key in result:
                text = _content_to_text(result.get(key))
                if text:
                    return _strip_thinking_tags(text)

        text = json.dumps(result, ensure_ascii=True)
    else:
        text = _content_to_text(result)

    return _strip_thinking_tags(text)


def build_langchain_tools(tool_registry: ToolRegistry) -> list[Any]:
    from langchain_core.tools import StructuredTool, tool

    @tool
    async def crm_upsert_quote(payload: dict[str, Any]) -> dict[str, Any]:
        """Store quote/lead data in kaax internal CRM registry using a structured payload."""

        return (await tool_registry.execute("crm_upsert_quote", {"payload": payload})).output

    async def _detect_lead_capture_readiness(
        business_context: dict[str, Any] | None = None,
        whatsapp_context: dict[str, Any] | None = None,
        crm_context: dict[str, Any] | None = None,
        agent_limits: dict[str, Any] | None = None,
        lead_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Detect if a lead has enough context and evidence to be registered internally in kaax CRM."""

        payload: dict[str, Any] = {}
        if business_context is not None:
            payload["business_context"] = business_context
        if whatsapp_context is not None:
            payload["whatsapp_context"] = whatsapp_context
        if crm_context is not None:
            payload["crm_context"] = crm_context
        if agent_limits is not None:
            payload["agent_limits"] = agent_limits
        if lead_data is not None:
            payload["lead_data"] = lead_data

        return (await tool_registry.execute("detect_lead_capture_readiness", payload)).output

    detect_lead_capture_readiness = StructuredTool.from_function(
        name="detect_lead_capture_readiness",
        description="Detect if a lead has enough context and evidence to be registered in kaax internal CRM.",
        args_schema=DetectLeadCaptureReadinessArgs,
        coroutine=_detect_lead_capture_readiness,
    )

    async def _capture_lead_if_ready(
        business_context: dict[str, Any] | None = None,
        whatsapp_context: dict[str, Any] | None = None,
        crm_context: dict[str, Any] | None = None,
        agent_limits: dict[str, Any] | None = None,
        lead_data: dict[str, Any] | None = None,
        notify_owner: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"notify_owner": bool(notify_owner)}
        if business_context is not None:
            payload["business_context"] = business_context
        if whatsapp_context is not None:
            payload["whatsapp_context"] = whatsapp_context
        if crm_context is not None:
            payload["crm_context"] = crm_context
        if agent_limits is not None:
            payload["agent_limits"] = agent_limits
        if lead_data is not None:
            payload["lead_data"] = lead_data

        return (await tool_registry.execute("capture_lead_if_ready", payload)).output

    capture_lead_if_ready = StructuredTool.from_function(
        name="capture_lead_if_ready",
        description=(
            "Validate lead readiness, register in kaax internal CRM when required fields are present, "
            "and optionally notify owner by WhatsApp."
        ),
        args_schema=CaptureLeadIfReadyArgs,
        coroutine=_capture_lead_if_ready,
    )

    return [
        crm_upsert_quote,
        detect_lead_capture_readiness,
        capture_lead_if_ready,
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
        llm_intent_router_enabled: bool,
        llm_intent_router_confidence_threshold: float,
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
        self._llm_intent_router_enabled = llm_intent_router_enabled
        self._llm_intent_router_confidence_threshold = llm_intent_router_confidence_threshold

        self._graph: Any | None = None
        self._graph_lock = asyncio.Lock()
        self._llm_intent_router: LLMIntentRouter | None = None
        self._llm_intent_router_init_failed = False
        self._seen_threads: set[str] = set()
        self._pending_lead_payload_by_thread: dict[str, dict[str, Any]] = {}
        self._user_messages_by_thread: dict[str, list[str]] = {}
        self._captured_lead_threads: set[str] = set()

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

                user_text = self._sanitizer.sanitize(req.user_text)
                logger.info("langchain_runtime_inbound thread_id=%s text=%s", req.thread_id, user_text[:300])
                self._track_user_message(req.thread_id, user_text)
                if (
                    lead_response := await self._maybe_handle_explicit_lead_capture(
                        thread_id=req.thread_id,
                        user_text=user_text,
                    )
                ) is not None:
                    attachments = await self._attachment_store.get_recent(req.thread_id, limit=20)
                    return {
                        "response": lead_response["response"],
                        "tools_used": lead_response["tools_used"],
                        "completion_time": round(time.perf_counter() - started_at, 3),
                        "conversation_id": req.thread_id,
                        "run_id": run_id,
                        "attachments": attachments,
                    }

                first_turn = self._mark_and_check_first_turn(req.thread_id)
                routing = await self._route_intent(user_text)
                if routing.route != "in_scope":
                    answer = build_routing_response(
                        routing,
                        first_turn_greeting=first_turn and is_greeting_message(user_text),
                    )
                    attachments = await self._attachment_store.get_recent(req.thread_id, limit=20)
                    return {
                        "response": answer,
                        "tools_used": [],
                        "completion_time": round(time.perf_counter() - started_at, 3),
                        "conversation_id": req.thread_id,
                        "run_id": run_id,
                        "attachments": attachments,
                    }

                graph = await self._get_graph()

                result = await graph.ainvoke(
                    {"messages": [{"role": "user", "content": user_text}]},
                    config={"configurable": {"thread_id": req.thread_id}},
                )
                answer = _extract_response_text(result)
                if not answer:
                    answer = "No se generó respuesta del modelo."
                auto_capture_tools: list[str] = []
                auto_capture_response = await self._maybe_auto_capture_lead(
                    thread_id=req.thread_id,
                    user_text=user_text,
                )
                if auto_capture_response is not None:
                    answer = f"{answer}\n\n{auto_capture_response['response']}".strip()
                    auto_capture_tools = auto_capture_response["tools_used"]

                attachments = await self._attachment_store.get_recent(req.thread_id, limit=20)

            return {
                "response": answer,
                "tools_used": self._extract_tools_used(result) + auto_capture_tools,
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

                user_text = self._sanitizer.sanitize(req.user_text)
                logger.info("langchain_runtime_inbound_stream thread_id=%s text=%s", req.thread_id, user_text[:300])
                self._track_user_message(req.thread_id, user_text)
                if (
                    lead_response := await self._maybe_handle_explicit_lead_capture(
                        thread_id=req.thread_id,
                        user_text=user_text,
                    )
                ) is not None:
                    for tool_name in lead_response["tools_used"]:
                        yield StreamingEvent(
                            type="tool_start",
                            tool=tool_name,
                            payload={"source": "lead_capture_fast_path"},
                            thread_id=req.thread_id,
                            run_id=run_id,
                        )
                        yield StreamingEvent(
                            type="tool_result",
                            tool=tool_name,
                            payload={"status": "done"},
                            thread_id=req.thread_id,
                            run_id=run_id,
                        )
                    for chunk in self._chunk_text(lead_response["response"]):
                        yield StreamingEvent(
                            type="content",
                            content=chunk,
                            thread_id=req.thread_id,
                            run_id=run_id,
                        )
                    attachments = await self._attachment_store.get_recent(req.thread_id, limit=20)
                    yield StreamingEvent(
                        type="complete",
                        payload={
                            "tools_used": lead_response["tools_used"],
                            "attachments": attachments,
                        },
                        thread_id=req.thread_id,
                        run_id=run_id,
                    )
                    return

                first_turn = self._mark_and_check_first_turn(req.thread_id)
                routing = await self._route_intent(user_text)

                if routing.route != "in_scope":
                    answer = build_routing_response(
                        routing,
                        first_turn_greeting=first_turn and is_greeting_message(user_text),
                    )
                    for chunk in self._chunk_text(answer):
                        yield StreamingEvent(
                            type="content",
                            content=chunk,
                            thread_id=req.thread_id,
                            run_id=run_id,
                        )
                    attachments = await self._attachment_store.get_recent(req.thread_id, limit=20)
                    yield StreamingEvent(
                        type="complete",
                        payload={
                            "tools_used": [],
                            "attachments": attachments,
                        },
                        thread_id=req.thread_id,
                        run_id=run_id,
                    )
                    return

                graph = await self._get_graph()

                chunks: list[str] = []
                tools_used: list[str] = []
                final_output: Any = None
                thinking_filter = _ThinkingStreamFilter()

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
                            visible = thinking_filter.feed(text)
                            if visible:
                                chunks.append(visible)
                                yield StreamingEvent(
                                    type="content",
                                    content=visible,
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
                else:
                    tail = thinking_filter.finish()
                    if tail:
                        cleaned_tail = _strip_thinking_tags(tail)
                        if cleaned_tail:
                            answer = f"{answer}{cleaned_tail}".strip()
                if not answer:
                    answer = "No se generó respuesta del modelo."

                auto_capture_response = await self._maybe_auto_capture_lead(
                    thread_id=req.thread_id,
                    user_text=user_text,
                )
                if auto_capture_response is not None:
                    for tool_name in auto_capture_response["tools_used"]:
                        if tool_name not in tools_used:
                            tools_used.append(tool_name)
                        yield StreamingEvent(
                            type="tool_start",
                            tool=tool_name,
                            payload={"source": "lead_capture_auto"},
                            thread_id=req.thread_id,
                            run_id=run_id,
                        )
                        yield StreamingEvent(
                            type="tool_result",
                            tool=tool_name,
                            payload={"status": "done"},
                            thread_id=req.thread_id,
                            run_id=run_id,
                        )
                    for chunk in self._chunk_text(auto_capture_response["response"]):
                        yield StreamingEvent(
                            type="content",
                            content=chunk,
                            thread_id=req.thread_id,
                            run_id=run_id,
                        )

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

    async def _route_intent(self, user_text: str) -> IntentDecision:
        if not self._llm_intent_router_enabled:
            return IntentDecision(route="in_scope", confidence=1.0, reason="intent_router_disabled")

        router = await self._get_llm_intent_router()
        if router is None:
            return IntentDecision(route="in_scope", confidence=0.0, reason="intent_router_unavailable")

        decision = await router.route(user_text)
        if decision.reason == "llm_router_failed":
            return IntentDecision(route="in_scope", confidence=0.0, reason="intent_router_failed_open")
        return decision

    async def _maybe_handle_explicit_lead_capture(
        self,
        *,
        thread_id: str,
        user_text: str,
    ) -> dict[str, Any] | None:
        parsed_payload = parse_lead_payload_from_text(user_text)
        if parsed_payload:
            self._pending_lead_payload_by_thread[thread_id] = parsed_payload
            logger.info("langchain_lead_payload_staged thread_id=%s sections=%s", thread_id, sorted(parsed_payload.keys()))

        wants_capture = is_capture_request(user_text) or is_affirmative_capture(user_text)
        if not wants_capture:
            return None

        payload = parsed_payload or self._pending_lead_payload_by_thread.get(thread_id)
        if payload is None:
            return {
                "response": (
                    "Para registrarlo necesito estos datos minimos: empresa, contacto (email o telefono), "
                    "necesidad principal y timeline."
                ),
                "tools_used": [],
            }

        tool_payload = dict(payload)
        tool_payload["notify_owner"] = False
        result = await self._tool_registry.execute("capture_lead_if_ready", tool_payload)
        logger.info(
            "langchain_lead_capture_result thread_id=%s output=%s",
            thread_id,
            json.dumps(result.output, ensure_ascii=True),
        )
        if str(result.output.get("status", "")).lower() == "captured":
            self._pending_lead_payload_by_thread.pop(thread_id, None)
            self._captured_lead_threads.add(thread_id)

        return {"response": build_capture_response(result.output), "tools_used": [result.tool]}

    async def _maybe_auto_capture_lead(
        self,
        *,
        thread_id: str,
        user_text: str,
    ) -> dict[str, Any] | None:
        if thread_id in self._captured_lead_threads:
            return None

        payload = build_conversational_lead_payload(
            self._user_messages_as_state(thread_id),
            latest_user_text=user_text,
        )
        if payload is None:
            return None

        readiness = await self._tool_registry.execute("detect_lead_capture_readiness", payload)
        if not bool(readiness.output.get("ready_for_capture")):
            logger.info(
                "langchain_lead_capture_auto_skipped thread_id=%s missing=%s",
                thread_id,
                readiness.output.get("missing_critical_fields", []),
            )
            return None

        capture_payload = dict(payload)
        capture_payload["notify_owner"] = False
        result = await self._tool_registry.execute("capture_lead_if_ready", capture_payload)
        logger.info(
            "langchain_lead_capture_auto_result thread_id=%s output=%s",
            thread_id,
            json.dumps(result.output, ensure_ascii=True),
        )
        if str(result.output.get("status", "")).lower() != "captured":
            return None

        self._captured_lead_threads.add(thread_id)
        return {"response": build_capture_response(result.output), "tools_used": [result.tool]}

    def _mark_and_check_first_turn(self, thread_id: str) -> bool:
        first_turn = thread_id not in self._seen_threads
        self._seen_threads.add(thread_id)
        return first_turn

    def _track_user_message(self, thread_id: str, user_text: str) -> None:
        history = self._user_messages_by_thread.setdefault(thread_id, [])
        cleaned = user_text.strip()
        if cleaned:
            history.append(cleaned)
        if len(history) > 20:
            del history[:-20]

    def _user_messages_as_state(self, thread_id: str) -> list[dict[str, Any]]:
        history = self._user_messages_by_thread.get(thread_id, [])
        return [{"role": "user", "content": message} for message in history]

    async def _get_llm_intent_router(self) -> LLMIntentRouter | None:
        if self._llm_intent_router is not None:
            return self._llm_intent_router
        if self._llm_intent_router_init_failed:
            return None

        try:
            from langchain_aws import ChatBedrockConverse

            model = ChatBedrockConverse(
                model_id=self._small_model_name,
                region_name=self._aws_region,
                temperature=0,
                disable_streaming=True,
            )
            structured_model = model.with_structured_output(IntentRoutingStructuredOutput)
            self._llm_intent_router = LLMIntentRouter(
                model=structured_model,
                system_prompt=load_prompt("intent_router"),
                confidence_threshold=self._llm_intent_router_confidence_threshold,
            )
            return self._llm_intent_router
        except Exception as exc:  # pragma: no cover - fallback safety
            self._llm_intent_router_init_failed = True
            logger.warning("llm_intent_router_unavailable_fallback_keywords: %s", exc)
            return None

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 80) -> list[str]:
        return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)] or [""]

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
