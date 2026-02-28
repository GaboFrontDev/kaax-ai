from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
import logging
import re
import unicodedata
from typing import Any, Awaitable, Callable, Literal

from langgraph.graph import END, START, StateGraph

from app.agent.prompt_loader import load_prompt
from app.agent.orchestration.routing_rules import derive_router_and_state
from app.agent.orchestration.schemas import OrchestrationState, RouterDecision
from app.agent.orchestration.subagents import (
    SubagentRunner,
    invoke_core_capture,
    invoke_greeting,
    invoke_inventory,
    invoke_knowledge,
)
from app.agent.tools.capture_lead_if_ready_tool import CaptureLeadIfReadyTool
from app.agent.tools.context import ToolRequestContextManager
from app.knowledge.providers import KnowledgeProvider
from app.memory.session_manager import SessionManager

logger = logging.getLogger(__name__)


def build_mvp_orchestration_graph(
    *,
    session_manager: SessionManager,
    knowledge_provider: KnowledgeProvider,
    tool_context_manager: ToolRequestContextManager,
    capture_tool: CaptureLeadIfReadyTool | None,
    subagent_runner: SubagentRunner | None = None,
    knowledge_search_limit: int = 3,
    notify_owner: bool = False,
    supervisor_model_name: str = "anthropic.claude-3-haiku-20240307-v1:0",
    aws_region: str = "us-east-1",
    supervisor_temperature: float = 0,
    supervisor_override: Callable[[str, dict[str, Any]], Awaitable[RouterDecision | dict[str, Any] | None]]
    | None = None,
) -> Any:
    supervisor_prompt = load_prompt("supervisor")
    supervisor_lock = asyncio.Lock()
    supervisor_model: Any | None = None
    supervisor_unavailable = False

    async def _get_supervisor_model() -> Any | None:
        nonlocal supervisor_model, supervisor_unavailable
        if supervisor_model is not None:
            return supervisor_model
        if supervisor_unavailable:
            return None

        async with supervisor_lock:
            if supervisor_model is not None:
                return supervisor_model
            if supervisor_unavailable:
                return None

            try:
                from langchain_aws import ChatBedrockConverse

                llm = ChatBedrockConverse(
                    model_id=supervisor_model_name,
                    region_name=aws_region,
                    temperature=supervisor_temperature,
                    disable_streaming=True,
                )
                supervisor_model = llm.with_structured_output(RouterDecision)
                return supervisor_model
            except Exception as exc:  # pragma: no cover - provider/network dependent
                supervisor_unavailable = True
                logger.warning("supervisor_model_unavailable: %s", exc)
                return None

    async def _llm_supervisor_decision(
        *,
        user_message: str,
        serialized_state: dict[str, Any],
    ) -> RouterDecision | None:
        if supervisor_override is not None:
            raw_override = await supervisor_override(user_message, serialized_state)
            if raw_override is None:
                return None
            if isinstance(raw_override, RouterDecision):
                return raw_override
            try:
                return RouterDecision.model_validate(raw_override)
            except Exception:
                return None

        model = await _get_supervisor_model()
        if model is None:
            return None

        payload = json.dumps(
            {
                "last_user_message": user_message,
                "conversation_state": serialized_state,
            },
            ensure_ascii=True,
        )
        try:
            decision = await model.ainvoke(
                [
                    {"role": "system", "content": supervisor_prompt},
                    {"role": "user", "content": payload},
                ]
            )
            if isinstance(decision, RouterDecision):
                return decision
            return RouterDecision.model_validate(decision)
        except Exception as exc:  # pragma: no cover - provider/network dependent
            logger.warning("supervisor_llm_route_failed: %s", exc)
            return None

    async def supervisor_route(state: OrchestrationState) -> dict[str, Any]:
        fallback_router, conversation_state = derive_router_and_state(
            user_message=state.last_user_message,
            conversation_state=state.conversation_state,
        )
        serialized_state = conversation_state.model_dump()
        llm_router = await _llm_supervisor_decision(
            user_message=state.last_user_message,
            serialized_state=serialized_state,
        )
        if llm_router is None:
            router = _coerce_safe_router_fallback(
                router=fallback_router,
                missing_fields=fallback_router.missing_fields,
            )
        else:
            router = _apply_router_guardrails(
                router=llm_router,
                missing_fields=fallback_router.missing_fields,
            )
            conversation_state.lead.intent = router.intent
            conversation_state.lead.qualification = router.qualification
            conversation_state.mode = router.mode

        if router.mode == "capture_completion":
            conversation_state.lead.status = "calificado" if not router.missing_fields else "en_revision"
        elif router.mode == "handoff" and conversation_state.lead.status != "no_calificado":
            conversation_state.lead.status = "en_revision"

        return {
            "conversation_state": conversation_state,
            "router": router,
            "tool_result": None,
        }

    async def agent_core_capture(state: OrchestrationState) -> dict[str, Any]:
        router = state.router
        if router is None:
            return {"draft_response": "Para avanzar, confirme su nombre, telefono y horario preferido de contacto."}

        if router.next_action == "capture_lead":
            return {"draft_response": "Confirmado. Procedere al registro de su solicitud comercial."}

        missing_fields = list(router.missing_fields)
        if not missing_fields:
            return {"draft_response": "Confirmado. Procedere al registro de su solicitud comercial."}

        llm_response = await invoke_core_capture(
            runner=subagent_runner,
            user_message=state.last_user_message,
            context={
                "router": router.model_dump(),
                "conversation_state": state.conversation_state.model_dump(),
                "missing_fields": missing_fields,
            },
        )
        if llm_response:
            return {"draft_response": llm_response}

        question = _build_missing_fields_question(missing_fields)
        return {"draft_response": question}

    async def agent_greeting(state: OrchestrationState) -> dict[str, Any]:
        llm_response = await invoke_greeting(
            runner=subagent_runner,
            user_message=state.last_user_message,
            context={
                "router": state.router.model_dump() if state.router is not None else None,
                "conversation_state": state.conversation_state.model_dump(),
            },
        )
        if llm_response:
            return {"draft_response": llm_response}

        return {
            "draft_response": (
                "Hola. Soy Kaax AI. Puedo ayudarle con soporte, funcionamiento del producto o precios. "
                "Indiqueme que desea revisar."
            )
        }

    async def agent_knowledge(state: OrchestrationState) -> dict[str, Any]:
        router = state.router
        if router is not None and router.mode == "greeting":
            return {
                "draft_response": (
                    "Hola. Soy Kaax AI. Automatizamos atencion y calificacion de prospectos en WhatsApp. "
                    "Desea soporte o evaluar implementacion?"
                )
            }

        matches = await _search_knowledge(
            knowledge_provider=knowledge_provider,
            tool_context_manager=tool_context_manager,
            requestor=state.requestor,
            query=state.last_user_message,
            limit=knowledge_search_limit,
        )
        relevant = _pick_relevant_match(state.last_user_message, matches)

        llm_response = await invoke_knowledge(
            runner=subagent_runner,
            user_message=state.last_user_message,
            context={
                "router": router.model_dump() if router is not None else None,
                "conversation_state": state.conversation_state.model_dump(),
                "knowledge_matches": matches[:3],
            },
        )
        if llm_response:
            return {"draft_response": llm_response}

        if router is not None and router.mode == "support_answer":
            if relevant is not None:
                response = (
                    f"Segun la base disponible para soporte: {_compact_text(relevant.get('content', ''))}. "
                    "Si el problema persiste, indique mensaje de error y canal afectado para escalar."
                )
            else:
                response = (
                    "Para soporte tecnico, comparta el mensaje de error exacto, cuando ocurre y canal afectado "
                    "(WhatsApp, API o Slack) para diagnosticarlo."
                )
            return {"draft_response": response}

        if router is not None and router.intent in {"product_inquiry", "unknown"}:
            response = (
                "Kaax AI automatiza atencion inicial, clasificacion de prospectos y handoff comercial, "
                "manteniendo contexto por conversacion."
            )
            if router.intent == "unknown":
                response = (
                    f"{response} Indique si desea foco en implementacion, integraciones o flujo comercial."
                )
            return {"draft_response": response}

        if relevant is not None:
            response = f"Segun la base disponible: {_compact_text(relevant.get('content', ''))}"
        else:
            response = "No tengo informacion factual confirmada en la base para responder eso en este momento."

        if router is not None and router.intent == "purchase_intent":
            response = (
                f"{response} Si desea avanzar con implementacion, puedo activar el flujo de coordinacion comercial."
            )
        return {"draft_response": response}

    async def agent_inventory(state: OrchestrationState) -> dict[str, Any]:
        matches = await _search_knowledge(
            knowledge_provider=knowledge_provider,
            tool_context_manager=tool_context_manager,
            requestor=state.requestor,
            query=state.last_user_message,
            limit=knowledge_search_limit,
        )

        normalized_message = (state.last_user_message or "").lower()
        relevant = _pick_relevant_match(state.last_user_message, matches)
        snapshot_summary = (state.conversation_state.pricing_context.verified_summary or "").strip()
        asks_period = "mensual" in normalized_message or "anual" in normalized_message
        pricing_question = _build_pricing_clarification_question(
            normalized_message=normalized_message,
            asks_period=asks_period,
        )

        updated_conversation_state = state.conversation_state.model_copy(deep=True)
        selected_source = "none"
        selected_summary = ""
        conflict = False

        if relevant is not None:
            selected_source = "kb"
            selected_summary = _compact_text(relevant.get("content", ""))
            conflict = bool(snapshot_summary) and _pricing_facts_conflict(snapshot_summary, selected_summary)
            updated_conversation_state.pricing_context.verified_summary = selected_summary
            updated_conversation_state.pricing_context.source = "kb"
            updated_conversation_state.pricing_context.query = state.last_user_message
            updated_conversation_state.pricing_context.updated_at = datetime.now(UTC).isoformat()

            if conflict:
                response = (
                    "Detecto una diferencia entre la informacion comercial previamente verificada "
                    "y la base actual. Para evitar errores, confirme cual version debo usar. "
                    f"{pricing_question}"
                )
            else:
                response = f"Informacion confirmada de planes/precios: {selected_summary}"
                if pricing_question:
                    response = f"{response} {pricing_question}"
        elif snapshot_summary:
            selected_source = "snapshot"
            selected_summary = snapshot_summary
            response = f"Con base en la informacion verificada previamente: {snapshot_summary}"
            if pricing_question:
                response = f"{response} {pricing_question}"
        else:
            response = (
                "No tengo informacion confirmada de precios o planes en este momento. "
                f"{pricing_question}"
            )

        # Fallback comercial oficial cuando el usuario pide precio y no hay monto claro.
        if _is_pricing_request(normalized_message) and not conflict:
            response = _enforce_pricing_answer(
                base_response=response,
                normalized_message=normalized_message,
            )

        if not conflict:
            llm_response = await invoke_inventory(
                runner=subagent_runner,
                user_message=state.last_user_message,
                context={
                    "router": state.router.model_dump() if state.router is not None else None,
                    "conversation_state": updated_conversation_state.model_dump(),
                    "knowledge_matches": matches[:3],
                    "pricing_source": selected_source,
                    "pricing_summary": selected_summary,
                    "pricing_conflict": conflict,
                    "required_clarification": pricing_question,
                },
            )
            if llm_response:
                if _is_pricing_request(normalized_message):
                    llm_response = _enforce_pricing_answer(
                        base_response=llm_response,
                        normalized_message=normalized_message,
                    )
                response = llm_response

        if state.router is not None and state.router.intent == "purchase_intent":
            response = f"{response} Si desea, puedo pasar al flujo de coordinacion comercial."
        return {
            "draft_response": response,
            "conversation_state": updated_conversation_state,
        }

    async def tool_capture_lead_if_ready(state: OrchestrationState) -> dict[str, Any]:
        if capture_tool is None:
            return {
                "tool_result": {
                    "status": "failed",
                    "error": "capture_tool_unavailable",
                },
                "tools_used": _append_tool(state.tools_used, "capture_lead_if_ready"),
            }

        lead_data = {
            "contact_name": state.conversation_state.captured.contact_name,
            "phone": state.conversation_state.captured.phone,
            "contact_schedule": state.conversation_state.captured.contact_schedule,
            "intent": state.conversation_state.lead.intent,
            "qualification": state.conversation_state.lead.qualification,
        }
        payload: dict[str, Any] = {
            "lead_data": lead_data,
            "notify_owner": bool(notify_owner),
        }
        try:
            result = await capture_tool.execute(payload)
            return {
                "tool_result": result,
                "tools_used": _append_tool(state.tools_used, "capture_lead_if_ready"),
            }
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning("capture_lead_if_ready_failed thread_id=%s error=%s", state.thread_id, exc)
            return {
                "tool_result": {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                "tools_used": _append_tool(state.tools_used, "capture_lead_if_ready"),
            }

    async def persist_state(state: OrchestrationState) -> dict[str, Any]:
        conversation_state = state.conversation_state.model_copy(deep=True)
        router = state.router
        if router is not None:
            conversation_state.mode = router.mode
            conversation_state.lead.intent = router.intent
            conversation_state.lead.qualification = router.qualification

            if router.mode == "capture_completion":
                conversation_state.lead.status = "calificado" if not router.missing_fields else "en_revision"
            elif router.mode == "handoff" and conversation_state.lead.status != "no_calificado":
                conversation_state.lead.status = "en_revision"

        normalized_tool_result = _normalize_tool_result(state.tool_result)
        if normalized_tool_result is not None:
            status = normalized_tool_result.get("status")
            if status == "captured":
                conversation_state.lead.status = "calificado"
            elif status == "missing":
                conversation_state.lead.status = "en_revision"
            elif status == "not_qualified":
                conversation_state.lead.status = "no_calificado"

        persisted: dict[str, Any] = {
            "conversation_state": conversation_state.model_dump(),
            "last_user_message": state.last_user_message,
        }
        if normalized_tool_result is not None:
            persisted["tool_result"] = normalized_tool_result
        await session_manager.put_state(state.thread_id, persisted)
        return {
            "conversation_state": conversation_state,
            "tool_result": normalized_tool_result,
        }

    def finalize(state: OrchestrationState) -> dict[str, Any]:
        router = state.router
        if router is not None and router.next_action == "handoff":
            return {
                "final_response": "Entendido. Escalare su caso con un asesor humano para continuar por el canal comercial."
            }

        tool_result = state.tool_result or {}
        if tool_result:
            status = str(tool_result.get("status") or "").lower()
            if status == "captured":
                return {
                    "final_response": (
                        "Registro completado correctamente. Un asesor comercial continuara con usted "
                        "en el horario indicado."
                    )
                }
            if status == "missing":
                missing = _normalize_missing_fields(tool_result.get("missing_fields"))
                return {"final_response": _build_missing_fields_question(missing)}
            if status == "not_qualified":
                return {
                    "final_response": (
                        "Gracias por la informacion. En este momento no es posible avanzar "
                        "comercialmente segun la politica de calificacion."
                    )
                }
            if status == "error":
                return {
                    "final_response": (
                        "No fue posible completar el registro por una falla operativa. "
                        "Compartire el caso para seguimiento humano."
                    )
                }

        draft = (state.draft_response or "").strip()
        if draft:
            return {"final_response": draft}

        if router is not None and router.mode == "greeting":
            return {
                "final_response": (
                    "Hola. Soy Kaax AI. Automatizamos atencion y calificacion de prospectos en WhatsApp."
                )
            }
        return {"final_response": "Recibido. Comparta mas detalle para avanzar con precision."}

    def pick_agent(state: OrchestrationState) -> Literal["greeting", "core_capture", "knowledge", "inventory"]:
        router = state.router
        if router is None:
            return "knowledge"
        return router.agent

    def after_agent(state: OrchestrationState) -> Literal["tool_capture_lead_if_ready", "persist_state"]:
        router = state.router
        if router is not None and router.next_action == "capture_lead":
            return "tool_capture_lead_if_ready"
        return "persist_state"

    builder = StateGraph(OrchestrationState)
    builder.add_node("supervisor_route", supervisor_route)
    builder.add_node("agent_greeting", agent_greeting)
    builder.add_node("agent_core_capture", agent_core_capture)
    builder.add_node("agent_knowledge", agent_knowledge)
    builder.add_node("agent_inventory", agent_inventory)
    builder.add_node("tool_capture_lead_if_ready", tool_capture_lead_if_ready)
    builder.add_node("persist_state", persist_state)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "supervisor_route")
    builder.add_conditional_edges(
        "supervisor_route",
        pick_agent,
        {
            "greeting": "agent_greeting",
            "core_capture": "agent_core_capture",
            "knowledge": "agent_knowledge",
            "inventory": "agent_inventory",
        },
    )
    builder.add_conditional_edges(
        "agent_greeting",
        after_agent,
        {
            "tool_capture_lead_if_ready": "tool_capture_lead_if_ready",
            "persist_state": "persist_state",
        },
    )
    builder.add_conditional_edges(
        "agent_core_capture",
        after_agent,
        {
            "tool_capture_lead_if_ready": "tool_capture_lead_if_ready",
            "persist_state": "persist_state",
        },
    )
    builder.add_conditional_edges(
        "agent_knowledge",
        after_agent,
        {
            "tool_capture_lead_if_ready": "tool_capture_lead_if_ready",
            "persist_state": "persist_state",
        },
    )
    builder.add_conditional_edges(
        "agent_inventory",
        after_agent,
        {
            "tool_capture_lead_if_ready": "tool_capture_lead_if_ready",
            "persist_state": "persist_state",
        },
    )
    builder.add_edge("tool_capture_lead_if_ready", "persist_state")
    builder.add_edge("persist_state", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()


async def _search_knowledge(
    *,
    knowledge_provider: KnowledgeProvider,
    tool_context_manager: ToolRequestContextManager,
    requestor: str,
    query: str,
    limit: int,
) -> list[dict[str, Any]]:
    context = tool_context_manager.get_context()
    tenant_id = context.tenant_id if context is not None else (requestor.strip() or "anonymous")
    agent_id = context.agent_id if context is not None else "default"
    search_query = str(query or "").strip()
    if not search_query:
        return []

    try:
        matches = await knowledge_provider.search(
            tenant_id=tenant_id,
            agent_id=agent_id,
            query=search_query,
            limit=max(1, int(limit or 1)),
        )
    except Exception as exc:  # pragma: no cover - defensive path
        logger.warning(
            "knowledge_search_failed tenant_id=%s agent_id=%s query=%s error=%s",
            tenant_id,
            agent_id,
            search_query[:120],
            exc,
        )
        return []

    return [
        {
            "topic": match.topic,
            "content": match.content,
            "score": float(match.score),
            "updated_at": match.updated_at.isoformat(),
        }
        for match in matches
    ]


def _append_tool(current: list[str], tool_name: str) -> list[str]:
    tools = list(current)
    if tool_name not in tools:
        tools.append(tool_name)
    return tools


def _coerce_safe_router_fallback(*, router: RouterDecision, missing_fields: list[str]) -> RouterDecision:
    if router.next_action in {"capture_lead", "handoff"}:
        return _apply_router_guardrails(router=router, missing_fields=missing_fields)
    if router.mode == "greeting":
        return _apply_router_guardrails(router=router, missing_fields=missing_fields)
    if router.intent != "unknown":
        return _apply_router_guardrails(router=router, missing_fields=missing_fields)

    return RouterDecision(
        mode="discovery",
        agent="knowledge",
        intent="unknown",
        qualification="cold",
        missing_fields=_normalize_missing_fields(missing_fields),
        next_action="ask_question",
    )


def _apply_router_guardrails(*, router: RouterDecision, missing_fields: list[str]) -> RouterDecision:
    normalized_missing = _normalize_missing_fields(router.missing_fields or missing_fields)
    candidate = router.model_copy(deep=True)
    candidate.missing_fields = normalized_missing

    if candidate.mode == "handoff":
        candidate.agent = "knowledge"
        candidate.next_action = "handoff"
        return candidate

    if candidate.mode == "greeting":
        candidate.agent = "greeting"
    elif candidate.mode == "capture_completion":
        candidate.agent = "core_capture"

    if candidate.intent == "pricing" and candidate.mode not in {"greeting", "capture_completion"}:
        candidate.agent = "inventory"

    if candidate.mode == "capture_completion":
        candidate.next_action = "capture_lead" if not normalized_missing else "ask_question"
    elif candidate.mode == "greeting":
        candidate.next_action = "ask_question"

    return candidate


def _normalize_tool_result(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return {
            "status": "error",
            "success": False,
            "missing_fields": [],
            "error": "invalid_tool_payload",
            "data": {},
        }

    status = str(raw.get("status") or "").strip().lower()
    if status == "captured":
        return {
            "status": "captured",
            "success": True,
            "missing_fields": [],
            "error": None,
            "data": {
                "lead_status": raw.get("lead_status"),
                "qualification_evidence": raw.get("qualification_evidence"),
                "crm_result": raw.get("crm_result"),
                "structured_payload": raw.get("structured_payload"),
            },
        }

    if status == "missing_fields":
        return {
            "status": "missing",
            "success": False,
            "missing_fields": _map_tool_missing_fields(raw.get("missing_critical_fields")),
            "error": None,
            "data": {
                "lead_status": raw.get("lead_status"),
                "qualification_evidence": raw.get("qualification_evidence"),
                "structured_payload": raw.get("structured_payload"),
            },
        }

    if status == "not_qualified":
        return {
            "status": "not_qualified",
            "success": False,
            "missing_fields": [],
            "error": None,
            "data": {
                "lead_status": raw.get("lead_status"),
                "qualification_evidence": raw.get("qualification_evidence"),
                "structured_payload": raw.get("structured_payload"),
            },
        }

    error = raw.get("error")
    error_text = str(error) if error is not None else "capture_lead_if_ready_failed"
    return {
        "status": "error",
        "success": False,
        "missing_fields": [],
        "error": error_text,
        "data": {},
    }


def _map_tool_missing_fields(raw_missing: Any) -> list[str]:
    mapped: list[str] = []
    if isinstance(raw_missing, list):
        for item in raw_missing:
            value = str(item or "").strip()
            if value in {"lead_data.contact_name", "contact_name"}:
                mapped.append("contact_name")
            elif value in {"lead_data.phone", "phone"}:
                mapped.append("phone")
            elif value in {"lead_data.contact_schedule", "contact_schedule"}:
                mapped.append("contact_schedule")
    return _normalize_missing_fields(mapped)


def _normalize_missing_fields(raw_missing: Any) -> list[str]:
    if not isinstance(raw_missing, list):
        return []

    found = {
        str(item or "").strip()
        for item in raw_missing
    }
    ordered_fields = ("contact_name", "phone", "contact_schedule")
    normalized: list[str] = []
    for field in ordered_fields:
        if field in found:
            normalized.append(field)

    # Guard against unexpected extras while preserving determinism.
    for item in raw_missing:
        value = str(item or "").strip()
        if value in ordered_fields:
            continue
        if value in normalized:
            continue
        if value in {"contact_name", "phone", "contact_schedule"}:
            normalized.append(value)
    return normalized


def _build_pricing_clarification_question(*, normalized_message: str, asks_period: bool) -> str:
    needs_currency_or_country = not any(
        token in normalized_message
        for token in (
            "mxn",
            "usd",
            "eur",
            "moneda",
            "pais",
            "mexico",
            "usa",
            "estados unidos",
            "colombia",
            "argentina",
            "chile",
            "peru",
            "espana",
        )
    )
    parts: list[str] = []
    if not asks_period:
        parts.append("Indique si requiere plan mensual o anual.")
    if needs_currency_or_country and not _is_pricing_request(normalized_message):
        parts.append("Comparta pais o moneda objetivo para cotizar con precision.")
    return " ".join(parts).strip()


def _is_pricing_request(normalized_message: str) -> bool:
    direct_tokens = (
        "precio",
        "precios",
        "cuanto cuesta",
        "costo",
        "cotizacion",
        "cotizar",
        "mensual",
        "anual",
    )
    if any(token in normalized_message for token in direct_tokens):
        return True
    return bool(re.search(r"\bplan(?:es)?\b", normalized_message))


def _enforce_pricing_answer(*, base_response: str, normalized_message: str) -> str:
    monthly_line = "El precio del plan mensual es 18,000 MXN + IVA al mes."
    annual_line = "No tenemos tarifa anual publicada en este momento; se cotiza por separado."

    text = str(base_response or "").strip()
    lowered = _normalize_text(text)
    has_numeric = bool(re.search(r"\d", text))
    has_currency = any(
        token in lowered
        for token in ("mxn", "usd", "eur", "peso", "pesos", "$", "iva")
    )
    has_price_figure = (("18,000" in text) or ("18000" in text) or ("mxn" in lowered) or (has_numeric and has_currency))

    wants_annual = "anual" in normalized_message
    wants_monthly = ("mensual" in normalized_message) or ("mes" in normalized_message)

    if wants_annual and wants_monthly:
        return f"{monthly_line} {annual_line}"
    if wants_annual:
        return f"{monthly_line} {annual_line}"
    if wants_monthly:
        if has_price_figure:
            return text
        return monthly_line

    if has_price_figure:
        return text
    return f"{monthly_line} Si desea, tambien puedo cotizar un esquema anual."


def _pricing_facts_conflict(snapshot_summary: str, kb_summary: str) -> bool:
    left = _normalize_text(_compact_text(snapshot_summary, max_len=400))
    right = _normalize_text(_compact_text(kb_summary, max_len=400))
    if not left or not right:
        return False
    if left == right:
        return False
    left_tokens = _content_tokens(left)
    right_tokens = _content_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    overlap = left_tokens.intersection(right_tokens)
    similarity = float(len(overlap)) / float(max(len(left_tokens), len(right_tokens)))
    return similarity <= 0.6


def _build_missing_fields_question(missing_fields: list[str]) -> str:
    labels = []
    for field in missing_fields:
        if field == "contact_name":
            labels.append("nombre de contacto")
        elif field == "phone":
            labels.append("telefono")
        elif field == "contact_schedule":
            labels.append("horario preferido de contacto")

    if not labels:
        return "Para continuar, confirme nombre, telefono y horario preferido de contacto."
    if len(labels) == 1:
        return f"Para continuar, compartame su {labels[0]}."
    if len(labels) == 2:
        return f"Para continuar, compartame su {labels[0]} y {labels[1]}."
    return "Para continuar, compartame su nombre de contacto, telefono y horario preferido de contacto."


def _compact_text(raw: str, *, max_len: int = 280) -> str:
    text = " ".join(str(raw or "").split())
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 1].rstrip()}..."


def _pick_relevant_match(query: str, matches: list[dict[str, Any]]) -> dict[str, Any] | None:
    query_tokens = _content_tokens(query)
    if not query_tokens:
        return matches[0] if matches else None

    for match in matches:
        content = f"{match.get('topic', '')} {match.get('content', '')}"
        content_tokens = _content_tokens(content)
        if not content_tokens:
            continue
        if query_tokens.intersection(content_tokens):
            return match
    return None


def _content_tokens(text: str) -> set[str]:
    normalized = _normalize_text(text)
    raw_tokens = re.findall(r"[a-z0-9]+", normalized)
    stopwords = {
        "que",
        "como",
        "para",
        "con",
        "por",
        "una",
        "unos",
        "unas",
        "este",
        "esta",
        "sobre",
        "desde",
        "dame",
        "detalle",
        "detalles",
        "quiero",
    }
    return {token for token in raw_tokens if len(token) >= 4 and token not in stopwords}


def _normalize_text(text: str) -> str:
    lowered = str(text or "").strip().lower()
    stripped = "".join(
        char
        for char in unicodedata.normalize("NFD", lowered)
        if unicodedata.category(char) != "Mn"
    )
    return re.sub(r"\s+", " ", stripped)
