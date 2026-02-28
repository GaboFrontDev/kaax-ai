from __future__ import annotations

import re
import unicodedata
from typing import Any

from pydantic import ValidationError

from app.agent.orchestration.schemas import (
    CONTACT_FIELD_ORDER,
    ConversationState,
    Intent,
    LeadState,
    MissingField,
    Mode,
    RouterDecision,
)

_SUPPORT_KEYWORDS = (
    "error",
    "no funciona",
    "soporte",
    "support",
    "ayuda",
    "bug",
    "falla",
    "problema tecnico",
    "problema",
)
_PRODUCT_KEYWORDS = (
    "que es",
    "como funciona",
    "integraciones",
    "servicios",
    "servicio",
    "funcionalidades",
)
_PRICING_KEYWORDS = (
    "precio",
    "precios",
    "plan",
    "planes",
    "cuanto cuesta",
    "cotizacion",
    "cotizar",
)
_PURCHASE_KEYWORDS = (
    "quiero demo",
    "demo",
    "contratar",
    "implementar",
    "implementacion",
    "evaluar implementacion",
    "agendar",
    "agenda",
    "llamada",
)
_COMMERCIAL_SIGNAL_KEYWORDS = (
    "demo",
    "precio",
    "precios",
    "cotizacion",
    "cotizar",
    "contratar",
    "agendar",
    "agenda",
    "llamada",
    "implementar",
    "implementacion",
    "evaluar implementacion",
)
_URGENCY_KEYWORDS = (
    "hoy",
    "urgente",
    "manana",
)
_ADVANCE_KEYWORDS = (
    "avanzar",
    "procedamos",
    "empecemos",
)
_NEED_KEYWORDS = (
    "necesito",
    "busco",
    "queremos",
    "quiero",
    "nos hace falta",
)
_CONTEXT_KEYWORDS = (
    "empresa",
    "equipo",
    "ventas",
    "clientes",
    "soporte",
    "operacion",
    "whatsapp",
)
_GREETING_EXACT = {
    "hola",
    "buenas",
    "buen dia",
    "buenos dias",
    "buenas tardes",
    "buenas noches",
    "que tal",
    "hello",
    "hi",
}

_PHONE_RE = re.compile(r"(?:\+?\d[\d\-\s\(\)]{6,}\d)")
_SCHEDULE_TIME_RE = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm|hrs|h)\b", flags=re.IGNORECASE)
_SCHEDULE_WITH_HINT_RE = re.compile(
    r"(?:horario|disponibilidad|contacto|llamarme|llamame|contactarme|agendar)\s*[:\-]?\s*([^.,;!?]{3,80})",
    flags=re.IGNORECASE,
)
_NAME_PATTERNS = (
    re.compile(
        r"\bmi nombre es\s+([a-zA-Z][a-zA-Z'\-]{1,30}(?:\s+[a-zA-Z][a-zA-Z'\-]{1,30}){0,2})(?=\s*(?:,|\.|;|$|\sy\smi\b))",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bsoy\s+([a-zA-Z][a-zA-Z'\-]{1,30}(?:\s+[a-zA-Z][a-zA-Z'\-]{1,30}){0,2})(?=\s*(?:,|\.|;|$|\sy\smi\b))",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bme llamo\s+([a-zA-Z][a-zA-Z'\-]{1,30}(?:\s+[a-zA-Z][a-zA-Z'\-]{1,30}){0,2})(?=\s*(?:,|\.|;|$|\sy\smi\b))",
        flags=re.IGNORECASE,
    ),
)

_INVALID_NAME_TERMS = {
    "kaax",
    "demo",
    "precio",
    "soporte",
    "implementar",
}

_HUMAN_HANDOFF_PATTERNS = (
    re.compile(r"\bhablar con (un|una)?\s*(asesor|agente|persona|humano|representante|ejecutivo)\b"),
    re.compile(r"\bpasame con (un|una)?\s*(asesor|agente|persona|humano|representante|ejecutivo)\b"),
    re.compile(r"\bquiero (hablar|contactar) con (un|una)?\s*(asesor|agente|persona|humano|representante|ejecutivo)\b"),
    re.compile(r"\bnecesito (un|una)?\s*(asesor|agente|persona|humano|representante|ejecutivo)\b"),
    re.compile(r"\bagente humano\b"),
    re.compile(r"\basesor humano\b"),
)

_PRICING_FOLLOWUP_KEYWORDS = (
    "mensual",
    "anual",
    "mes",
    "ano",
    "mxn",
    "usd",
    "iva",
    "moneda",
    "pais",
)


def normalize_conversation_state(raw: Any) -> ConversationState:
    if isinstance(raw, ConversationState):
        state = raw.model_copy(deep=True)
    elif isinstance(raw, dict):
        try:
            state = _parse_legacy_state(raw)
        except ValidationError:
            state = ConversationState()
    else:
        state = ConversationState()

    state.captured.contact_name = _normalize_optional_text(state.captured.contact_name)
    state.captured.phone = _normalize_optional_text(state.captured.phone)
    state.captured.contact_schedule = _normalize_optional_text(state.captured.contact_schedule)
    state.pricing_context.verified_summary = _normalize_optional_text(state.pricing_context.verified_summary)
    state.pricing_context.query = _normalize_optional_text(state.pricing_context.query)
    return state


def derive_router_and_state(
    *,
    user_message: str,
    conversation_state: ConversationState | dict[str, Any] | None,
) -> tuple[RouterDecision, ConversationState]:
    state = normalize_conversation_state(conversation_state)
    captured = state.captured.model_copy(deep=True)

    if _is_missing(captured.contact_name):
        captured.contact_name = _extract_contact_name(user_message)
    if _is_missing(captured.phone):
        captured.phone = _extract_phone(user_message)
    if _is_missing(captured.contact_schedule):
        captured.contact_schedule = _extract_contact_schedule(user_message)

    missing_fields = _compute_missing_fields(captured.model_dump())
    normalized_text = _normalize_text(user_message)
    is_greeting = _is_simple_greeting(normalized_text)
    has_support = _contains_any(normalized_text, _SUPPORT_KEYWORDS)
    has_pricing = _contains_any(normalized_text, _PRICING_KEYWORDS)
    has_purchase = _contains_any(normalized_text, _PURCHASE_KEYWORDS)
    has_product = _contains_any(normalized_text, _PRODUCT_KEYWORDS)
    has_commercial_signal = _contains_any(normalized_text, _COMMERCIAL_SIGNAL_KEYWORDS)
    asks_human = _asks_human_handoff(normalized_text)
    has_advance_verb = _contains_any(normalized_text, _ADVANCE_KEYWORDS)
    was_capture_flow = state.mode == "capture_completion"

    intent = _resolve_intent(
        has_purchase=has_purchase,
        has_pricing=has_pricing,
        has_support=has_support,
        has_product=has_product,
        has_commercial_signal=has_commercial_signal,
        previous_intent=state.lead.intent,
        has_advance_verb=has_advance_verb,
        normalized_text=normalized_text,
    )

    score = 0
    hot_signal = False
    if _contains_any(normalized_text, ("demo", "llamada", "cotizacion")):
        score += 3
        hot_signal = True
    if _contains_any(normalized_text, _URGENCY_KEYWORDS):
        score += 3
        hot_signal = True
    if has_pricing:
        score += 2
    if _contains_any(normalized_text, _NEED_KEYWORDS):
        score += 2
    if _contains_any(normalized_text, _CONTEXT_KEYWORDS):
        score += 2
    if is_greeting:
        score += 1

    if score >= 5 or (hot_signal and has_advance_verb):
        qualification = "hot"
    elif 2 <= score <= 4:
        qualification = "warm"
    else:
        qualification = "cold"

    if asks_human:
        mode: Mode = "handoff"
    elif _should_use_capture_completion(
        was_capture_flow=was_capture_flow,
        has_commercial_signal=has_commercial_signal,
        has_purchase=has_purchase,
        has_pricing=has_pricing,
        has_advance_verb=has_advance_verb,
        missing_fields=missing_fields,
    ):
        mode = "capture_completion"
    elif has_support:
        mode = "support_answer"
    elif is_greeting:
        mode = "greeting"
    else:
        mode = "discovery"

    if mode == "greeting":
        agent = "greeting"
    elif mode == "capture_completion":
        agent = "core_capture"
    elif mode == "handoff":
        agent = "knowledge"
    elif intent == "pricing":
        agent = "inventory"
    else:
        agent = "knowledge"

    if mode == "handoff":
        next_action = "handoff"
    elif mode == "capture_completion" and not missing_fields:
        next_action = "capture_lead"
    elif mode == "capture_completion" and missing_fields:
        next_action = "ask_question"
    elif mode == "greeting":
        next_action = "ask_question"
    elif intent == "unknown":
        next_action = "ask_question"
    else:
        next_action = "answer"

    candidate = {
        "mode": mode,
        "agent": agent,
        "intent": intent,
        "qualification": qualification,
        "missing_fields": missing_fields,
        "next_action": next_action,
    }

    try:
        router = RouterDecision.model_validate(candidate)
    except ValidationError:
        router = RouterDecision(
            mode="discovery",
            agent="knowledge",
            intent="unknown",
            qualification="cold",
            missing_fields=missing_fields,
            next_action="ask_question",
        )

    updated_state = state.model_copy(deep=True)
    updated_state.mode = router.mode
    updated_state.captured = captured
    updated_state.lead = LeadState(
        intent=router.intent,
        qualification=router.qualification,
        status=_derive_lead_status(
            mode=router.mode,
            missing_fields=router.missing_fields,
            previous_status=state.lead.status,
        ),
    )
    return router, updated_state


def _parse_legacy_state(raw: dict[str, Any]) -> ConversationState:
    captured_raw = raw.get("captured")
    lead_raw = raw.get("lead")
    pricing_context_raw = raw.get("pricing_context")

    legacy_intent = raw.get("intent")
    legacy_qualification = raw.get("qualification")

    lead_payload: dict[str, Any]
    if isinstance(lead_raw, dict):
        lead_payload = dict(lead_raw)
    else:
        lead_payload = {}

    if "intent" not in lead_payload and isinstance(legacy_intent, str):
        lead_payload["intent"] = legacy_intent
    if "qualification" not in lead_payload and isinstance(legacy_qualification, str):
        lead_payload["qualification"] = legacy_qualification
    if "status" not in lead_payload:
        lead_payload["status"] = "en_revision"

    payload = {
        "mode": raw.get("mode", "greeting"),
        "captured": captured_raw if isinstance(captured_raw, dict) else {},
        "lead": lead_payload,
        "pricing_context": pricing_context_raw if isinstance(pricing_context_raw, dict) else {},
    }

    state = ConversationState.model_validate(payload)
    if state.lead.status not in {"calificado", "en_revision", "no_calificado"}:
        state.lead.status = "en_revision"
    return state


def _resolve_intent(
    *,
    has_purchase: bool,
    has_pricing: bool,
    has_support: bool,
    has_product: bool,
    has_commercial_signal: bool,
    previous_intent: Intent,
    has_advance_verb: bool,
    normalized_text: str,
) -> Intent:
    if has_purchase and has_pricing:
        return "purchase_intent"
    if has_support:
        return "support"
    if has_purchase:
        return "purchase_intent"
    if has_pricing:
        return "pricing"
    if has_product:
        return "product_inquiry"

    if has_advance_verb and previous_intent in {"purchase_intent", "pricing", "product_inquiry"}:
        return previous_intent
    if has_commercial_signal and previous_intent in {"purchase_intent", "pricing"}:
        return previous_intent
    if _is_pricing_followup(normalized_text) and previous_intent in {"pricing", "purchase_intent"}:
        return previous_intent
    return "unknown"


def _should_use_capture_completion(
    *,
    was_capture_flow: bool,
    has_commercial_signal: bool,
    has_purchase: bool,
    has_pricing: bool,
    has_advance_verb: bool,
    missing_fields: list[MissingField],
) -> bool:
    if has_purchase:
        return True
    if has_commercial_signal and has_advance_verb:
        return True
    if was_capture_flow and (has_advance_verb or has_pricing or not missing_fields):
        return True
    return False


def _derive_lead_status(
    *,
    mode: Mode,
    missing_fields: list[MissingField],
    previous_status: str,
) -> str:
    if previous_status == "no_calificado":
        return previous_status
    if mode == "capture_completion" and not missing_fields:
        return "calificado"
    return "en_revision"


def _compute_missing_fields(captured: dict[str, Any]) -> list[MissingField]:
    missing: list[MissingField] = []
    for field in CONTACT_FIELD_ORDER:
        if _is_missing(captured.get(field)):
            missing.append(field)
    return missing


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _asks_human_handoff(normalized_text: str) -> bool:
    compact = " ".join(normalized_text.split())
    return any(pattern.search(compact) for pattern in _HUMAN_HANDOFF_PATTERNS)


def _is_pricing_followup(normalized_text: str) -> bool:
    return any(keyword in normalized_text for keyword in _PRICING_FOLLOWUP_KEYWORDS)


def _is_simple_greeting(normalized_text: str) -> bool:
    compact = " ".join(normalized_text.split())
    compact = re.sub(r"[^a-z0-9\s]", "", compact)
    if compact in _GREETING_EXACT:
        return True
    tokens = compact.split()
    if not tokens or len(tokens) > 3:
        return False
    return all(token in {"hola", "buenas", "dia", "dias", "tardes", "noches", "hi", "hello"} for token in tokens)


def _normalize_text(text: str) -> str:
    lowered = str(text or "").strip().lower()
    stripped = "".join(
        char
        for char in unicodedata.normalize("NFD", lowered)
        if unicodedata.category(char) != "Mn"
    )
    return re.sub(r"\s+", " ", stripped)


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def _extract_phone(message: str) -> str | None:
    match = _PHONE_RE.search(message or "")
    if match is None:
        return None

    candidate = match.group(0).strip()
    clean = re.sub(r"[^\d+]", "", candidate)
    digits = re.sub(r"\D", "", clean)
    if len(digits) < 7:
        return None
    if clean.startswith("++"):
        clean = clean[1:]
    return clean


def _extract_contact_schedule(message: str) -> str | None:
    raw = str(message or "").strip()
    if not raw:
        return None

    hinted = _SCHEDULE_WITH_HINT_RE.search(raw)
    if hinted is not None:
        value = hinted.group(1).strip()
        if value:
            return value[:80]

    timed = _SCHEDULE_TIME_RE.search(raw)
    if timed is None:
        return None

    start = max(0, timed.start() - 20)
    end = min(len(raw), timed.end() + 20)
    snippet = raw[start:end].strip(" .,:;")
    if not snippet:
        return None
    return snippet[:80]


def _extract_contact_name(message: str) -> str | None:
    raw = str(message or "")
    for pattern in _NAME_PATTERNS:
        matched = pattern.search(raw)
        if matched is None:
            continue
        candidate = _normalize_name_candidate(matched.group(1))
        if candidate is not None:
            return candidate
    return None


def _normalize_name_candidate(value: str) -> str | None:
    text = re.sub(r"\s+", " ", str(value or "").strip(" .,:;!?"))
    if not text:
        return None
    normalized = _normalize_text(text)
    if any(term in normalized for term in _INVALID_NAME_TERMS):
        return None
    return text[:60]
