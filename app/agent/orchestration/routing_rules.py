from __future__ import annotations

from difflib import SequenceMatcher
import re
import unicodedata
from typing import Any

from pydantic import ValidationError

from app.agent.orchestration.schemas import (
    CONTACT_FIELD_ORDER,
    ConversationState,
    Intent,
    MissingField,
    Qualification,
    RouterDecision,
    Stage,
    TurnFlags,
)

_FACTUAL_KEYWORDS = (
    "precio",
    "precios",
    "plan",
    "planes",
    "costo",
    "cotizacion",
    "cotizar",
    "integracion",
    "integraciones",
    "crm",
    "soporte",
    "horario",
    "horarios",
    "implementacion",
    "implementar",
    "politica",
    "politicas",
)

_BUYING_SIGNAL_KEYWORDS = (
    "demo",
    "agendar",
    "agenda",
    "llamada",
    "quiero avanzar",
    "contratar",
    "comenzar",
    "empezar",
    "contacten",
    "contactarme",
    "llamen",
)

_URGENCY_KEYWORDS = ("hoy", "urgente", "esta semana", "manana")

_DISCOVERY_USE_CASE_HINTS = (
    "lead",
    "leads",
    "prospect",
    "prospectos",
    "ventas",
    "comercial",
    "soporte",
    "atencion",
    "seguimiento",
)

_DISCOVERY_CHANNEL_HINTS = {
    "whatsapp": "whatsapp",
    "instagram": "instagram",
    "facebook": "facebook",
    "web": "web",
    "sitio": "web",
    "email": "email",
    "telefono": "telefono",
    "llamada": "telefono",
}

_PAIN_POINT_HINTS = {
    "respuesta": "tiempo de respuesta",
    "responden tarde": "tiempo de respuesta",
    "calificar": "calificacion manual",
    "seguimiento": "seguimiento inconsistente",
    "pierdo": "fuga de oportunidades",
    "no contestan": "cobertura insuficiente",
}

_GREETING_EXACT = {
    "hola",
    "buenas",
    "buen dia",
    "buenos dias",
    "buenas tardes",
    "buenas noches",
    "hello",
    "hi",
}

_PHONE_RE = re.compile(r"(?:\+?\d[\d\-\s\(\)]{6,}\d)")
_SCHEDULE_TIME_RE = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm|hrs|h)\b", flags=re.IGNORECASE)
_SCHEDULE_WITH_HINT_RE = re.compile(
    r"(?:horario|disponibilidad|contacto|llamarme|llamame|contactarme|agendar)\s*[:\-]?\s*([^.,;!?]{3,80})",
    flags=re.IGNORECASE,
)
_SCHEDULE_RANGE_RE = re.compile(
    r"\bde\s+\d{1,2}(?::\d{2})?\s*(?:a|-|hasta)\s*\d{1,2}(?::\d{2})?(?:\s*(?:entre semana|de lunes a viernes|lunes a viernes|l-v))?\b",
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
    re.compile(
        r"\b(?:mi\s+)?nombre\s+es\s+([a-zA-Z][a-zA-Z'\-]{1,30}(?:\s+[a-zA-Z][a-zA-Z'\-]{1,30}){0,2})(?=\s*(?:,|\.|;|$|\sy\smi\b))",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^\s*([a-zA-Z][a-zA-Z'\-]{1,30}(?:\s+[a-zA-Z][a-zA-Z'\-]{1,30}){0,2})\s*(?:,|;)?\s*(?:numero|telefono|tel|phone|cel)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^\s*([a-zA-Z][a-zA-Z'\-]{1,30}(?:\s+[a-zA-Z][a-zA-Z'\-]{1,30}){0,2})\s+\+?\d",
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


def normalize_conversation_state(raw: Any) -> ConversationState:
    if isinstance(raw, ConversationState):
        state = raw.model_copy(deep=True)
    elif isinstance(raw, dict):
        try:
            state = ConversationState.model_validate(raw)
        except ValidationError:
            state = _parse_legacy_state(raw)
    else:
        state = ConversationState()

    state.user_profile.name = _normalize_optional_text(state.user_profile.name)
    state.user_profile.phone = _normalize_optional_text(state.user_profile.phone)
    state.user_profile.role = _normalize_optional_text(state.user_profile.role)
    state.user_profile.company = _normalize_optional_text(state.user_profile.company)

    state.lead_data.contact_name = _normalize_optional_text(state.lead_data.contact_name)
    state.lead_data.phone = _normalize_optional_text(state.lead_data.phone)
    state.lead_data.contact_schedule = _normalize_optional_text(state.lead_data.contact_schedule)
    state.lead_data.intent = _normalize_optional_text(state.lead_data.intent)
    state.lead_data.qualification = _normalize_optional_text(state.lead_data.qualification)

    state.business_context.industry = _normalize_optional_text(state.business_context.industry)
    state.business_context.use_case = _normalize_optional_text(state.business_context.use_case)
    state.business_context.volume_estimate = _normalize_optional_text(state.business_context.volume_estimate)
    state.business_context.pain_points = _dedupe_preserve_order(state.business_context.pain_points)
    state.business_context.channels = _dedupe_preserve_order(state.business_context.channels)

    return state


def derive_router_and_state(
    *,
    user_message: str,
    conversation_state: ConversationState | dict[str, Any] | None,
) -> tuple[RouterDecision, ConversationState]:
    state = normalize_conversation_state(conversation_state)
    message = str(user_message or "")
    normalized = _normalize_text(message)

    updated = state.model_copy(deep=True)

    if _is_missing(updated.lead_data.contact_name):
        updated.lead_data.contact_name = _extract_contact_name(message)
    if _is_missing(updated.lead_data.phone):
        updated.lead_data.phone = _extract_phone(message)
    if _is_missing(updated.lead_data.contact_schedule):
        updated.lead_data.contact_schedule = _extract_contact_schedule(message)

    if _is_missing(updated.user_profile.name):
        updated.user_profile.name = updated.lead_data.contact_name
    if _is_missing(updated.user_profile.phone):
        updated.user_profile.phone = updated.lead_data.phone

    _update_business_context(updated, message, normalized)

    is_repeat = _is_repeat_question(normalized, updated)
    needs_factual_lookup = _contains_any(normalized, _FACTUAL_KEYWORDS)
    has_buying_signal = _contains_any(normalized, _BUYING_SIGNAL_KEYWORDS)

    intent = _resolve_intent(
        normalized=normalized,
        has_buying_signal=has_buying_signal,
        needs_factual_lookup=needs_factual_lookup,
        has_discovery_context=_has_discovery_context(updated),
        previous_intent=updated.sales.intent,
    )
    qualification = _resolve_qualification(
        normalized=normalized,
        has_buying_signal=has_buying_signal,
        needs_factual_lookup=needs_factual_lookup,
        has_discovery_context=_has_discovery_context(updated),
    )

    missing_fields = _compute_missing_fields(updated.lead_data.model_dump())
    lead_capture_ready = has_buying_signal and (
        _has_discovery_context(updated)
        or updated.stage in {"lead_capture", "demo_cta"}
    )

    stage = _resolve_stage(
        previous_stage=updated.stage,
        normalized=normalized,
        has_buying_signal=has_buying_signal,
        has_discovery_context=_has_discovery_context(updated),
        missing_fields=missing_fields,
    )

    route = "discovery_value"
    if is_repeat:
        route = "repeat_handler"
    elif needs_factual_lookup:
        route = "memory_lookup"

    next_action: str
    if lead_capture_ready and not missing_fields:
        next_action = "capture_lead"
    elif lead_capture_ready and missing_fields:
        next_action = "ask_question"
    elif stage == "demo_cta":
        next_action = "demo_cta"
    elif route in {"repeat_handler", "memory_lookup"}:
        next_action = "answer"
    else:
        next_action = "ask_question"

    flags = TurnFlags(
        is_repeat_question=is_repeat,
        needs_factual_lookup=needs_factual_lookup,
        has_buying_signal=has_buying_signal,
        lead_capture_ready=lead_capture_ready,
    )

    updated.stage = stage
    updated.sales.intent = intent
    updated.sales.qualification = qualification
    updated.flags = flags
    updated.lead_data.intent = intent
    updated.lead_data.qualification = qualification

    if has_buying_signal:
        signal = _first_detected_signal(normalized)
        if signal and signal not in updated.sales.buying_signals:
            updated.sales.buying_signals.append(signal)

    router = RouterDecision(
        route=route,
        stage=stage,
        intent=intent,
        qualification=qualification,
        missing_fields=missing_fields,
        next_action=next_action,  # type: ignore[arg-type]
        flags=flags,
    )
    return router, updated


def _parse_legacy_state(raw: dict[str, Any]) -> ConversationState:
    captured = raw.get("captured") if isinstance(raw.get("captured"), dict) else {}
    lead = raw.get("lead") if isinstance(raw.get("lead"), dict) else {}
    mode = str(raw.get("mode") or "").strip().lower()

    legacy_intent = str(lead.get("intent") or raw.get("intent") or "unknown")
    mapped_intent: Intent
    if legacy_intent in {"purchase_intent"}:
        mapped_intent = "demo_requested"
    elif legacy_intent in {"pricing", "product_inquiry"}:
        mapped_intent = "interested"
    elif legacy_intent in {"support"}:
        mapped_intent = "exploring"
    else:
        mapped_intent = "unknown"

    legacy_qualification = str(lead.get("qualification") or raw.get("qualification") or "cold")
    mapped_qualification: Qualification = "cold"
    if legacy_qualification in {"warm", "hot"}:
        mapped_qualification = legacy_qualification  # type: ignore[assignment]

    stage_map: dict[str, Stage] = {
        "greeting": "greeting",
        "capture_completion": "lead_capture",
        "support_answer": "discovery",
        "handoff": "demo_cta",
        "discovery": "discovery",
    }

    payload = {
        "stage": stage_map.get(mode, "discovery"),
        "user_profile": {
            "name": captured.get("contact_name"),
            "phone": captured.get("phone"),
        },
        "business_context": {},
        "sales": {
            "intent": mapped_intent,
            "qualification": mapped_qualification,
            "buying_signals": [],
            "objections": [],
        },
        "lead_data": {
            "contact_name": captured.get("contact_name"),
            "phone": captured.get("phone"),
            "contact_schedule": captured.get("contact_schedule"),
            "intent": mapped_intent,
            "qualification": mapped_qualification,
        },
        "qa_memory": {
            "answered_questions": [],
            "factual_cache": {},
        },
        "tooling": {
            "last_memory_route_mode": None,
            "last_capture_result": None,
        },
        "flags": {
            "is_repeat_question": False,
            "needs_factual_lookup": False,
            "has_buying_signal": False,
            "lead_capture_ready": False,
        },
    }

    try:
        return ConversationState.model_validate(payload)
    except ValidationError:
        return ConversationState()


def _resolve_stage(
    *,
    previous_stage: Stage,
    normalized: str,
    has_buying_signal: bool,
    has_discovery_context: bool,
    missing_fields: list[MissingField],
) -> Stage:
    if has_buying_signal:
        return "lead_capture"

    if previous_stage == "greeting":
        if _is_simple_greeting(normalized):
            return "greeting"
        return "discovery"

    if previous_stage in {"discovery", "greeting"} and has_discovery_context:
        return "opportunity"

    if previous_stage == "opportunity" and has_discovery_context:
        return "value_mapping"

    if previous_stage == "lead_capture" and not missing_fields:
        return "demo_cta"

    return previous_stage


def _resolve_intent(
    *,
    normalized: str,
    has_buying_signal: bool,
    needs_factual_lookup: bool,
    has_discovery_context: bool,
    previous_intent: Intent,
) -> Intent:
    if has_buying_signal:
        return "demo_requested"
    if needs_factual_lookup or has_discovery_context:
        return "interested"
    if _is_simple_greeting(normalized) and previous_intent == "unknown":
        return "exploring"
    if previous_intent != "unknown":
        return previous_intent
    return "exploring"


def _resolve_qualification(
    *,
    normalized: str,
    has_buying_signal: bool,
    needs_factual_lookup: bool,
    has_discovery_context: bool,
) -> Qualification:
    score = 0
    if has_buying_signal:
        score += 3
    if _contains_any(normalized, _URGENCY_KEYWORDS):
        score += 2
    if needs_factual_lookup:
        score += 1
    if has_discovery_context:
        score += 1
    if _contains_any(normalized, ("equipo", "empresa", "clientes", "volumen", "pipeline")):
        score += 1

    if score >= 5:
        return "hot"
    if score >= 3:
        return "warm"
    return "cold"


def _update_business_context(state: ConversationState, message: str, normalized: str) -> None:
    if _is_missing(state.business_context.use_case):
        if _contains_any(normalized, _DISCOVERY_USE_CASE_HINTS):
            state.business_context.use_case = _compact_text(message, max_len=90)

    for token, canonical in _DISCOVERY_CHANNEL_HINTS.items():
        if token in normalized and canonical not in state.business_context.channels:
            state.business_context.channels.append(canonical)

    for token, label in _PAIN_POINT_HINTS.items():
        if token in normalized and label not in state.business_context.pain_points:
            state.business_context.pain_points.append(label)

    if _is_missing(state.business_context.industry):
        industry = _extract_industry_hint(message)
        if industry:
            state.business_context.industry = industry


def _extract_industry_hint(message: str) -> str | None:
    raw = str(message or "")
    pattern = re.compile(r"\b(?:somos|soy)\s+(?:una|un)?\s*([a-zA-Z\s]{4,40})", flags=re.IGNORECASE)
    match = pattern.search(raw)
    if match is None:
        return None
    return _compact_text(match.group(1).strip(" .,:;"), max_len=40)


def _is_repeat_question(normalized_message: str, state: ConversationState) -> bool:
    if not normalized_message:
        return False

    if normalized_message in state.qa_memory.factual_cache:
        return True

    for previous in state.qa_memory.answered_questions[-8:]:
        previous_q = _normalize_text(previous.normalized_question)
        if not previous_q:
            continue
        if normalized_message == previous_q:
            return True

        ratio = SequenceMatcher(a=normalized_message, b=previous_q).ratio()
        if ratio >= 0.86:
            return True

        shared = _token_overlap(normalized_message, previous_q)
        if shared >= 0.8 and min(len(normalized_message), len(previous_q)) >= 14:
            return True
    return False


def _token_overlap(left: str, right: str) -> float:
    left_tokens = {token for token in re.findall(r"[a-z0-9]+", left) if len(token) > 2}
    right_tokens = {token for token in re.findall(r"[a-z0-9]+", right) if len(token) > 2}
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = left_tokens.intersection(right_tokens)
    union = left_tokens.union(right_tokens)
    if not union:
        return 0.0
    return float(len(intersection)) / float(len(union))


def _first_detected_signal(normalized: str) -> str | None:
    for keyword in _BUYING_SIGNAL_KEYWORDS:
        if keyword in normalized:
            return keyword
    return None


def _compute_missing_fields(captured: dict[str, Any]) -> list[MissingField]:
    missing: list[MissingField] = []
    for field in CONTACT_FIELD_ORDER:
        if _is_missing(captured.get(field)):
            missing.append(field)
    return missing


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _has_discovery_context(state: ConversationState) -> bool:
    business = state.business_context
    return bool(
        (business.use_case and business.use_case.strip())
        or business.pain_points
        or business.channels
        or (business.industry and business.industry.strip())
    )


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

    range_match = _SCHEDULE_RANGE_RE.search(raw)
    if range_match is not None:
        value = range_match.group(0).strip(" .,:;")
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


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_text(value)
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(value.strip())
    return unique


def _compact_text(raw: str, *, max_len: int) -> str:
    text = " ".join(str(raw or "").split())
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 1].rstrip()}..."
