from __future__ import annotations

import re
import unicodedata
from typing import Any


_SECTION_KEYS = (
    "business_context",
    "whatsapp_context",
    "crm_context",
    "agent_limits",
    "lead_data",
)

_LIST_LIKE_FIELDS = {
    "qualification_fields",
    "first_call_questions",
    "required_fields",
    "resolves_alone",
    "escalation_triggers",
    "forbidden_statements",
}

_CAPTURE_REQUEST_PATTERNS = (
    "registrar un lead",
    "registralo en crm",
    "registralo ahora en crm",
    "registrarlo en crm",
    "capturalo en crm",
    "capturalo",
    "capturalo ahora",
    "capturarlo",
)

_AFFIRMATIVE_CAPTURE_PATTERNS = (
    "si registralo",
    "si, registralo",
    "si registralo ahora",
    "si capturalo",
    "si, capturalo",
    "hazlo",
    "adelante",
    "registralo ahora",
    "capturalo ahora",
)

_DEFAULT_BUSINESS_CONTEXT: dict[str, Any] = {
    "what_sells": "Automatizacion de chats para ventas y soporte",
    "sales_cycle": "30-45 dias",
    "qualification_fields": ["company", "budget", "timeline", "decision_role", "email"],
    "first_call_questions": ["caso de uso", "equipo", "urgencia"],
}

_DEFAULT_WHATSAPP_CONTEXT: dict[str, Any] = {
    "brand_tone": "consultivo",
    "service_hours": "Lun-Vie 9-18",
    "primary_language": "es",
    "flow_type": "inbound_outbound",
}

_DEFAULT_CRM_CONTEXT: dict[str, Any] = {
    "crm_name": "HubSpot",
    "required_fields": ["email", "company", "timeline"],
    "qualified_pipeline_stage": "SQL",
}

_DEFAULT_AGENT_LIMITS: dict[str, Any] = {
    "resolves_alone": ["FAQ", "descubrimiento inicial"],
    "escalation_triggers": ["alto_valor"],
    "forbidden_statements": ["promesas sin confirmacion"],
    "disqualification_closure": "cierre cordial y recontacto",
}

_COMMERCIAL_TERMS = (
    "contratar",
    "cotizacion",
    "cotizar",
    "presupuesto",
    "precio",
    "propuesta",
    "demo",
    "implementar",
    "implementacion",
    "comprar",
    "adquirir",
    "plan",
    "licencia",
)

_DOMAIN_TERMS = (
    "whatsapp",
    "crm",
    "hubspot",
    "automatiz",
    "lead",
    "leads",
    "ventas",
    "soporte",
    "chat",
    "chatbot",
    "atencion",
    "atencion al cliente",
)

_NEGATIVE_TERMS = (
    "no me interesa",
    "no quiero contratar",
    "solo curiosidad",
    "solo queria preguntar",
)

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", flags=re.IGNORECASE)
_PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{7,}\d")
_BUDGET_RE = re.compile(
    r"\b(?:presupuesto|budget)\s*(?:de|aprox|aproximado|:)?\s*([0-9][0-9.,\s]*(?:usd|mxn|eur|us\$|\$)?)",
    flags=re.IGNORECASE,
)
_COMPANY_PATTERNS = (
    re.compile(r"\b(?:empresa|compania|compañia)\s*(?:es|:)?\s*([A-Za-z0-9 .&_-]{2,})", flags=re.IGNORECASE),
    re.compile(r"\btrabajo en\s+([A-Za-z0-9 .&_-]{2,})", flags=re.IGNORECASE),
    re.compile(r"\bsoy\s+[A-Za-zÁÉÍÓÚáéíóúÑñ ]+\s+de\s+([A-Za-z0-9 .&_-]{2,})", flags=re.IGNORECASE),
    re.compile(r"\bsoy de\s+([A-Za-z0-9 .&_-]{2,})", flags=re.IGNORECASE),
)
_NAME_PATTERNS = (
    re.compile(r"\bme llamo\s+([A-Za-zÁÉÍÓÚáéíóúÑñ ]{2,})", flags=re.IGNORECASE),
    re.compile(r"\bmi nombre es\s+([A-Za-zÁÉÍÓÚáéíóúÑñ ]{2,})", flags=re.IGNORECASE),
)
_TIMELINE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\beste trimestre\b", flags=re.IGNORECASE), "este trimestre"),
    (re.compile(r"\beste mes\b", flags=re.IGNORECASE), "este mes"),
    (re.compile(r"\beste ano\b|\beste año\b", flags=re.IGNORECASE), "este ano"),
    (re.compile(r"\ben [0-9]{1,2}\s*(dias|días|semanas|meses)\b", flags=re.IGNORECASE), "corto plazo"),
    (re.compile(r"\burgente\b|\blo antes posible\b", flags=re.IGNORECASE), "urgente"),
)
_HIGH_INTENT_PATTERNS = (
    re.compile(r"\bquiero contratar\b", flags=re.IGNORECASE),
    re.compile(r"\bme interesa contratar\b", flags=re.IGNORECASE),
    re.compile(r"\bquiero una demo\b", flags=re.IGNORECASE),
    re.compile(r"\bquiero cotizar\b", flags=re.IGNORECASE),
    re.compile(r"\bnecesito una propuesta\b", flags=re.IGNORECASE),
)
_NEED_PATTERNS = (
    re.compile(r"\b(?:necesito|quiero|busco|me interesa)\s+(.+)", flags=re.IGNORECASE),
)
_PAIN_PATTERNS = (
    re.compile(r"\b(?:problema|dolor|pain point|cuello de botella)\s*[:\-]?\s*(.+)", flags=re.IGNORECASE),
    re.compile(r"\btiempos? de respuesta (?:altos|lentos)\b", flags=re.IGNORECASE),
    re.compile(r"\bno damos abasto\b", flags=re.IGNORECASE),
)
_DECISION_ROLE_PATTERNS = (
    re.compile(r"\b(?:soy|rol|cargo)\s*(?:de|:)?\s*(director[a-z ]+|gerente[a-z ]+|founder|ceo|coo|cto)\b", flags=re.IGNORECASE),
)


def parse_lead_payload_from_text(text: str) -> dict[str, Any] | None:
    sections: dict[str, dict[str, Any]] = {key: {} for key in _SECTION_KEYS}
    current_section: str | None = None

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        section_match = re.match(r"^([a-z_]+)\s*:\s*$", line.lower())
        if section_match:
            candidate = section_match.group(1)
            if candidate in sections:
                current_section = candidate
                continue

        if current_section is None:
            continue

        item = _strip_bullet(line)
        if ":" not in item:
            continue

        key, value = item.split(":", 1)
        normalized_key = key.strip()
        normalized_value = value.strip()
        if not normalized_key:
            continue

        sections[current_section][normalized_key] = _parse_value(normalized_key, normalized_value)

    non_empty_sections = {k: v for k, v in sections.items() if v}
    if "lead_data" not in non_empty_sections:
        return None
    if len(non_empty_sections) < 3:
        return None

    return non_empty_sections


def is_capture_request(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(pattern in normalized for pattern in _CAPTURE_REQUEST_PATTERNS)


def is_affirmative_capture(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(pattern in normalized for pattern in _AFFIRMATIVE_CAPTURE_PATTERNS)


def build_capture_response(output: dict[str, Any]) -> str:
    status = str(output.get("status", "")).strip().lower()
    if status == "captured":
        return "Gracias. Pronto estaremos en contacto contigo para continuar con tu solicitud."

    if status == "missing_fields":
        missing = output.get("missing_critical_fields")
        if isinstance(missing, list) and missing:
            missing_text = ", ".join(str(item) for item in missing)
            return f"Para avanzar me faltan estos datos: {missing_text}."
        return "Para avanzar necesito algunos datos clave adicionales."

    if status == "not_qualified":
        return "Gracias por la informacion. Por ahora no parece un caso calificado, pero podemos retomarlo cuando quieras."

    error = output.get("error")
    if isinstance(error, str) and error:
        return f"No pude registrar el lead por ahora: {error}"

    return "No pude confirmar el registro del lead en este momento."


def build_conversational_lead_payload(messages: list[dict[str, Any]], *, latest_user_text: str = "") -> dict[str, Any] | None:
    user_messages = _collect_user_messages(messages)
    if latest_user_text.strip():
        user_messages.append(latest_user_text.strip())
    if not user_messages:
        return None

    corpus = "\n".join(user_messages)
    normalized = _normalize_text(corpus)
    if not _has_commercial_behavior(normalized):
        return None

    lead_data = _extract_lead_data(corpus)
    if not lead_data:
        return None

    return {
        "business_context": dict(_DEFAULT_BUSINESS_CONTEXT),
        "whatsapp_context": dict(_DEFAULT_WHATSAPP_CONTEXT),
        "crm_context": dict(_DEFAULT_CRM_CONTEXT),
        "agent_limits": dict(_DEFAULT_AGENT_LIMITS),
        "lead_data": lead_data,
    }


def _strip_bullet(value: str) -> str:
    if value.startswith("-"):
        return value[1:].strip()
    if value.startswith("*"):
        return value[1:].strip()
    if value.startswith("•"):
        return value[1:].strip()
    return value


def _parse_value(key: str, value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    if key in _LIST_LIKE_FIELDS and "," in value:
        return [item.strip() for item in value.split(",") if item.strip()]

    return value


def _normalize_text(value: str) -> str:
    base = (value or "").strip().lower()
    normalized = unicodedata.normalize("NFD", base)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _collect_user_messages(messages: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).lower() != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            lines.append(content.strip())
    return lines[-20:]


def _has_commercial_behavior(normalized_text: str) -> bool:
    if any(term in normalized_text for term in _NEGATIVE_TERMS):
        return False

    high_intent = any(pattern.search(normalized_text) for pattern in _HIGH_INTENT_PATTERNS)
    commercial_hits = sum(1 for term in _COMMERCIAL_TERMS if term in normalized_text)
    domain_hits = sum(1 for term in _DOMAIN_TERMS if term in normalized_text)

    return high_intent or (commercial_hits >= 2 and domain_hits >= 1) or (commercial_hits >= 1 and domain_hits >= 2)


def _extract_lead_data(corpus: str) -> dict[str, Any]:
    lead_data: dict[str, Any] = {}

    email = _extract_first(_EMAIL_RE, corpus)
    if email:
        lead_data["email"] = email

    phone = _extract_first(_PHONE_RE, corpus)
    if phone:
        lead_data["phone"] = _normalize_phone(phone)

    company = _extract_with_patterns(_COMPANY_PATTERNS, corpus)
    if company:
        lead_data["company"] = company

    contact_name = _extract_with_patterns(_NAME_PATTERNS, corpus)
    if contact_name:
        lead_data["contact_name"] = contact_name

    timeline = _extract_timeline(corpus)
    if timeline:
        lead_data["timeline"] = timeline

    budget = _extract_budget(corpus)
    if budget:
        lead_data["budget"] = budget

    need = _extract_need(corpus)
    if need:
        lead_data["need"] = need

    pain_point = _extract_pain_point(corpus)
    if pain_point:
        lead_data["pain_point"] = pain_point

    decision_role = _extract_with_patterns(_DECISION_ROLE_PATTERNS, corpus)
    if decision_role:
        lead_data["decision_role"] = decision_role

    buying_intent = _infer_buying_intent(corpus)
    if buying_intent:
        lead_data["buying_intent"] = buying_intent

    high_value = _infer_high_value(lead_data)
    if high_value:
        lead_data["high_value_opportunity"] = True

    return lead_data


def _extract_first(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    if not match:
        return None
    if match.groups():
        return str(match.group(1)).strip()
    return str(match.group(0)).strip()


def _extract_with_patterns(patterns: tuple[re.Pattern[str], ...], text: str) -> str | None:
    for pattern in patterns:
        value = _extract_first(pattern, text)
        if value:
            return value.strip(" .,:;")
    return None


def _extract_timeline(text: str) -> str | None:
    for pattern, label in _TIMELINE_PATTERNS:
        if pattern.search(text):
            return label
    return None


def _extract_budget(text: str) -> str | None:
    value = _extract_first(_BUDGET_RE, text)
    if not value:
        return None
    return " ".join(value.split())


def _extract_need(text: str) -> str | None:
    for pattern in _NEED_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        candidate = match.group(1).strip()
        if len(candidate) < 8:
            continue
        return candidate[:220]
    return None


def _extract_pain_point(text: str) -> str | None:
    for pattern in _PAIN_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        if match.groups():
            candidate = match.group(1).strip()
            if candidate:
                return candidate[:220]
            continue
        return match.group(0).strip()
    return None


def _infer_buying_intent(text: str) -> str | None:
    normalized = _normalize_text(text)
    if any(pattern.search(normalized) for pattern in _HIGH_INTENT_PATTERNS):
        return "alta"

    medium_terms = ("precio", "presupuesto", "cotizar", "propuesta", "demo")
    if any(term in normalized for term in medium_terms):
        return "media"
    return None


def _infer_high_value(lead_data: dict[str, Any]) -> bool:
    if str(lead_data.get("buying_intent", "")).lower() == "alta":
        return True
    if lead_data.get("budget"):
        return True
    return False


def _normalize_phone(value: str) -> str:
    raw = value.strip()
    has_plus = raw.startswith("+")
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return raw
    return f"+{digits}" if has_plus else digits
