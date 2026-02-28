from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

IntentRoute = Literal["in_scope", "needs_clarification", "out_of_scope", "conversation_end"]

_DOMAIN_KEYWORDS = {
    "kaax",
    "whatsapp",
    "chat",
    "chats",
    "mensajes",
    "automatizar",
    "automatizacion",
    "automatización",
    "crm",
    "hubspot",
    "odoo",
    "zoho",
    "lead",
    "leads",
    "pipeline",
    "ventas",
    "soporte",
    "atencion",
    "atención",
    "agente",
    "agentes",
    "bot",
    "chatbot",
    "calificar",
    "calificacion",
    "calificación",
    "integracion",
    "integración",
    "servicio",
}

_OUT_OF_SCOPE_KEYWORDS = {
    "react",
    "javascript",
    "typescript",
    "python",
    "java",
    "css",
    "html",
    "programar",
    "codigo",
    "código",
    "componente",
    "algoritmo",
    "docker",
    "kubernetes",
    "sql",
    "receta",
    "clima",
    "pelicula",
    "película",
    "musica",
    "música",
}

_GENERIC_MESSAGES = {
    "hola",
    "buenas",
    "buenos dias",
    "buenos días",
    "necesito ayuda",
    "ayuda",
    "info",
}

_GREETING_MESSAGES = {
    "hola",
    "buenas",
    "buenos dias",
    "buenos días",
}

_END_MESSAGES = {
    "gracias",
    "muchas gracias",
    "ok gracias",
    "listo gracias",
    "perfecto gracias",
    "eso es todo",
    "todo bien",
    "nada mas",
    "nada más",
    "adios",
    "adiós",
    "hasta luego",
    "hasta pronto",
    "bye",
    "chao",
}

_NO_TASK_PATTERNS = (
    re.compile(r"^\s*soy de\b", flags=re.IGNORECASE),
    re.compile(r"^\s*vivo en\b", flags=re.IGNORECASE),
    re.compile(r"^\s*me llamo\b", flags=re.IGNORECASE),
)


@dataclass(frozen=True, slots=True)
class IntentDecision:
    route: IntentRoute
    confidence: float
    reason: str


def route_intent(user_text: str, *, confidence_threshold: float = 0.7) -> IntentDecision:
    text = _normalize(user_text)
    words = [word for word in text.split(" ") if word]
    word_count = len(words)

    if not text:
        return IntentDecision(route="needs_clarification", confidence=1.0, reason="empty_input")

    if text in _GENERIC_MESSAGES:
        return IntentDecision(route="needs_clarification", confidence=0.98, reason="generic_message")

    if _looks_like_conversation_end(text):
        return IntentDecision(route="conversation_end", confidence=0.99, reason="explicit_conversation_end")

    if any(pattern.search(text) for pattern in _NO_TASK_PATTERNS):
        return IntentDecision(route="needs_clarification", confidence=0.95, reason="no_task_detected")

    domain_score = _keyword_score(text, _DOMAIN_KEYWORDS)
    out_score = _keyword_score(text, _OUT_OF_SCOPE_KEYWORDS)

    if domain_score == 0 and out_score == 0:
        if word_count <= 4:
            return IntentDecision(route="needs_clarification", confidence=0.9, reason="short_ambiguous_message")
        return IntentDecision(route="needs_clarification", confidence=0.75, reason="ambiguous_message")

    if out_score > 0 and domain_score == 0:
        confidence = min(0.65 + (out_score * 0.12), 0.97)
        if confidence >= confidence_threshold:
            return IntentDecision(route="out_of_scope", confidence=confidence, reason="out_of_scope_keywords")
        return IntentDecision(route="needs_clarification", confidence=confidence, reason="low_confidence_out_of_scope")

    if domain_score > 0 and out_score == 0:
        confidence = min(0.65 + (domain_score * 0.1), 0.96)
        if confidence >= confidence_threshold:
            return IntentDecision(route="in_scope", confidence=confidence, reason="domain_keywords")
        return IntentDecision(route="needs_clarification", confidence=confidence, reason="low_confidence_in_scope")

    if domain_score >= out_score + 2:
        return IntentDecision(route="in_scope", confidence=0.78, reason="domain_dominates")
    if out_score >= domain_score + 2:
        return IntentDecision(route="out_of_scope", confidence=0.78, reason="out_of_scope_dominates")

    return IntentDecision(route="needs_clarification", confidence=0.68, reason="mixed_signals")


def build_routing_response(
    decision: IntentDecision,
    *,
    first_turn_greeting: bool = False,
) -> str:
    if decision.route == "conversation_end":
        return (
            "Perfecto, cerramos por ahora. Cuando quieras retomar la automatizacion de conversaciones "
            "de tu negocio, aqui estoy para ayudarte."
        )

    if decision.route == "out_of_scope":
        return (
            "Puedo ayudarte con kaax ai como servicio de IA para automatizar conversaciones de negocio "
            "(WhatsApp, calificacion de leads, integracion CRM y handoff humano). Tu solicitud actual parece fuera "
            "de ese alcance. Si quieres, cuentame tu flujo de atencion o ventas y validamos encaje."
        )

    return _build_needs_clarification_response(decision, first_turn_greeting=first_turn_greeting)


def is_greeting_message(user_text: str) -> bool:
    return _normalize(user_text) in _GREETING_MESSAGES


def _build_needs_clarification_response(
    decision: IntentDecision,
    *,
    first_turn_greeting: bool = False,
) -> str:
    if first_turn_greeting:
        return (
            "Hola, soy kaax ai. Ayudo a empresas a automatizar WhatsApp, calificar leads e integrar su CRM para vender y atender mejor.\n\n"
            "Para proponerte el mejor flujo, comparteme:\n"
            "- Nombre y empresa\n"
            "- Proceso a automatizar (ventas, soporte o ambos)\n"
            "- Volumen mensual de conversaciones\n"
            "- CRM actual\n"
            "- Contacto preferido (email o telefono)"
        )

    if decision.reason == "no_task_detected":
        return (
            "Perfecto, gracias por el contexto. Para ayudarte mejor, comparteme: "
            "proceso a automatizar, volumen mensual de conversaciones y CRM actual."
        )

    return (
        "Perfecto. Para continuar, comparteme: "
        "proceso a automatizar, volumen mensual de conversaciones, CRM actual y contacto preferido."
    )


def _keyword_score(text: str, keywords: set[str]) -> int:
    score = 0
    for keyword in keywords:
        if keyword in text:
            score += 1
    return score


def _normalize(value: str) -> str:
    return " ".join(value.lower().strip().split())


def _looks_like_conversation_end(text: str) -> bool:
    cleaned = re.sub(r"[^\w\sáéíóúüñ]", " ", text.lower())
    normalized = " ".join(cleaned.split())
    if not normalized:
        return False

    if normalized in _END_MESSAGES:
        return True

    if any(token in normalized for token in ("adios", "adiós", "hasta luego", "hasta pronto", "bye", "chao")):
        return True

    if "gracias" in normalized and any(
        phrase in normalized for phrase in ("eso es todo", "nada mas", "nada más", "listo", "perfecto")
    ):
        return True

    return False
