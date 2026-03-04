from __future__ import annotations

import re
import unicodedata
from typing import Any


def append_tool(current: list[str], tool_name: str) -> list[str]:
    tools = list(current)
    if tool_name not in tools:
        tools.append(tool_name)
    return tools


def normalize_missing_fields(raw_missing: Any) -> list[str]:
    if not isinstance(raw_missing, list):
        return []

    found = {str(item or "").strip() for item in raw_missing}
    ordered = ("contact_name", "phone", "contact_schedule")
    return [field for field in ordered if field in found]


def build_missing_fields_question(missing_fields: list[str]) -> str:
    labels = []
    for field in missing_fields:
        if field == "contact_name":
            labels.append("nombre de contacto")
        elif field == "phone":
            labels.append("teléfono")
        elif field == "contact_schedule":
            labels.append("horario preferido de contacto")

    if not labels:
        return "Para avanzar, compárteme nombre, teléfono y horario preferido de contacto."
    if len(labels) == 1:
        return f"Para avanzar, compárteme tu {labels[0]}."
    if len(labels) == 2:
        return f"Para avanzar, compárteme tu {labels[0]} y {labels[1]}."
    return "Para avanzar, compárteme nombre de contacto, teléfono y horario preferido de contacto."


def lookup_previous_summary(*, user_message: str, conversation_state: dict[str, Any]) -> str | None:
    normalized = normalize_text(user_message)
    qa_memory = conversation_state.get("qa_memory") if isinstance(conversation_state, dict) else {}
    if not isinstance(qa_memory, dict):
        return None

    factual_cache = qa_memory.get("factual_cache") if isinstance(qa_memory.get("factual_cache"), dict) else {}
    cached = factual_cache.get(normalized)
    if isinstance(cached, str) and cached.strip():
        return compact_text(cached, max_len=160)

    answered = qa_memory.get("answered_questions")
    if not isinstance(answered, list):
        return None

    best_summary: str | None = None
    best_score = 0.0
    for item in answered[-8:]:
        if not isinstance(item, dict):
            continue
        previous_q = normalize_text(str(item.get("normalized_question") or ""))
        summary = str(item.get("answer_summary") or "").strip()
        if not previous_q or not summary:
            continue
        score = similarity(normalized, previous_q)
        if score > best_score:
            best_score = score
            best_summary = summary

    if best_score >= 0.86 and best_summary:
        return compact_text(best_summary, max_len=160)
    return None


def similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0

    left_tokens = set(left.split())
    right_tokens = set(right.split())
    union = left_tokens.union(right_tokens)
    if not union:
        return 0.0

    return float(len(left_tokens.intersection(right_tokens))) / float(len(union))


def build_crm_external_key(thread_id: str, lead_data: dict[str, Any]) -> str:
    phone = str(lead_data.get("phone") or "").strip()
    name = str(lead_data.get("contact_name") or "").strip().lower()
    if phone:
        return f"{thread_id}:phone:{phone}"
    if name:
        return f"{thread_id}:name:{name}"
    return f"{thread_id}:lead"


def compact_text(raw: str, *, max_len: int) -> str:
    text = " ".join(str(raw or "").split())
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 1].rstrip()}..."


def normalize_text(text: str) -> str:
    lowered = str(text or "").strip().lower()
    stripped = "".join(
        char
        for char in unicodedata.normalize("NFD", lowered)
        if unicodedata.category(char) != "Mn"
    )
    alnum = re.sub(r"[^a-z0-9\s]", " ", stripped)
    return re.sub(r"\s+", " ", alnum).strip()
