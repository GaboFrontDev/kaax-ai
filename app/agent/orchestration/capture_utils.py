from __future__ import annotations

from typing import Any

from app.agent.orchestration.helpers import normalize_missing_fields


def normalize_tool_result(raw: Any) -> dict[str, Any]:
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
            },
        }

    if status == "missing_fields":
        return {
            "status": "missing",
            "success": False,
            "missing_fields": map_tool_missing_fields(raw.get("missing_critical_fields")),
            "error": None,
            "data": {
                "lead_status": raw.get("lead_status"),
                "qualification_evidence": raw.get("qualification_evidence"),
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
            },
        }

    return {
        "status": "error",
        "success": False,
        "missing_fields": [],
        "error": str(raw.get("error") or "capture_lead_if_ready_failed"),
        "data": {},
    }


def map_tool_missing_fields(raw_missing: Any) -> list[str]:
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
    return normalize_missing_fields(mapped)
