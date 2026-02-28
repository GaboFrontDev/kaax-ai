from __future__ import annotations

from typing import Any


class DetectLeadCaptureReadinessTool:
    name = "detect_lead_capture_readiness"

    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        business_context = self._as_dict(payload.get("business_context"))
        whatsapp_context = self._as_dict(payload.get("whatsapp_context"))
        crm_context = self._as_dict(payload.get("crm_context"))
        agent_limits = self._as_dict(payload.get("agent_limits"))
        lead_data = self._as_dict(payload.get("lead_data"))

        missing_critical_fields: list[str] = []

        if self._is_missing(lead_data.get("need")) and self._is_missing(lead_data.get("pain_point")):
            missing_critical_fields.append("lead_data.need_or_pain_point")
        if self._is_missing(lead_data.get("timeline")):
            missing_critical_fields.append("lead_data.timeline")
        if self._is_missing(lead_data.get("buying_intent")):
            missing_critical_fields.append("lead_data.buying_intent")

        if all(
            self._is_missing(lead_data.get(field))
            for field in ("email", "phone", "contact_name", "company")
        ):
            missing_critical_fields.append("lead_data.contact_identifier")

        required_fields = crm_context.get("required_fields")
        if isinstance(required_fields, list):
            for field in required_fields:
                field_name = str(field).strip()
                if field_name and self._is_missing(lead_data.get(field_name)):
                    missing_critical_fields.append(f"lead_data.{field_name}")

        disqualify_reason = lead_data.get("disqualify_reason")
        out_of_scope = bool(lead_data.get("out_of_scope"))
        is_disqualified = out_of_scope or not self._is_missing(disqualify_reason)
        buying_intent_value = str(lead_data.get("buying_intent", "")).strip().lower()
        has_commercial_intent = buying_intent_value in {"alta", "media", "high", "medium"}

        qualification_evidence = self._build_qualification_evidence(lead_data)
        if is_disqualified:
            if not self._is_missing(disqualify_reason):
                qualification_evidence.append(f"disqualify_reason: {disqualify_reason}")
            if out_of_scope:
                qualification_evidence.append("out_of_scope: true")
        elif buying_intent_value:
            qualification_evidence.append(f"intent_signal: {buying_intent_value}")

        if is_disqualified:
            lead_status = "no_calificado"
            next_action = "cierre_cordial"
            ready_for_capture = False
        elif not self._is_missing(lead_data.get("buying_intent")) and not has_commercial_intent:
            lead_status = "no_calificado"
            next_action = "cierre_cordial"
            ready_for_capture = False
        elif missing_critical_fields:
            lead_status = "en_revision"
            next_action = "solicitar_datos_faltantes"
            ready_for_capture = False
        else:
            lead_status = "calificado"
            needs_handoff = any(
                bool(lead_data.get(flag))
                for flag in ("high_value_opportunity", "policy_sensitive", "unresolved_tool_failures")
            )
            next_action = "handoff_humano" if needs_handoff else "registro_crm"
            ready_for_capture = True

        if not qualification_evidence:
            qualification_evidence.append("Sin evidencia suficiente en lead_data")

        suggested_crm_payload: dict[str, object] = {
            "business_context": business_context,
            "whatsapp_context": whatsapp_context,
            "crm_context": crm_context,
            "agent_limits": agent_limits,
            "lead_data": lead_data,
            "lead_status": lead_status,
            "qualification_evidence": qualification_evidence,
            "next_action": next_action,
        }

        return {
            "ready_for_capture": ready_for_capture,
            "lead_status": lead_status,
            "qualification_evidence": qualification_evidence,
            "missing_critical_fields": sorted(set(missing_critical_fields)),
            "next_action": next_action,
            "suggested_crm_payload": suggested_crm_payload,
        }

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        return {}

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, (list, tuple, set, dict)):
            return len(value) == 0
        return False

    @staticmethod
    def _build_qualification_evidence(lead_data: dict[str, Any]) -> list[str]:
        evidence: list[str] = []
        for field in (
            "pain_point",
            "need",
            "timeline",
            "buying_intent",
            "decision_role",
            "budget",
            "company",
        ):
            value = lead_data.get(field)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            evidence.append(f"{field}: {value}")
        return evidence
