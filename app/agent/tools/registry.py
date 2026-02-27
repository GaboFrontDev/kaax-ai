from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agent.tools.validator import ToolValidationError, validate_tool_input, validate_tool_output


@dataclass(slots=True)
class ToolExecutionResult:
    tool: str
    input: dict[str, Any]
    output: dict[str, Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._user_preferences: dict[str, dict[str, str]] = {}

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        return (
            "get_iso_country_code",
            "retrieve_markets",
            "retrieve_segments",
            "update_user_preferences",
            "crm_upsert_quote",
            "detect_lead_capture_readiness",
        )

    async def execute(self, tool_name: str, payload: dict[str, Any]) -> ToolExecutionResult:
        try:
            validated = validate_tool_input(tool_name, payload)
        except ToolValidationError as exc:
            output = validate_tool_output(tool_name, {"error": str(exc)})
            return ToolExecutionResult(tool=tool_name, input={}, output=output)

        try:
            if tool_name == "get_iso_country_code":
                output = self._get_iso_country_code(validated)
            elif tool_name == "retrieve_markets":
                output = self._retrieve_markets(validated)
            elif tool_name == "retrieve_segments":
                output = self._retrieve_segments(validated)
            elif tool_name == "update_user_preferences":
                output = self._update_user_preferences(validated)
            elif tool_name == "crm_upsert_quote":
                output = self._crm_upsert_quote(validated)
            elif tool_name == "detect_lead_capture_readiness":
                output = self._detect_lead_capture_readiness(validated)
            else:
                raise ToolValidationError(f"unsupported tool {tool_name}")
        except Exception as exc:
            output = {"error": f"tool execution failed: {type(exc).__name__}: {exc}"}

        normalized_output = validate_tool_output(tool_name, output)
        return ToolExecutionResult(tool=tool_name, input=validated, output=normalized_output)

    def _get_iso_country_code(self, payload: dict[str, Any]) -> dict[str, Any]:
        mapping = {
            "argentina": "AR",
            "chile": "CL",
            "mexico": "MX",
            "united states": "US",
            "usa": "US",
            "spain": "ES",
            "colombia": "CO",
            "peru": "PE",
        }
        normalized = str(payload["country_name"]).strip().lower()
        iso_code = mapping.get(normalized)
        if not iso_code:
            return {"error": f"country not found: {payload['country_name']}"}
        return {"iso_code": iso_code}

    def _retrieve_markets(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = str(payload["query"]).lower()
        country_code = payload.get("country_code", "US")
        limit = int(payload.get("limit", 10))
        sample = [
            {"name": "Retail", "country_code": country_code, "score": 0.92},
            {"name": "CPG", "country_code": country_code, "score": 0.84},
            {"name": "Fintech", "country_code": country_code, "score": 0.81},
            {"name": "Healthcare", "country_code": country_code, "score": 0.78},
        ]
        filtered = [item for item in sample if query in item["name"].lower() or len(query) < 4]
        return {"markets": (filtered or sample)[:limit]}

    def _retrieve_segments(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = str(payload["query"]).lower()
        limit = int(payload.get("limit", 10))
        segments = [
            {"id": "seg-1", "name": "Young Adults", "score": 0.91},
            {"id": "seg-2", "name": "Families", "score": 0.86},
            {"id": "seg-3", "name": "Professionals", "score": 0.82},
        ]
        selected = [s for s in segments if query in s["name"].lower() or len(query) < 4]
        return {"segments": (selected or segments)[:limit]}

    def _update_user_preferences(self, payload: dict[str, Any]) -> dict[str, Any]:
        email = str(payload["email"]).lower()
        prefs = {str(k): str(v) for k, v in dict(payload["preferences"]).items()}
        self._user_preferences[email] = prefs
        return {"status": "persisted", "email": email, "preferences": prefs}

    def _crm_upsert_quote(self, payload: dict[str, Any]) -> dict[str, Any]:
        quote = dict(payload["payload"])
        quote_id = str(quote.get("quote_id", "quote-temp"))
        return {"crm_id": quote_id, "status": "upserted"}

    def _detect_lead_capture_readiness(self, payload: dict[str, Any]) -> dict[str, Any]:
        business_context = self._as_dict(payload.get("business_context"))
        whatsapp_context = self._as_dict(payload.get("whatsapp_context"))
        crm_context = self._as_dict(payload.get("crm_context"))
        agent_limits = self._as_dict(payload.get("agent_limits"))
        lead_data = self._as_dict(payload.get("lead_data"))

        missing_critical_fields: list[str] = []
        missing_critical_fields.extend(
            self._missing_context_fields(
                business_context,
                "business_context",
                ("what_sells", "sales_cycle", "qualification_fields", "first_call_questions"),
            )
        )
        missing_critical_fields.extend(
            self._missing_context_fields(
                whatsapp_context,
                "whatsapp_context",
                ("brand_tone", "service_hours", "primary_language", "flow_type"),
            )
        )
        missing_critical_fields.extend(
            self._missing_context_fields(
                crm_context,
                "crm_context",
                ("crm_name", "required_fields", "qualified_pipeline_stage"),
            )
        )
        missing_critical_fields.extend(
            self._missing_context_fields(
                agent_limits,
                "agent_limits",
                ("resolves_alone", "escalation_triggers", "forbidden_statements", "disqualification_closure"),
            )
        )

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

        qualification_evidence = self._build_qualification_evidence(lead_data)
        if is_disqualified:
            if not self._is_missing(disqualify_reason):
                qualification_evidence.append(f"disqualify_reason: {disqualify_reason}")
            if out_of_scope:
                qualification_evidence.append("out_of_scope: true")

        if is_disqualified:
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

    def _missing_context_fields(
        self,
        context: dict[str, Any],
        prefix: str,
        required_fields: tuple[str, ...],
    ) -> list[str]:
        missing: list[str] = []
        for field in required_fields:
            if self._is_missing(context.get(field)):
                missing.append(f"{prefix}.{field}")
        return missing

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
