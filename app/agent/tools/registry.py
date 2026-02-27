from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agent.tools.validator import ToolValidationError, validate_tool_input, validate_tool_output
from app.channels.whatsapp_meta.client import send_meta_text_message


@dataclass(slots=True)
class ToolExecutionResult:
    tool: str
    input: dict[str, Any]
    output: dict[str, Any]


class ToolRegistry:
    def __init__(
        self,
        *,
        owner_notify_enabled: bool = False,
        owner_whatsapp_number: str | None = None,
        owner_phone_number_id: str | None = None,
        whatsapp_meta_access_token: str | None = None,
        whatsapp_meta_api_version: str = "v21.0",
    ) -> None:
        self._user_preferences: dict[str, dict[str, str]] = {}
        self._owner_notify_enabled = owner_notify_enabled
        self._owner_whatsapp_number = owner_whatsapp_number
        self._owner_phone_number_id = owner_phone_number_id
        self._whatsapp_meta_access_token = whatsapp_meta_access_token
        self._whatsapp_meta_api_version = whatsapp_meta_api_version

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        return (
            "crm_upsert_quote",
            "detect_lead_capture_readiness",
            "capture_lead_if_ready",
        )

    async def execute(self, tool_name: str, payload: dict[str, Any]) -> ToolExecutionResult:
        try:
            validated = validate_tool_input(tool_name, payload)
        except ToolValidationError as exc:
            output = validate_tool_output(tool_name, {"error": str(exc)})
            return ToolExecutionResult(tool=tool_name, input={}, output=output)

        try:
            if tool_name == "crm_upsert_quote":
                output = self._crm_upsert_quote(validated)
            elif tool_name == "detect_lead_capture_readiness":
                output = self._detect_lead_capture_readiness(validated)
            elif tool_name == "capture_lead_if_ready":
                output = await self._capture_lead_if_ready(validated)
            else:
                raise ToolValidationError(f"unsupported tool {tool_name}")
        except Exception as exc:
            output = {"error": f"tool execution failed: {type(exc).__name__}: {exc}"}

        normalized_output = validate_tool_output(tool_name, output)
        return ToolExecutionResult(tool=tool_name, input=validated, output=normalized_output)

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

    async def _capture_lead_if_ready(self, payload: dict[str, Any]) -> dict[str, Any]:
        readiness = self._detect_lead_capture_readiness(payload)
        lead_status = str(readiness["lead_status"])
        missing_critical_fields = list(readiness["missing_critical_fields"])
        qualification_evidence = list(readiness["qualification_evidence"])
        structured_payload = dict(readiness["suggested_crm_payload"])

        if lead_status == "no_calificado":
            return {
                "status": "not_qualified",
                "lead_status": lead_status,
                "missing_critical_fields": missing_critical_fields,
                "qualification_evidence": qualification_evidence,
                "crm_result": None,
                "owner_notification": "skipped",
                "owner_notification_error": None,
                "structured_payload": structured_payload,
            }

        if missing_critical_fields:
            return {
                "status": "missing_fields",
                "lead_status": lead_status,
                "missing_critical_fields": missing_critical_fields,
                "qualification_evidence": qualification_evidence,
                "crm_result": None,
                "owner_notification": "skipped",
                "owner_notification_error": None,
                "structured_payload": structured_payload,
            }

        crm_result = self._crm_upsert_quote({"payload": structured_payload})
        owner_notification = "skipped"
        owner_notification_error: str | None = None

        should_notify_owner = bool(payload.get("notify_owner"))
        if should_notify_owner:
            owner_notification, owner_notification_error = await self._notify_owner_about_captured_lead(
                structured_payload=structured_payload,
            )

        return {
            "status": "captured",
            "lead_status": lead_status,
            "missing_critical_fields": [],
            "qualification_evidence": qualification_evidence,
            "crm_result": crm_result,
            "owner_notification": owner_notification,
            "owner_notification_error": owner_notification_error,
            "structured_payload": structured_payload,
        }

    async def _notify_owner_about_captured_lead(
        self,
        *,
        structured_payload: dict[str, object],
    ) -> tuple[str, str | None]:
        if not self._owner_notify_enabled:
            return "skipped", "owner notifications are disabled"

        if not (
            self._owner_whatsapp_number
            and self._owner_phone_number_id
            and self._whatsapp_meta_access_token
        ):
            return "skipped", "owner WhatsApp configuration is incomplete"

        lead_status = str(structured_payload.get("lead_status", "unknown"))
        next_action = str(structured_payload.get("next_action", "unknown"))
        evidence = structured_payload.get("qualification_evidence", [])
        evidence_text = ", ".join(str(item) for item in evidence[:4]) if isinstance(evidence, list) else str(evidence)
        message = (
            "Nuevo lead capturado por kaax.\n"
            f"Estado: {lead_status}\n"
            f"Siguiente accion: {next_action}\n"
            f"Evidencia: {evidence_text or 'N/A'}"
        )

        try:
            await send_meta_text_message(
                api_version=self._whatsapp_meta_api_version,
                phone_number_id=self._owner_phone_number_id,
                access_token=self._whatsapp_meta_access_token,
                to=self._owner_whatsapp_number,
                text=message,
            )
            return "sent", None
        except Exception as exc:
            return "failed", f"{type(exc).__name__}: {exc}"

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
