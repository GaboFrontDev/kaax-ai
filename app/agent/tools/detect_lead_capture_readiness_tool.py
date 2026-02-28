from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, ConfigDict, Field


class DetectLeadCaptureReadinessArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    business_context: dict[str, Any] = Field(default_factory=dict)
    whatsapp_context: dict[str, Any] = Field(default_factory=dict)
    crm_context: dict[str, Any] = Field(default_factory=dict)
    agent_limits: dict[str, Any] = Field(default_factory=dict)
    lead_data: dict[str, Any] = Field(default_factory=dict)


class DetectLeadCaptureReadinessTool(BaseTool):
    name: str = "detect_lead_capture_readiness"
    description: str = (
        "Detect if a lead is ready for contact capture using a minimal policy: "
        "name, phone, and preferred contact schedule."
    )
    args_schema: Optional[ArgsSchema] = DetectLeadCaptureReadinessArgs
    return_direct: bool = False

    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        business_context = self._as_dict(payload.get("business_context"))
        whatsapp_context = self._as_dict(payload.get("whatsapp_context"))
        crm_context = self._as_dict(payload.get("crm_context"))
        agent_limits = self._as_dict(payload.get("agent_limits"))
        lead_data = self._as_dict(payload.get("lead_data"))
        normalized_lead_data = dict(lead_data)

        missing_critical_fields: list[str] = []

        contact_name = self._first_present(
            lead_data,
            "contact_name",
            "name",
            "nombre",
        )
        phone = self._first_present(
            lead_data,
            "phone",
            "telefono",
            "phone_number",
            "numero",
            "numero_telefono",
            "whatsapp_number",
        )
        contact_schedule = self._first_present(
            lead_data,
            "contact_schedule",
            "horario",
            "horario_contacto",
            "best_time",
            "best_time_to_contact",
            "availability",
            "disponibilidad",
        )

        if self._is_missing(contact_name):
            missing_critical_fields.append("lead_data.contact_name")
        else:
            normalized_lead_data["contact_name"] = str(contact_name).strip()

        if self._is_missing(phone):
            missing_critical_fields.append("lead_data.phone")
        else:
            normalized_lead_data["phone"] = str(phone).strip()

        if self._is_missing(contact_schedule):
            missing_critical_fields.append("lead_data.contact_schedule")
        else:
            normalized_lead_data["contact_schedule"] = str(contact_schedule).strip()

        disqualify_reason = lead_data.get("disqualify_reason")
        out_of_scope = bool(lead_data.get("out_of_scope"))
        is_disqualified = out_of_scope or not self._is_missing(disqualify_reason)
        qualification_evidence = self._build_qualification_evidence(normalized_lead_data)
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
            next_action = "registro_crm"
            ready_for_capture = True

        if not qualification_evidence:
            qualification_evidence.append("Sin evidencia suficiente en lead_data")

        suggested_crm_payload: dict[str, object] = {
            "business_context": business_context,
            "whatsapp_context": whatsapp_context,
            "crm_context": crm_context,
            "agent_limits": agent_limits,
            "lead_data": normalized_lead_data,
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

    async def _arun(
        self,
        business_context: dict[str, Any] | None = None,
        whatsapp_context: dict[str, Any] | None = None,
        crm_context: dict[str, Any] | None = None,
        agent_limits: dict[str, Any] | None = None,
        lead_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if business_context is not None:
            payload["business_context"] = business_context
        if whatsapp_context is not None:
            payload["whatsapp_context"] = whatsapp_context
        if crm_context is not None:
            payload["crm_context"] = crm_context
        if agent_limits is not None:
            payload["agent_limits"] = agent_limits
        if lead_data is not None:
            payload["lead_data"] = lead_data
        return await self.execute(payload)

    def _run(
        self,
        business_context: dict[str, Any] | None = None,
        whatsapp_context: dict[str, Any] | None = None,
        crm_context: dict[str, Any] | None = None,
        agent_limits: dict[str, Any] | None = None,
        lead_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError("DetectLeadCaptureReadinessTool only supports async execution.")

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
            "contact_name",
            "phone",
            "contact_schedule",
        ):
            value = lead_data.get(field)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            evidence.append(f"{field}: {value}")
        return evidence

    @staticmethod
    def _first_present(data: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            value = data.get(key)
            if not DetectLeadCaptureReadinessTool._is_missing(value):
                return value
        return None
