from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, ConfigDict, Field

from app.agent.tools.crm_upsert_quote_tool import CrmUpsertQuoteTool
from app.agent.tools.detect_lead_capture_readiness_tool import DetectLeadCaptureReadinessTool
from app.channels.whatsapp_meta.client import send_meta_text_message


class CaptureLeadIfReadyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    business_context: dict[str, Any] = Field(default_factory=dict)
    whatsapp_context: dict[str, Any] = Field(default_factory=dict)
    crm_context: dict[str, Any] = Field(default_factory=dict)
    agent_limits: dict[str, Any] = Field(default_factory=dict)
    lead_data: dict[str, Any] = Field(default_factory=dict)
    notify_owner: bool = False


class CaptureLeadIfReadyTool(BaseTool):
    name: str = "capture_lead_if_ready"
    description: str = (
        "Capture contact leads in kaax internal CRM with minimal required fields: "
        "contact_name, phone, and contact_schedule. Optionally notify owner by WhatsApp."
    )
    args_schema: Optional[ArgsSchema] = CaptureLeadIfReadyArgs
    return_direct: bool = False
    crm_upsert_tool: CrmUpsertQuoteTool
    readiness_tool: DetectLeadCaptureReadinessTool
    owner_notify_enabled: bool = False
    owner_whatsapp_number: str | None = None
    owner_phone_number_id: str | None = None
    whatsapp_meta_access_token: str | None = None
    whatsapp_meta_api_version: str = "v21.0"
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        readiness = await self.readiness_tool.execute(payload)
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

        crm_result = await self.crm_upsert_tool.execute({"payload": structured_payload})
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

    async def _arun(
        self,
        business_context: dict[str, Any] | None = None,
        whatsapp_context: dict[str, Any] | None = None,
        crm_context: dict[str, Any] | None = None,
        agent_limits: dict[str, Any] | None = None,
        lead_data: dict[str, Any] | None = None,
        notify_owner: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"notify_owner": bool(notify_owner)}
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
        notify_owner: bool = False,
    ) -> dict[str, Any]:
        raise NotImplementedError("CaptureLeadIfReadyTool only supports async execution.")

    async def _notify_owner_about_captured_lead(
        self,
        *,
        structured_payload: dict[str, object],
    ) -> tuple[str, str | None]:
        if not self.owner_notify_enabled:
            return "skipped", "owner notifications are disabled"

        if not (
            self.owner_whatsapp_number
            and self.owner_phone_number_id
            and self.whatsapp_meta_access_token
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
                api_version=self.whatsapp_meta_api_version,
                phone_number_id=self.owner_phone_number_id,
                access_token=self.whatsapp_meta_access_token,
                to=self.owner_whatsapp_number,
                text=message,
            )
            return "sent", None
        except Exception as exc:
            return "failed", f"{type(exc).__name__}: {exc}"
