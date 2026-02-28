from __future__ import annotations

from typing import Any

from app.agent.tools.crm_upsert_quote_tool import CrmUpsertQuoteTool
from app.agent.tools.detect_lead_capture_readiness_tool import DetectLeadCaptureReadinessTool
from app.channels.whatsapp_meta.client import send_meta_text_message


class CaptureLeadIfReadyTool:
    name = "capture_lead_if_ready"

    def __init__(
        self,
        *,
        crm_upsert_tool: CrmUpsertQuoteTool,
        readiness_tool: DetectLeadCaptureReadinessTool,
        owner_notify_enabled: bool = False,
        owner_whatsapp_number: str | None = None,
        owner_phone_number_id: str | None = None,
        whatsapp_meta_access_token: str | None = None,
        whatsapp_meta_api_version: str = "v21.0",
    ) -> None:
        self._crm_upsert_tool = crm_upsert_tool
        self._readiness_tool = readiness_tool
        self._owner_notify_enabled = owner_notify_enabled
        self._owner_whatsapp_number = owner_whatsapp_number
        self._owner_phone_number_id = owner_phone_number_id
        self._whatsapp_meta_access_token = whatsapp_meta_access_token
        self._whatsapp_meta_api_version = whatsapp_meta_api_version

    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        readiness = await self._readiness_tool.execute(payload)
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

        crm_result = await self._crm_upsert_tool.execute({"payload": structured_payload})
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
