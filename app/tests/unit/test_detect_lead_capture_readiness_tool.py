from __future__ import annotations

import asyncio

from app.agent.tools.detect_lead_capture_readiness_tool import DetectLeadCaptureReadinessTool


def test_readiness_accepts_minimal_contact_fields_with_spanish_aliases() -> None:
    tool = DetectLeadCaptureReadinessTool()

    result = asyncio.run(
        tool.execute(
            {
                "lead_data": {
                    "nombre": "Gabriel",
                    "telefono": "+525511223344",
                    "horario": "Lunes a viernes 9:00-18:00",
                }
            }
        )
    )

    assert result["ready_for_capture"] is True
    assert result["lead_status"] == "calificado"
    assert result["missing_critical_fields"] == []
    normalized = result["suggested_crm_payload"]["lead_data"]
    assert normalized["contact_name"] == "Gabriel"
    assert normalized["phone"] == "+525511223344"
    assert normalized["contact_schedule"] == "Lunes a viernes 9:00-18:00"


def test_readiness_requests_only_missing_minimal_fields() -> None:
    tool = DetectLeadCaptureReadinessTool()

    result = asyncio.run(
        tool.execute(
            {
                "lead_data": {
                    "contact_name": "Sofia",
                    "phone": "+5215511122233",
                }
            }
        )
    )

    assert result["ready_for_capture"] is False
    assert result["lead_status"] == "en_revision"
    assert result["missing_critical_fields"] == ["lead_data.contact_schedule"]


def test_readiness_marks_not_qualified_when_out_of_scope() -> None:
    tool = DetectLeadCaptureReadinessTool()

    result = asyncio.run(
        tool.execute(
            {
                "lead_data": {
                    "contact_name": "Luis",
                    "phone": "+5215550000000",
                    "contact_schedule": "Tardes",
                    "out_of_scope": True,
                }
            }
        )
    )

    assert result["ready_for_capture"] is False
    assert result["lead_status"] == "no_calificado"
