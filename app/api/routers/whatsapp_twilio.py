from __future__ import annotations

from xml.sax.saxutils import escape

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from app.agent.runtime import AgentRuntime
from app.api.dependencies import get_agent_runtime
from app.channels.whatsapp_twilio.adapter import WhatsAppTwilioAdapter
from app.channels.whatsapp_twilio.webhook import validate_twilio_signature
from app.infra.settings import Settings, get_settings
from app.memory.session_manager import SessionBusyError

router = APIRouter(tags=["whatsapp_twilio"])
_adapter = WhatsAppTwilioAdapter()


def _twiml_message(text: str) -> str:
    body = (text or "Mensaje recibido.").strip()
    if len(body) > 1500:
        body = f"{body[:1497]}..."
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"<Response><Message>{escape(body)}</Message></Response>"
    )


@router.post("/webhooks/whatsapp/twilio")
async def whatsapp_twilio_webhook(
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
    settings: Settings = Depends(get_settings),
) -> Response:
    form = await request.form()
    payload = {key: str(value) for key, value in form.items()}

    if settings.whatsapp_twilio_auth_token:
        provided_signature = request.headers.get("X-Twilio-Signature", "")
        signed_url = settings.whatsapp_twilio_webhook_url or str(request.url).split("?", 1)[0]
        is_valid = validate_twilio_signature(
            url=signed_url,
            params=payload,
            signature=provided_signature,
            auth_token=settings.whatsapp_twilio_auth_token,
        )
        if not is_valid:
            raise HTTPException(status_code=401, detail="invalid twilio signature")

    req = await _adapter.normalize_inbound(payload)
    try:
        result = await runtime.invoke(req)
        message = str(result.get("response", "Mensaje recibido."))
    except SessionBusyError:
        message = "Estoy procesando tu mensaje anterior. Enseguida continuo contigo."

    return Response(content=_twiml_message(message), media_type="application/xml")
