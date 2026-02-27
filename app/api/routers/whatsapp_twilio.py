from __future__ import annotations

import logging
from xml.sax.saxutils import escape

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from app.agent.runtime import AgentRuntime
from app.api.dependencies import get_agent_runtime, get_interaction_metrics_store
from app.channels.whatsapp_twilio.adapter import WhatsAppTwilioAdapter
from app.channels.whatsapp_twilio.webhook import validate_twilio_signature
from app.infra.settings import Settings, get_settings
from app.memory.session_manager import SessionBusyError
from app.observability.metrics import InteractionMetricsStore

router = APIRouter(tags=["whatsapp_twilio"])
_adapter = WhatsAppTwilioAdapter()
logger = logging.getLogger(__name__)


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
    metrics_store: InteractionMetricsStore = Depends(get_interaction_metrics_store),
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
    user_id = payload.get("From")
    await _record_metrics_event(
        metrics_store,
        channel="whatsapp_twilio",
        user_id=user_id,
        thread_id=req.thread_id,
        direction="inbound",
        event_type="message_inbound",
        success=True,
        run_id=payload.get("MessageSid"),
        metadata={"to": payload.get("To")},
    )
    runtime_run_id: str | None = None
    try:
        result = await runtime.invoke(req)
        runtime_run_id = str(result.get("run_id") or "") or None
        message = str(result.get("response", "Mensaje recibido."))
        await _record_metrics_event(
            metrics_store,
            channel="whatsapp_twilio",
            user_id=user_id,
            thread_id=req.thread_id,
            direction="outbound",
            event_type="message_sent",
            success=True,
            run_id=runtime_run_id,
            metadata={"to": payload.get("From")},
        )
    except SessionBusyError:
        message = "Estoy procesando tu mensaje anterior. Enseguida continuo contigo."
        await _record_metrics_event(
            metrics_store,
            channel="whatsapp_twilio",
            user_id=user_id,
            thread_id=req.thread_id,
            direction="outbound",
            event_type="message_session_busy",
            success=False,
            run_id=runtime_run_id,
            metadata={"to": payload.get("From")},
        )
    except Exception:
        logger.exception("twilio_webhook_runtime_failed")
        message = "Tuvimos un problema temporal procesando tu mensaje. Puedes intentar de nuevo en unos segundos."
        await _record_metrics_event(
            metrics_store,
            channel="whatsapp_twilio",
            user_id=user_id,
            thread_id=req.thread_id,
            direction="outbound",
            event_type="message_runtime_error",
            success=False,
            run_id=runtime_run_id,
            metadata={"to": payload.get("From")},
        )

    return Response(content=_twiml_message(message), media_type="application/xml")


async def _record_metrics_event(
    metrics_store: InteractionMetricsStore,
    *,
    channel: str,
    user_id: str | None,
    thread_id: str,
    direction: str,
    event_type: str,
    success: bool,
    run_id: str | None,
    metadata: dict[str, object] | None,
) -> None:
    try:
        await metrics_store.record_event(
            channel=channel,
            user_id=user_id,
            thread_id=thread_id,
            direction=direction,
            event_type=event_type,
            success=success,
            run_id=run_id,
            metadata=metadata,
        )
    except Exception:
        # Metrics should never break message delivery.
        pass
