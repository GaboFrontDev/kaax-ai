from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

from app.agent.runtime import AgentRuntime
from app.api.dependencies import get_agent_runtime
from app.channels.whatsapp_meta.adapter import WhatsAppMetaAdapter
from app.channels.whatsapp_meta.client import send_meta_text_message
from app.channels.whatsapp_meta.webhook import validate_meta_signature, verify_meta_webhook_token
from app.infra.settings import Settings, get_settings
from app.memory.session_manager import SessionBusyError
from app.observability.logging import set_correlation

logger = logging.getLogger(__name__)
router = APIRouter(tags=["whatsapp_meta"])
_adapter = WhatsAppMetaAdapter()


def _extract_text_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    extracted: list[dict[str, str]] = []
    entries = payload.get("entry", [])
    if not isinstance(entries, list):
        return extracted

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        changes = entry.get("changes", [])
        if not isinstance(changes, list):
            continue

        for change in changes:
            if not isinstance(change, dict):
                continue
            value = change.get("value", {})
            if not isinstance(value, dict):
                continue
            metadata = value.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            phone_number_id = str(metadata.get("phone_number_id", ""))

            messages = value.get("messages", [])
            if not isinstance(messages, list):
                continue

            for message in messages:
                if not isinstance(message, dict):
                    continue
                if str(message.get("type", "")) != "text":
                    continue
                text_obj = message.get("text", {})
                if not isinstance(text_obj, dict):
                    continue

                text = str(text_obj.get("body", "")).strip()
                from_number = str(message.get("from", "")).strip()
                message_id = str(message.get("id", "")).strip()
                if not text or not from_number:
                    continue

                extracted.append(
                    {
                        "from": from_number,
                        "to": phone_number_id,
                        "text": text,
                        "wa_message_id": message_id,
                    }
                )

    return extracted


@router.get("/webhooks/whatsapp/meta")
async def whatsapp_meta_verify(
    hub_mode: str = Query(alias="hub.mode"),
    hub_verify_token: str = Query(alias="hub.verify_token"),
    hub_challenge: str = Query(alias="hub.challenge"),
    settings: Settings = Depends(get_settings),
) -> PlainTextResponse:
    expected_token = settings.whatsapp_meta_verify_token
    if not expected_token:
        raise HTTPException(status_code=503, detail="meta verify token not configured")
    if hub_mode != "subscribe":
        raise HTTPException(status_code=400, detail="invalid hub.mode")
    if not verify_meta_webhook_token(hub_verify_token, expected_token):
        raise HTTPException(status_code=403, detail="invalid verify token")
    return PlainTextResponse(content=hub_challenge)


@router.post("/webhooks/whatsapp/meta")
async def whatsapp_meta_webhook(
    request: Request,
    runtime: AgentRuntime = Depends(get_agent_runtime),
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    raw_body = await request.body()

    if settings.whatsapp_meta_app_secret:
        signature = request.headers.get("x-hub-signature-256", "")
        if not validate_meta_signature(raw_body, signature, settings.whatsapp_meta_app_secret):
            raise HTTPException(status_code=401, detail="invalid meta signature")

    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid json payload") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be an object")
    if payload.get("object") != "whatsapp_business_account":
        return {"ok": True, "ignored": True}

    inbound_messages = _extract_text_messages(payload)
    processed = 0
    sent = 0

    for inbound in inbound_messages:
        req = await _adapter.normalize_inbound(inbound)
        inbound_run_id = str(inbound.get("wa_message_id") or "wa-meta-webhook")
        set_correlation(thread_id=req.thread_id, run_id=inbound_run_id)
        logger.info(
            "meta_inbound_message thread_id=%s from=%s to=%s text=%s",
            req.thread_id,
            inbound.get("from", ""),
            inbound.get("to", ""),
            req.user_text[:300],
        )
        try:
            result = await runtime.invoke(req)
            runtime_run_id = str(result.get("run_id") or inbound_run_id)
            set_correlation(thread_id=req.thread_id, run_id=runtime_run_id)
            response_text = str(result.get("response", "Mensaje recibido."))
            logger.info(
                "meta_runtime_result thread_id=%s run_id=%s tools_used=%s",
                req.thread_id,
                result.get("run_id"),
                result.get("tools_used", []),
            )
        except SessionBusyError:
            response_text = "Estoy procesando tu mensaje anterior. Enseguida continuo contigo."
        except Exception:
            logger.exception("meta_webhook_runtime_failed")
            response_text = (
                "Tuvimos un problema temporal procesando tu mensaje. "
                "Puedes intentar de nuevo en unos segundos."
            )

        processed += 1
        phone_number_id = inbound.get("to", "").strip()
        to_number = inbound.get("from", "").strip()
        if not (settings.whatsapp_meta_access_token and phone_number_id and to_number):
            continue

        try:
            meta_response = await send_meta_text_message(
                api_version=settings.whatsapp_meta_api_version,
                phone_number_id=phone_number_id,
                access_token=settings.whatsapp_meta_access_token,
                to=to_number,
                text=response_text,
            )
            sent += 1
            logger.info(
                "meta_send_ok thread_id=%s to=%s message_id=%s",
                req.thread_id,
                to_number,
                (
                    meta_response.get("messages", [{}])[0].get("id")
                    if isinstance(meta_response.get("messages"), list)
                    else None
                ),
            )
        except httpx.HTTPStatusError as exc:
            body = exc.response.text if exc.response is not None else ""
            logger.error(
                "meta_send_failed_http_status=%s body=%s",
                exc.response.status_code if exc.response is not None else "unknown",
                body,
                extra={
                    "meta_phone_number_id": phone_number_id,
                    "meta_to_number": to_number,
                },
            )
        except Exception:
            logger.exception(
                "meta_send_failed",
                extra={
                    "meta_phone_number_id": phone_number_id,
                    "meta_to_number": to_number,
                },
            )
        finally:
            set_correlation(None, None)

    return {"ok": True, "processed": processed, "sent": sent}
