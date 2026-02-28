from __future__ import annotations

import logging
import os
from typing import Any

import chainlit as cl
import httpx

logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv("CHAINLIT_API_URL", "http://127.0.0.1:8200").rstrip("/")
API_TOKEN = os.getenv("CHAINLIT_API_TOKEN", "dev-token")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("CHAINLIT_API_TIMEOUT_SECONDS", "90"))


def _requestor() -> str:
    user = cl.user_session.get("user")
    if isinstance(user, str) and user.strip():
        return user.strip()
    return "local"


@cl.on_chat_start
async def on_chat_start() -> None:
    thread_id = f"chainlit:{cl.context.session.id}"
    cl.user_session.set("thread_id", thread_id)
    logger.info("chainlit_chat_start thread_id=%s", thread_id)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    thread_id = str(cl.user_session.get("thread_id") or f"chainlit:{cl.context.session.id}")

    payload: dict[str, Any] = {
        "userText": message.content,
        "requestor": f"chainlit:{_requestor()}",
        "sessionId": thread_id,
        "streamResponse": False,
    }

    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
    }
    url = f"{API_BASE_URL}/api/agent/assist"

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=payload, headers=headers)

        if response.status_code >= 400:
            detail = response.text
            try:
                body = response.json()
                detail = str(body.get("detail", body))
            except Exception:
                pass
            await cl.Message(content=f"Error API ({response.status_code}): {detail}").send()
            return

        data = response.json()
        answer = str(data.get("response", "")).strip()
        if not answer:
            answer = "No se generó respuesta del agente."

        await cl.Message(content=answer).send()
    except Exception as exc:
        logger.exception("chainlit_message_error")
        await cl.Message(content=f"No se pudo conectar con la API: {type(exc).__name__}: {exc}").send()
