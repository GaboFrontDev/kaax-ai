from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

import chainlit as cl
import httpx

logger = logging.getLogger(__name__)

API_URL = os.getenv("CHAINLIT_API_URL", "http://127.0.0.1:8200").rstrip("/")
API_TOKEN = os.getenv("CHAINLIT_API_TOKEN", "dev-token")
REQUESTOR = os.getenv("CHAINLIT_REQUESTOR", "chainlit:local")
FORMATTER = os.getenv("CHAINLIT_FORMATTER", "basic")
TOOL_CHOICE = os.getenv("CHAINLIT_TOOL_CHOICE", "auto")
PROMPT_NAME = os.getenv("CHAINLIT_PROMPT_NAME")
TIMEOUT_SECONDS = float(os.getenv("CHAINLIT_TIMEOUT_SECONDS", "120"))
MAX_FILE_SIZE_BYTES = int(os.getenv("CHAINLIT_MAX_FILE_SIZE_BYTES", str(8 * 1024 * 1024)))

SIDEBAR_TOOLS = {
    "catalog_vector_search",
    "recommendation_planner",
    "export_units",
    "export_postcodes",
}


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {API_TOKEN}",
        "Accept": "text/event-stream",
    }


def _validate_file_size(content: bytes, filename: str) -> None:
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File '{filename}' exceeds size limit: {len(content)} bytes > {MAX_FILE_SIZE_BYTES}"
        )


async def _collect_attachments(msg: cl.Message) -> list[dict[str, str]]:
    attachments: list[dict[str, str]] = []
    for element in msg.elements or []:
        path = getattr(element, "path", None)
        if not path:
            continue

        file_path = Path(path)
        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")

        content_bytes = file_path.read_bytes()
        filename = str(getattr(element, "name", file_path.name) or file_path.name)
        _validate_file_size(content_bytes, filename)

        attachments.append(
            {
                "filename": filename,
                "type": str(getattr(element, "mime", "application/octet-stream")),
                "content": base64.b64encode(content_bytes).decode("ascii"),
            }
        )

    return attachments


async def _stream_assist_events(
    *,
    user_text: str,
    thread_id: str,
    attachments: list[dict[str, str]],
) -> AsyncIterator[dict[str, Any]]:
    payload: dict[str, object] = {
        "userText": user_text,
        "requestor": REQUESTOR,
        "sessionId": thread_id,
        "streamResponse": True,
        "formatter": FORMATTER,
        "toolChoice": TOOL_CHOICE,
    }
    if PROMPT_NAME:
        payload["promptName"] = PROMPT_NAME
    if attachments:
        payload["attachments"] = attachments

    timeout = httpx.Timeout(TIMEOUT_SECONDS, read=TIMEOUT_SECONDS)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST",
            f"{API_URL}/api/agent/assist",
            headers=_headers(),
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue

                raw = line[6:]
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    yield event


async def _handle_tool_result(tool_name: str, payload: Any) -> None:
    if tool_name not in SIDEBAR_TOOLS:
        return

    data = payload if isinstance(payload, dict) else {"output": payload}
    body = json.dumps(data, ensure_ascii=True, indent=2)
    element = cl.File(
        name=f"{tool_name}_result.json",
        content=body.encode("utf-8"),
        display="side",
    )
    await cl.Message(content=f"Tool `{tool_name}` devolvio resultados.", elements=[element]).send()
    await cl.ElementSidebar.set_elements([element])


@cl.on_chat_start
async def on_chat_start() -> None:
    thread_id = str(uuid4())
    cl.user_session.set("thread_id", thread_id)
    await cl.Message(content=f"Chainlit conectado al API local. thread_id={thread_id}").send()


@cl.on_message
async def on_message(msg: cl.Message) -> None:
    thread_id = str(cl.user_session.get("thread_id") or uuid4())
    cl.user_session.set("thread_id", thread_id)

    answer = cl.Message(content="")
    await answer.send()

    has_files_attached = False
    attachments: list[dict[str, str]] = []

    try:
        if not msg.elements:
            logger.info("No files attached in this request.")
        else:
            has_files_attached = True
            attachments = await _collect_attachments(msg)
    except Exception as exc:
        await cl.Message(content=f"An error occurred while validating your files: {exc}").send()
        return

    try:
        del has_files_attached  # kept for parity with legacy flow

        async for event in _stream_assist_events(
            user_text=msg.content,
            thread_id=thread_id,
            attachments=attachments,
        ):
            event_type = event.get("type")

            if event_type == "content":
                token = str(event.get("content", ""))
                if token:
                    await answer.stream_token(token)
            elif event_type == "tool_result":
                tool_name = str(event.get("tool", "tool"))
                await _handle_tool_result(tool_name, event.get("payload"))
            elif event_type == "error":
                detail = str(event.get("content", "Unknown streaming error"))
                await answer.stream_token(f"\n\n[stream error] {detail}")

        if not answer.content:
            await answer.stream_token(
                "No pude generar una respuesta válida en este intento. Inténtalo nuevamente."
            )

        await answer.update()
    except httpx.ConnectError as exc:
        await cl.Message(
            content=(
                f"No pude conectar con el API en {API_URL}. "
                f"Verifica que `make run-api` este corriendo en ese host/puerto. "
                f"Detalle: {exc}"
            )
        ).send()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:800]
        await cl.Message(content=f"Assist HTTP {exc.response.status_code}: {detail}").send()
    except Exception as exc:  # pragma: no cover - UI integration path
        logger.exception("Error in message processing")
        await cl.Message(content=f"An error occurred while processing your request: {exc}").send()
