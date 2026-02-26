from __future__ import annotations

import httpx


async def send_meta_text_message(
    *,
    api_version: str,
    phone_number_id: str,
    access_token: str,
    to: str,
    text: str,
) -> dict[str, object]:
    body = (text or "").strip()
    if len(body) > 4096:
        body = f"{body[:4093]}..."

    url = f"https://graph.facebook.com/{api_version}/{phone_number_id}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": body or "Mensaje recibido."},
    }
    headers = {"Authorization": f"Bearer {access_token}"}

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return dict(response.json())
