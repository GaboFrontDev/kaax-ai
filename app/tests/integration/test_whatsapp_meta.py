from dataclasses import replace

from fastapi.testclient import TestClient

from app.api.dependencies import get_agent_runtime
from app.api.main import app
from app.infra.settings import get_settings


class _FakeRuntime:
    async def invoke(self, req):  # noqa: ANN001 - runtime protocol
        return {
            "response": f"ok:{req.user_text}",
            "tools_used": [],
            "completion_time": 0.01,
            "conversation_id": req.thread_id,
            "run_id": "run-meta-test",
            "attachments": [],
        }

    async def stream(self, req):  # noqa: ANN001 - runtime protocol
        if False:
            yield req


def test_whatsapp_meta_verify_ok() -> None:
    base = get_settings()
    app.dependency_overrides[get_settings] = lambda: replace(
        base,
        whatsapp_meta_verify_token="verify-token-123",
    )
    client = TestClient(app)

    try:
        response = client.get(
            "/webhooks/whatsapp/meta",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "verify-token-123",
                "hub.challenge": "777",
            },
        )
    finally:
        app.dependency_overrides.pop(get_settings, None)

    assert response.status_code == 200
    assert response.text == "777"


def test_whatsapp_meta_verify_rejects_bad_token() -> None:
    base = get_settings()
    app.dependency_overrides[get_settings] = lambda: replace(
        base,
        whatsapp_meta_verify_token="verify-token-123",
    )
    client = TestClient(app)

    try:
        response = client.get(
            "/webhooks/whatsapp/meta",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "wrong",
                "hub.challenge": "777",
            },
        )
    finally:
        app.dependency_overrides.pop(get_settings, None)

    assert response.status_code == 403


def test_whatsapp_meta_webhook_processes_text_and_sends_reply(monkeypatch) -> None:
    sent: list[dict[str, str]] = []

    async def _fake_send(**kwargs):  # noqa: ANN003 - explicit capture
        sent.append(
            {
                "phone_number_id": kwargs["phone_number_id"],
                "to": kwargs["to"],
                "text": kwargs["text"],
            }
        )
        return {"messages": [{"id": "wamid.out"}]}

    monkeypatch.setattr("app.api.routers.whatsapp_meta.send_meta_text_message", _fake_send)

    base = get_settings()
    app.dependency_overrides[get_settings] = lambda: replace(
        base,
        whatsapp_meta_access_token="meta-access-token",
        whatsapp_meta_api_version="v21.0",
        whatsapp_meta_app_secret=None,
    )
    app.dependency_overrides[get_agent_runtime] = lambda: _FakeRuntime()
    client = TestClient(app)

    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "metadata": {"phone_number_id": "123456789012345"},
                            "messages": [
                                {
                                    "from": "5215550000001",
                                    "id": "wamid.in.1",
                                    "type": "text",
                                    "text": {"body": "hola desde meta"},
                                }
                            ],
                        },
                    }
                ]
            }
        ],
    }

    try:
        response = client.post("/webhooks/whatsapp/meta", json=payload)
    finally:
        app.dependency_overrides.pop(get_agent_runtime, None)
        app.dependency_overrides.pop(get_settings, None)

    assert response.status_code == 200
    assert response.json() == {"ok": True, "processed": 1, "sent": 1}
    assert sent == [
        {
            "phone_number_id": "123456789012345",
            "to": "5215550000001",
            "text": "ok:hola desde meta",
        }
    ]
