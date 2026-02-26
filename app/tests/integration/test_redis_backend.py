import os

import pytest
from fastapi.testclient import TestClient

from app.api.main import app

AUTH_HEADERS = {"Authorization": "Bearer dev-token"}
pytestmark = pytest.mark.skipif(
    os.getenv("ATTACHMENT_BACKEND", "memory").lower() != "redis",
    reason="Requires ATTACHMENT_BACKEND=redis",
)


def test_redis_backend_ready_and_attachment_roundtrip() -> None:
    thread_id = "docker-redis-thread"

    with TestClient(app) as client:
        ready = client.get("/health/ready")
        assert ready.status_code == 200
        ready_payload = ready.json()
        assert ready_payload["status"] == "ready"
        assert ready_payload["attachment_backend"] == "redis"

        first = client.post(
            "/api/agent/assist",
            headers=AUTH_HEADERS,
            json={
                "userText": "hola con adjunto",
                "requestor": "redis-test",
                "sessionId": thread_id,
                "streamResponse": False,
                "attachments": [
                    {
                        "filename": "brief.txt",
                        "content": "aGVsbG8=",
                        "type": "text/plain",
                    }
                ],
            },
        )
        assert first.status_code == 200

        second = client.post(
            "/api/agent/assist",
            headers=AUTH_HEADERS,
            json={
                "userText": "segunda vuelta",
                "requestor": "redis-test",
                "sessionId": thread_id,
                "streamResponse": False,
            },
        )
        assert second.status_code == 200
        attachments = second.json().get("attachments") or []
        assert any(item.get("filename") == "brief.txt" for item in attachments)
