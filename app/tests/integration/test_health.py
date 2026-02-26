from fastapi.testclient import TestClient

from app.api.main import app


def test_health_ok() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_live_ready() -> None:
    client = TestClient(app)

    live = client.get("/health/live")
    ready = client.get("/health/ready")

    assert live.status_code == 200
    assert live.json() == {"status": "ok"}
    assert ready.status_code == 200
    body = ready.json()
    assert body["status"] == "ready"
    assert body["checkpoint_backend"] == "memory"
    assert body["lock_backend"] == "memory"
    assert body["attachment_backend"] == "memory"
    assert body["message_queue_backend"] == "memory"
    assert body["langgraph_checkpoint_backend"] == "disabled"
