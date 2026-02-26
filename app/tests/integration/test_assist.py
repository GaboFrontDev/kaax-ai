import asyncio
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.dependencies import get_agent_runtime, get_session_manager
from app.memory.session_manager import SessionManager


AUTH_HEADERS = {"Authorization": "Bearer dev-token"}


def test_assist_requires_auth() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/agent/assist",
        json={"userText": "hola", "requestor": "test", "streamResponse": False},
    )

    assert response.status_code == 401


def test_assist_sync_response() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/agent/assist",
        headers=AUTH_HEADERS,
        json={"userText": "Necesito markets en US", "requestor": "test", "streamResponse": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["response"], str)
    assert "run_id" in payload
    assert payload["conversation_id"] == "test:default"


def test_assist_stream_response() -> None:
    client = TestClient(app)

    with client.stream(
        "POST",
        "/api/agent/assist",
        headers=AUTH_HEADERS,
        json={
            "userText": "Dame el ISO de Mexico",
            "requestor": "test",
            "streamResponse": True,
            "sessionId": "thread-1",
        },
    ) as response:
        body = "".join(chunk for chunk in response.iter_text())

    assert response.status_code == 200
    assert "event: message" in body
    assert '"type": "complete"' in body


def test_assist_idempotent_replay_with_request_id() -> None:
    client = TestClient(app)
    headers = {**AUTH_HEADERS, "X-Request-Id": "req-idem-1"}
    body = {
        "userText": "Necesito markets en US",
        "requestor": "test-idempotency",
        "sessionId": "thread-idem-1",
        "streamResponse": False,
    }

    first = client.post("/api/agent/assist", headers=headers, json=body)
    second = client.post("/api/agent/assist", headers=headers, json=body)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json() == first.json()


def test_assist_concurrent_same_thread_returns_409() -> None:
    class SlowLockedRuntime:
        def __init__(self, session_manager: SessionManager) -> None:
            self._session_manager = session_manager

        async def invoke(self, req):  # noqa: ANN001 - runtime protocol
            async with self._session_manager.session_lock(req.thread_id):
                await asyncio.sleep(0.15)
                return {
                    "response": "ok",
                    "tools_used": [],
                    "completion_time": 0.15,
                    "conversation_id": req.thread_id,
                    "run_id": "run-test",
                    "attachments": [],
                }

        async def stream(self, req):  # noqa: ANN001 - runtime protocol
            if False:
                yield req

    app.dependency_overrides[get_agent_runtime] = lambda: SlowLockedRuntime(get_session_manager())

    body = {
        "userText": "hola",
        "requestor": "test-concurrency",
        "sessionId": f"thread-concurrency-{uuid4()}",
        "streamResponse": False,
    }

    async def _run() -> list[httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            gate = asyncio.Event()

            async def _call() -> httpx.Response:
                await gate.wait()
                return await client.post("/api/agent/assist", headers=AUTH_HEADERS, json=body)

            tasks = [asyncio.create_task(_call()), asyncio.create_task(_call())]
            await asyncio.sleep(0)
            gate.set()
            return await asyncio.gather(*tasks)

    try:
        responses = asyncio.run(_run())
    finally:
        app.dependency_overrides.pop(get_agent_runtime, None)

    statuses = sorted(response.status_code for response in responses)
    assert statuses == [200, 409]
    conflict = next(response for response in responses if response.status_code == 409)
    assert "session lock unavailable" in conflict.text
