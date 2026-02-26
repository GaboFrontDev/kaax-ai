import asyncio
import json
import os

import pytest
from fastapi.testclient import TestClient

from app.api.main import app

AUTH_HEADERS = {"Authorization": "Bearer dev-token"}
pytestmark = pytest.mark.skipif(
    os.getenv("CHECKPOINT_BACKEND", "memory").lower() != "postgres",
    reason="Requires CHECKPOINT_BACKEND=postgres",
)


async def _fetch_checkpoint(thread_id: str) -> dict[str, object] | None:
    import asyncpg

    connection = await asyncpg.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "55432")),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
        database=os.getenv("DB_NAME", "postgres"),
    )
    try:
        row = await connection.fetchrow(
            "SELECT state FROM agent_checkpoints WHERE thread_id = $1",
            thread_id,
        )
    finally:
        await connection.close()

    if row is None:
        return None

    state = row["state"]
    if isinstance(state, dict):
        return state
    if isinstance(state, str):
        parsed = json.loads(state)
        if isinstance(parsed, dict):
            return parsed
    return None


def test_postgres_backend_ready_and_persists_checkpoint() -> None:
    thread_id = "docker-pg-thread"

    with TestClient(app) as client:
        ready = client.get("/health/ready")
        assert ready.status_code == 200
        ready_payload = ready.json()
        assert ready_payload["status"] == "ready"
        assert ready_payload["checkpoint_backend"] == "postgres"
        assert ready_payload["lock_backend"] == "postgres"
        assert "langgraph_checkpoint_backend" in ready_payload

        response = client.post(
            "/api/agent/assist",
            headers=AUTH_HEADERS,
            json={
                "userText": "Necesito markets en US",
                "requestor": "docker-test",
                "sessionId": thread_id,
                "streamResponse": False,
            },
        )
        assert response.status_code == 200

    stored_state = asyncio.run(_fetch_checkpoint(thread_id))
    assert stored_state is not None
    assert isinstance(stored_state.get("messages"), list)
    assert len(stored_state["messages"]) >= 2
