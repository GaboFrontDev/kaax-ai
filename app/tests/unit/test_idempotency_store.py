import asyncio

from app.memory.idempotency import InMemoryIdempotencyStore


def test_idempotency_store_replays_completed_response() -> None:
    store = InMemoryIdempotencyStore(ttl_seconds=60)

    first = asyncio.run(store.begin(thread_id="t-1", request_id="r-1"))
    asyncio.run(
        store.complete(
            thread_id="t-1",
            request_id="r-1",
            response={"response": "ok", "tools_used": [], "completion_time": 0.1},
        )
    )
    second = asyncio.run(store.begin(thread_id="t-1", request_id="r-1"))

    assert first.state == "new"
    assert second.state == "replay"
    assert second.response is not None
    assert second.response["response"] == "ok"
