import asyncio

from app.observability.metrics import InMemoryMetrics


def test_inmemory_interaction_metrics_summary() -> None:
    store = InMemoryMetrics()

    async def _run() -> dict[str, object]:
        await store.record_event(
            channel="whatsapp_meta",
            user_id="5213311111111",
            thread_id="wa-meta:100:5213311111111",
            direction="inbound",
            event_type="message_inbound",
            success=True,
            run_id="run-1",
            metadata=None,
        )
        await store.record_event(
            channel="whatsapp_meta",
            user_id="5213311111111",
            thread_id="wa-meta:100:5213311111111",
            direction="outbound",
            event_type="message_sent",
            success=True,
            run_id="run-1",
            metadata=None,
        )
        await store.record_event(
            channel="whatsapp_meta",
            user_id="5213322222222",
            thread_id="wa-meta:100:5213322222222",
            direction="inbound",
            event_type="message_inbound",
            success=True,
            run_id="run-2",
            metadata=None,
        )
        return await store.summarize(since_hours=24, top_users_limit=10)

    summary = asyncio.run(_run())

    totals = summary["totals"]  # type: ignore[index]
    assert totals["inbound_messages"] == 2  # type: ignore[index]
    assert totals["outbound_messages"] == 1  # type: ignore[index]
    assert totals["unique_users"] == 2  # type: ignore[index]
    assert totals["active_threads"] == 2  # type: ignore[index]

    channels = summary["channels"]  # type: ignore[index]
    assert isinstance(channels, list)
    assert channels[0]["channel"] == "whatsapp_meta"
