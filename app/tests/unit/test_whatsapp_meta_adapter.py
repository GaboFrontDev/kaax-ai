import asyncio

from app.channels.whatsapp_meta.adapter import WhatsAppMetaAdapter


def test_meta_adapter_uses_stable_thread_per_conversation_pair() -> None:
    adapter = WhatsAppMetaAdapter()
    inbound_a = {
        "from": "5215550000001",
        "to": "123456789012345",
        "text": "hola",
        "wa_message_id": "wamid.A",
    }
    inbound_b = {
        "from": "5215550000001",
        "to": "123456789012345",
        "text": "sigo aqui",
        "wa_message_id": "wamid.B",
    }

    req_a = asyncio.run(adapter.normalize_inbound(inbound_a))
    req_b = asyncio.run(adapter.normalize_inbound(inbound_b))

    assert req_a.thread_id == req_b.thread_id
    assert req_a.thread_id == "wa-meta:123456789012345:5215550000001"
