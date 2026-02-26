import asyncio

from app.channels.whatsapp_twilio.adapter import WhatsAppTwilioAdapter


def test_twilio_adapter_uses_stable_thread_per_conversation_pair() -> None:
    adapter = WhatsAppTwilioAdapter()
    payload_a = {
        "From": "whatsapp:+5215550000001",
        "To": "whatsapp:+14155238886",
        "Body": "hola",
        "MessageSid": "SM1",
    }
    payload_b = {
        "From": "whatsapp:+5215550000001",
        "To": "whatsapp:+14155238886",
        "Body": "sigo aqui",
        "MessageSid": "SM2",
    }

    req_a = asyncio.run(adapter.normalize_inbound(payload_a))
    req_b = asyncio.run(adapter.normalize_inbound(payload_b))

    assert req_a.thread_id == req_b.thread_id
    assert req_a.thread_id == "wa-twilio:whatsapp:+14155238886:whatsapp:+5215550000001"
