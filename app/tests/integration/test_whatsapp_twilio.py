from fastapi.testclient import TestClient

from app.api.main import app


def test_whatsapp_twilio_webhook_returns_twiml_message() -> None:
    client = TestClient(app)

    response = client.post(
        "/webhooks/whatsapp/twilio",
        data={
            "From": "whatsapp:+5215550000001",
            "To": "whatsapp:+14155238886",
            "Body": "hola",
            "MessageSid": "SM123",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    assert "<Response>" in response.text
    assert "<Message>" in response.text
