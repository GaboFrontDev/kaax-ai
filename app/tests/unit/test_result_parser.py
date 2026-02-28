from __future__ import annotations

from app.agent.result_parser import extract_response_text, strip_thinking_sections


def test_strip_thinking_sections_removes_block() -> None:
    text = "<thinking>internal plan</thinking>\nRespuesta final al usuario."
    assert strip_thinking_sections(text) == "Respuesta final al usuario."


def test_extract_response_text_strips_thinking_from_assistant_message() -> None:
    result = {
        "messages": [
            {"role": "user", "content": "hola"},
            {
                "role": "assistant",
                "content": "<thinking>debug interno</thinking>\nTu plan mensual es de 499 MXN.",
            },
        ]
    }

    assert extract_response_text(result) == "Tu plan mensual es de 499 MXN."


def test_extract_response_text_strips_unclosed_opening_tag() -> None:
    result = "<thinking>\nesto no debe salir\n</thinking>\nHorario: lunes a viernes."
    assert extract_response_text(result) == "Horario: lunes a viernes."
