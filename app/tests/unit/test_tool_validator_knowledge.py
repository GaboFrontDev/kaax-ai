from __future__ import annotations

import pytest

from app.agent.tools.validator import ToolValidationError, validate_tool_input, validate_tool_output


def test_validate_knowledge_search_input_accepts_valid_payload() -> None:
    payload = validate_tool_input("knowledge_search", {"query": "precio", "limit": 3})
    assert payload == {"query": "precio", "limit": 3}


def test_validate_knowledge_learn_input_rejects_oversized_payload() -> None:
    with pytest.raises(ToolValidationError):
        validate_tool_input(
            "knowledge_learn",
            {"source_text": "x" * 3001},
        )


def test_validate_knowledge_search_output_accepts_matches_shape() -> None:
    output = validate_tool_output(
        "knowledge_search",
        {
            "matches": [
                {
                    "topic": "pricing",
                    "content": "El plan base inicia en 99 USD.",
                    "score": 0.82,
                    "updated_at": "2026-02-28T00:00:00+00:00",
                }
            ]
        },
    )
    assert output["matches"][0]["topic"] == "pricing"


def test_validate_knowledge_learn_output_rejects_invalid_status() -> None:
    with pytest.raises(ToolValidationError):
        validate_tool_output(
            "knowledge_learn",
            {
                "status": "saved",
                "message": "ok",
                "pending": False,
            },
        )
