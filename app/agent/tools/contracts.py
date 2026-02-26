from __future__ import annotations

TOOL_CONTRACTS_MINIMAL: dict[str, object] = {
    "$id": "tool_contracts_minimal",
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "$defs": {
        "iso_input": {
            "type": "object",
            "additionalProperties": False,
            "required": ["country_name"],
            "properties": {"country_name": {"type": "string", "minLength": 1}},
        },
        "markets_input": {
            "type": "object",
            "additionalProperties": False,
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "country_code": {"type": "string", "pattern": "^[A-Z]{2}$"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            },
        },
    },
}
