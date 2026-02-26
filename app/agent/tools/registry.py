from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agent.tools.validator import ToolValidationError, validate_tool_input, validate_tool_output


@dataclass(slots=True)
class ToolExecutionResult:
    tool: str
    input: dict[str, Any]
    output: dict[str, Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._user_preferences: dict[str, dict[str, str]] = {}

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        return (
            "get_iso_country_code",
            "retrieve_markets",
            "retrieve_segments",
            "update_user_preferences",
            "crm_upsert_quote",
        )

    async def execute(self, tool_name: str, payload: dict[str, Any]) -> ToolExecutionResult:
        try:
            validated = validate_tool_input(tool_name, payload)
        except ToolValidationError as exc:
            output = validate_tool_output(tool_name, {"error": str(exc)})
            return ToolExecutionResult(tool=tool_name, input={}, output=output)

        try:
            if tool_name == "get_iso_country_code":
                output = self._get_iso_country_code(validated)
            elif tool_name == "retrieve_markets":
                output = self._retrieve_markets(validated)
            elif tool_name == "retrieve_segments":
                output = self._retrieve_segments(validated)
            elif tool_name == "update_user_preferences":
                output = self._update_user_preferences(validated)
            elif tool_name == "crm_upsert_quote":
                output = self._crm_upsert_quote(validated)
            else:
                raise ToolValidationError(f"unsupported tool {tool_name}")
        except Exception as exc:
            output = {"error": f"tool execution failed: {type(exc).__name__}: {exc}"}

        normalized_output = validate_tool_output(tool_name, output)
        return ToolExecutionResult(tool=tool_name, input=validated, output=normalized_output)

    def _get_iso_country_code(self, payload: dict[str, Any]) -> dict[str, Any]:
        mapping = {
            "argentina": "AR",
            "chile": "CL",
            "mexico": "MX",
            "united states": "US",
            "usa": "US",
            "spain": "ES",
            "colombia": "CO",
            "peru": "PE",
        }
        normalized = str(payload["country_name"]).strip().lower()
        iso_code = mapping.get(normalized)
        if not iso_code:
            return {"error": f"country not found: {payload['country_name']}"}
        return {"iso_code": iso_code}

    def _retrieve_markets(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = str(payload["query"]).lower()
        country_code = payload.get("country_code", "US")
        limit = int(payload.get("limit", 10))
        sample = [
            {"name": "Retail", "country_code": country_code, "score": 0.92},
            {"name": "CPG", "country_code": country_code, "score": 0.84},
            {"name": "Fintech", "country_code": country_code, "score": 0.81},
            {"name": "Healthcare", "country_code": country_code, "score": 0.78},
        ]
        filtered = [item for item in sample if query in item["name"].lower() or len(query) < 4]
        return {"markets": (filtered or sample)[:limit]}

    def _retrieve_segments(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = str(payload["query"]).lower()
        limit = int(payload.get("limit", 10))
        segments = [
            {"id": "seg-1", "name": "Young Adults", "score": 0.91},
            {"id": "seg-2", "name": "Families", "score": 0.86},
            {"id": "seg-3", "name": "Professionals", "score": 0.82},
        ]
        selected = [s for s in segments if query in s["name"].lower() or len(query) < 4]
        return {"segments": (selected or segments)[:limit]}

    def _update_user_preferences(self, payload: dict[str, Any]) -> dict[str, Any]:
        email = str(payload["email"]).lower()
        prefs = {str(k): str(v) for k, v in dict(payload["preferences"]).items()}
        self._user_preferences[email] = prefs
        return {"status": "persisted", "email": email, "preferences": prefs}

    def _crm_upsert_quote(self, payload: dict[str, Any]) -> dict[str, Any]:
        quote = dict(payload["payload"])
        quote_id = str(quote.get("quote_id", "quote-temp"))
        return {"crm_id": quote_id, "status": "upserted"}
