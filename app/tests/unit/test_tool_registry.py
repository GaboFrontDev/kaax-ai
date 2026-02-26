import asyncio

from app.agent.tools.registry import ToolRegistry


def test_tool_registry_returns_structured_error_for_invalid_input() -> None:
    registry = ToolRegistry()

    result = asyncio.run(registry.execute("retrieve_markets", {"country_code": "US"}))

    assert result.tool == "retrieve_markets"
    assert "error" in result.output


def test_tool_registry_validates_success_output_shape() -> None:
    registry = ToolRegistry()

    result = asyncio.run(registry.execute("retrieve_markets", {"query": "ret", "country_code": "US", "limit": 2}))

    assert "markets" in result.output
    first = result.output["markets"][0]
    assert set(first.keys()) == {"name", "country_code", "score"}
