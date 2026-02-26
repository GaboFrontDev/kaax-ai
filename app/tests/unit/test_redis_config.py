import pytest

from app.infra.redis import parse_sentinel_nodes


def test_parse_sentinel_nodes_success() -> None:
    parsed = parse_sentinel_nodes(("localhost:26379", "127.0.0.1:26380"))

    assert parsed == (("localhost", 26379), ("127.0.0.1", 26380))


def test_parse_sentinel_nodes_raises_for_invalid_node() -> None:
    with pytest.raises(ValueError):
        parse_sentinel_nodes(("localhost",))
