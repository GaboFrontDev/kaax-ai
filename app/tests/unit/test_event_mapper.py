from __future__ import annotations

from app.agent.event_mapper import LangChainStreamEventMapper


def test_stream_mapper_drops_thinking_blocks_across_chunks() -> None:
    mapper = LangChainStreamEventMapper(thread_id="t1", run_id="r1")

    out_1 = mapper.map_event(
        {"event": "on_chat_model_stream", "data": {"chunk": "<thinking>analizando"}}
    )
    out_2 = mapper.map_event(
        {"event": "on_chat_model_stream", "data": {"chunk": " contexto</thinking>Respuesta visible"}}
    )

    assert out_1 == []
    assert len(out_2) == 1
    assert out_2[0]["type"] == "content"
    assert out_2[0]["content"] == "Respuesta visible"


def test_stream_mapper_keeps_plain_text() -> None:
    mapper = LangChainStreamEventMapper(thread_id="t1", run_id="r1")
    out = mapper.map_event({"event": "on_chat_model_stream", "data": {"chunk": "Hola mundo"}})

    assert len(out) == 1
    assert out[0]["content"] == "Hola mundo"
