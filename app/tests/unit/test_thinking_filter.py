from app.agent.langchain_runtime import _ThinkingStreamFilter, _strip_thinking_tags


def test_strip_thinking_tags_removes_block_and_preserves_visible_text() -> None:
    raw = "<thinking>analisis interno</thinking>Respuesta final"
    assert _strip_thinking_tags(raw) == "Respuesta final"


def test_strip_thinking_tags_handles_tag_attributes() -> None:
    raw = '<thinking type="internal">oculto</thinking>Visible'
    assert _strip_thinking_tags(raw) == "Visible"


def test_thinking_stream_filter_handles_split_tags() -> None:
    filt = _ThinkingStreamFilter()
    out1 = filt.feed("<thin")
    out2 = filt.feed("king>oculto</thi")
    out3 = filt.feed("nking>Hola")
    out4 = filt.feed(" mundo")
    tail = filt.finish()

    assert out1 == ""
    assert out2 == ""
    assert out3 == "Hola"
    assert out4 == " mundo"
    assert tail == ""


def test_thinking_stream_filter_keeps_text_outside_tags() -> None:
    filt = _ThinkingStreamFilter()
    out1 = filt.feed("Inicio ")
    out2 = filt.feed("<thinking>oculto</thinking>")
    out3 = filt.feed("Fin")

    assert out1 == "Inicio "
    assert out2 == ""
    assert out3 == "Fin"
