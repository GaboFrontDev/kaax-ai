import asyncio

from app.agent.middleware.summarization import SummarizationMiddleware


def test_summarization_trims_when_threshold_reached() -> None:
    middleware = SummarizationMiddleware(max_tokens_before_summary=1, messages_to_keep=2)
    state = {
        "messages": [
            {"role": "user", "content": "uno"},
            {"role": "assistant", "content": "dos"},
            {"role": "user", "content": "tres"},
        ]
    }

    output = asyncio.run(middleware.maybe_summarize(state))

    assert len(output["messages"]) == 2
    assert "summary" in output
