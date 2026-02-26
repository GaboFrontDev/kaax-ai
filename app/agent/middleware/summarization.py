from __future__ import annotations

from typing import Any


class SummarizationMiddleware:
    def __init__(
        self,
        *,
        max_tokens_before_summary: int = 150_000,
        messages_to_keep: int = 20,
    ) -> None:
        self._max_tokens = max_tokens_before_summary
        self._messages_to_keep = messages_to_keep

    async def maybe_summarize(self, state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get("messages", [])
        token_estimate = self._estimate_tokens(messages)
        if token_estimate < self._max_tokens:
            return state

        kept = messages[-self._messages_to_keep :]
        older = messages[: -self._messages_to_keep]
        summary = {
            "facts": [m["content"] for m in older if m.get("role") == "assistant"][:8],
            "decisions": [m["content"] for m in older if "decid" in m.get("content", "").lower()][:5],
            "open_loops": [],
            "user_preferences": [],
        }
        state["summary"] = summary
        state["messages"] = kept
        return state

    @staticmethod
    def _estimate_tokens(messages: list[dict[str, str]]) -> int:
        chars = sum(len(message.get("content", "")) for message in messages)
        return max(1, chars // 4)
