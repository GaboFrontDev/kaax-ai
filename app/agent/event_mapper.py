from __future__ import annotations

import re
from typing import Any

from app.agent.result_parser import content_to_text

_THINKING_BLOCK_RE = re.compile(r"<thinking\b[^>]*>.*?</thinking>", re.IGNORECASE | re.DOTALL)
_THINKING_OPEN_TAG_RE = re.compile(r"<thinking\b[^>]*>", re.IGNORECASE)
_THINKING_CLOSE_TAG_RE = re.compile(r"</thinking\s*>", re.IGNORECASE)


class LangChainStreamEventMapper:
    def __init__(self, *, thread_id: str, run_id: str) -> None:
        self.thread_id = thread_id
        self.run_id = run_id
        self.tools_used: list[str] = []
        self.final_output: Any = None
        self._thinking_carry = ""

    def _sanitize_stream_text(self, text: str) -> str:
        combined = f"{self._thinking_carry}{text}"
        self._thinking_carry = ""

        # Remove complete <thinking>...</thinking> sections first.
        combined = _THINKING_BLOCK_RE.sub("", combined)
        lowered = combined.lower()

        # If an opening tag remains without a closing tag, keep it for the next chunk.
        open_idx = lowered.rfind("<thinking")
        close_idx = lowered.rfind("</thinking>")
        if open_idx >= 0 and close_idx < open_idx:
            self._thinking_carry = combined[open_idx:]
            combined = combined[:open_idx]

        # Keep trailing partial opening tags for next chunk.
        marker = "<thinking"
        trailing = combined[-(len(marker) - 1) :]
        if trailing and marker.startswith(trailing.lower()):
            self._thinking_carry = trailing + self._thinking_carry
            combined = combined[: -(len(trailing))]

        combined = _THINKING_OPEN_TAG_RE.sub("", combined)
        combined = _THINKING_CLOSE_TAG_RE.sub("", combined)
        return combined

    def map_event(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        mapped: list[dict[str, Any]] = []
        event_name = str(event.get("event", ""))
        data = event.get("data", {}) if isinstance(event.get("data"), dict) else {}

        if event_name == "on_chat_model_stream":
            text = content_to_text(data.get("chunk"))
            if text:
                text = self._sanitize_stream_text(text)
            if text:
                mapped.append(
                    {
                        "type": "content",
                        "content": text,
                        "thread_id": self.thread_id,
                        "run_id": self.run_id,
                    }
                )
            return mapped

        if event_name == "on_tool_start":
            tool_name = str(event.get("name", "tool"))
            if tool_name not in self.tools_used:
                self.tools_used.append(tool_name)
            mapped.append(
                {
                    "type": "tool_start",
                    "tool": tool_name,
                    "payload": data.get("input") if isinstance(data.get("input"), dict) else {"input": data.get("input")},
                    "thread_id": self.thread_id,
                    "run_id": self.run_id,
                }
            )
            return mapped

        if event_name == "on_tool_end":
            tool_name = str(event.get("name", "tool"))
            output = data.get("output")
            mapped.append(
                {
                    "type": "tool_result",
                    "tool": tool_name,
                    "payload": output if isinstance(output, dict) else {"output": output},
                    "thread_id": self.thread_id,
                    "run_id": self.run_id,
                }
            )
            return mapped

        if event_name == "on_chain_end" and "output" in data:
            self.final_output = data.get("output")
            self._thinking_carry = ""
            return mapped

        return mapped
