from __future__ import annotations

from typing import Any

from app.agent.result_parser import content_to_text


class LangChainStreamEventMapper:
    def __init__(self, *, thread_id: str, run_id: str) -> None:
        self.thread_id = thread_id
        self.run_id = run_id
        self.tools_used: list[str] = []
        self.final_output: Any = None

    def map_event(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        mapped: list[dict[str, Any]] = []
        event_name = str(event.get("event", ""))
        data = event.get("data", {}) if isinstance(event.get("data"), dict) else {}

        if event_name == "on_chat_model_stream":
            text = content_to_text(data.get("chunk"))
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
            return mapped

        return mapped
