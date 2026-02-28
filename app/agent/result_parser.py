from __future__ import annotations

import json
from typing import Any


def content_to_text(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        for key in ("text", "content", "output", "answer"):
            value = content.get(key)
            if isinstance(value, str):
                return value
        return json.dumps(content, ensure_ascii=True)

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    text_attr = getattr(content, "content", None)
    if text_attr is not None and text_attr is not content:
        return content_to_text(text_attr)

    return str(content)


def extract_response_text(result: Any) -> str:
    if isinstance(result, str):
        return result.strip()

    if isinstance(result, dict):
        messages = result.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                role = ""
                content: Any = None
                if isinstance(message, dict):
                    role = str(message.get("role") or message.get("type") or "")
                    content = message.get("content")
                else:
                    role = str(getattr(message, "type", ""))
                    content = getattr(message, "content", None)
                if role in {"ai", "assistant"}:
                    text = content_to_text(content).strip()
                    if text:
                        return text

        for key in ("output", "response", "answer"):
            if key in result:
                text = content_to_text(result.get(key)).strip()
                if text:
                    return text

        return json.dumps(result, ensure_ascii=True)

    return content_to_text(result).strip()


def dedupe_tools(tools: list[str]) -> list[str]:
    deduped: list[str] = []
    for tool in tools:
        if tool and tool not in deduped:
            deduped.append(tool)
    return deduped


def extract_tools_used(result: Any) -> list[str]:
    tools_used: list[str] = []
    if not isinstance(result, dict):
        return tools_used

    messages = result.get("messages")
    if not isinstance(messages, list):
        return tools_used

    for message in messages:
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)
        if not isinstance(tool_calls, list):
            continue

        for tool_call in tool_calls:
            name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)
            if isinstance(name, str) and name and name not in tools_used:
                tools_used.append(name)

    return tools_used
