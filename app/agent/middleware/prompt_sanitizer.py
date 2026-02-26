from __future__ import annotations

import re


class PromptSanitizerMiddleware:
    _PATTERNS = [
        re.compile(r"ignore\s+previous\s+instructions", re.IGNORECASE),
        re.compile(r"reveal\s+(your\s+)?(system\s+)?prompt", re.IGNORECASE),
        re.compile(r"muestra\s+el\s+prompt\s+interno", re.IGNORECASE),
    ]

    def sanitize(self, text: str) -> str:
        sanitized = text
        for pattern in self._PATTERNS:
            sanitized = pattern.sub("[REDACTED_INJECTION_ATTEMPT]", sanitized)
        return sanitized.strip()
