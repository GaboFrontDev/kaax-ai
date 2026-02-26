from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(slots=True)
class InMemoryMetrics:
    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def inc(self, name: str, value: int = 1) -> None:
        self.counters[name] += value

    def snapshot(self) -> dict[str, int]:
        return dict(self.counters)
