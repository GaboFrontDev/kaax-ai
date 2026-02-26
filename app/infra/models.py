from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class UserPreference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str
    preferences: dict[str, str]
