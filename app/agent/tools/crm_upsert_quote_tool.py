from __future__ import annotations

from typing import Any

from app.crm.providers import CRMProvider


class CrmUpsertQuoteTool:
    name = "crm_upsert_quote"

    def __init__(self, crm_provider: CRMProvider) -> None:
        self._crm_provider = crm_provider

    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        quote = dict(payload["payload"])
        return await self._crm_provider.upsert_quote(quote)
