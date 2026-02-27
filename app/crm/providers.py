from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

from app.infra.db import PostgresPoolManager


class CRMProvider(Protocol):
    async def upsert_quote(self, payload: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(slots=True)
class InMemoryCRMProvider:
    _records: dict[str, dict[str, Any]]

    def __init__(self) -> None:
        self._records = {}

    async def upsert_quote(self, payload: dict[str, Any]) -> dict[str, Any]:
        candidate = payload.get("quote_id")
        if isinstance(candidate, str) and candidate.strip():
            crm_id = candidate.strip()
        else:
            crm_id = str(uuid4())

        self._records[crm_id] = dict(payload)
        return {"crm_id": crm_id, "status": "upserted"}


class PostgresCRMProvider:
    def __init__(self, pool_manager: PostgresPoolManager, *, table_name: str = "crm_leads") -> None:
        self._pool_manager = pool_manager
        self._table_name = table_name

    async def upsert_quote(self, payload: dict[str, Any]) -> dict[str, Any]:
        candidate = payload.get("quote_id")
        if isinstance(candidate, str) and candidate.strip():
            crm_id = candidate.strip()
        else:
            crm_id = str(uuid4())

        external_key = str(payload.get("external_key") or crm_id)
        lead_status = payload.get("lead_status")
        lead_status_value = str(lead_status) if isinstance(lead_status, str) else None

        qualification_evidence = payload.get("qualification_evidence")
        if isinstance(qualification_evidence, list):
            evidence_json = json.dumps(qualification_evidence, ensure_ascii=True)
        else:
            evidence_json = json.dumps([], ensure_ascii=True)

        next_action = payload.get("next_action")
        next_action_value = str(next_action) if isinstance(next_action, str) else None
        payload_json = json.dumps(payload, ensure_ascii=True)

        connection = await self._pool_manager.acquire()
        try:
            row = await connection.fetchrow(
                f"""
                INSERT INTO {self._table_name}
                    (crm_id, external_key, payload, lead_status, qualification_evidence, next_action, updated_at)
                VALUES
                    ($1, $2, $3::jsonb, $4, $5::jsonb, $6, NOW())
                ON CONFLICT (crm_id) DO UPDATE
                SET
                    external_key = EXCLUDED.external_key,
                    payload = EXCLUDED.payload,
                    lead_status = EXCLUDED.lead_status,
                    qualification_evidence = EXCLUDED.qualification_evidence,
                    next_action = EXCLUDED.next_action,
                    updated_at = NOW()
                RETURNING crm_id
                """,
                crm_id,
                external_key,
                payload_json,
                lead_status_value,
                evidence_json,
                next_action_value,
            )
            persisted_id = str(row["crm_id"]) if row is not None and "crm_id" in row else crm_id
            return {"crm_id": persisted_id, "status": "upserted"}
        finally:
            await self._pool_manager.release(connection)
