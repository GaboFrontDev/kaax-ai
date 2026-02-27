from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError


class ToolValidationError(ValueError):
    pass


class IsoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    country_name: str = Field(min_length=1)


class MarketsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    country_code: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    limit: int = Field(default=10, ge=1, le=50)


class SegmentsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    country_code: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    taxonomy: str = Field(default="default", min_length=1)
    limit: int = Field(default=10, ge=1, le=50)


class FormatsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market_name: str = Field(min_length=1)
    format_query: str = Field(min_length=1)
    country_code: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    limit: int = Field(default=10, ge=1, le=50)


class FindUnitsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_ids: list[str] = Field(min_length=1)
    markets: list[str] = Field(min_length=1)
    media_formats: list[str] | None = None
    limit: int = Field(default=10, ge=1, le=50)


class UpdatePreferencesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=3)
    preferences: dict[str, str]


class CrmUpsertQuoteInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    payload: dict[str, object]


class DetectLeadCaptureReadinessInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    business_context: dict[str, object] = Field(default_factory=dict)
    whatsapp_context: dict[str, object] = Field(default_factory=dict)
    crm_context: dict[str, object] = Field(default_factory=dict)
    agent_limits: dict[str, object] = Field(default_factory=dict)
    lead_data: dict[str, object] = Field(default_factory=dict)


class ErrorOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: str = Field(min_length=1)


class IsoOutputSuccess(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iso_code: str = Field(pattern=r"^[A-Z]{2}$")


class MarketItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    country_code: str = Field(pattern=r"^[A-Z]{2}$")
    score: float = Field(ge=0, le=1)


class MarketsOutputSuccess(BaseModel):
    model_config = ConfigDict(extra="forbid")

    markets: list[MarketItem]


class SegmentItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    score: float = Field(ge=0, le=1)


class SegmentsOutputSuccess(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segments: list[SegmentItem]


class FormatItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    score: float = Field(ge=0, le=1)


class FormatsOutputSuccess(BaseModel):
    model_config = ConfigDict(extra="forbid")

    formats: list[FormatItem]


class UnitItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit_id: str
    market: str
    segment_id: str
    score: float = Field(ge=0, le=1)


class FindUnitsOutputSuccess(BaseModel):
    model_config = ConfigDict(extra="forbid")

    units: list[UnitItem]


class UpdatePreferencesOutputSuccess(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["persisted"]
    email: str
    preferences: dict[str, str]


class CrmUpsertOutputSuccess(BaseModel):
    model_config = ConfigDict(extra="forbid")

    crm_id: str
    status: Literal["upserted"]


class LeadCaptureReadinessOutputSuccess(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ready_for_capture: bool
    lead_status: Literal["calificado", "no_calificado", "en_revision"]
    qualification_evidence: list[str]
    missing_critical_fields: list[str]
    next_action: Literal["registro_crm", "solicitar_datos_faltantes", "handoff_humano", "cierre_cordial"]
    suggested_crm_payload: dict[str, object]


_INPUT_MODELS: dict[str, type[BaseModel]] = {
    "get_iso_country_code": IsoInput,
    "retrieve_markets": MarketsInput,
    "retrieve_segments": SegmentsInput,
    "retrieve_formats": FormatsInput,
    "find_units": FindUnitsInput,
    "update_user_preferences": UpdatePreferencesInput,
    "crm_upsert_quote": CrmUpsertQuoteInput,
    "detect_lead_capture_readiness": DetectLeadCaptureReadinessInput,
}

_OUTPUT_ADAPTERS: dict[str, TypeAdapter[Any]] = {
    "get_iso_country_code": TypeAdapter(IsoOutputSuccess | ErrorOutput),
    "retrieve_markets": TypeAdapter(MarketsOutputSuccess | ErrorOutput),
    "retrieve_segments": TypeAdapter(SegmentsOutputSuccess | ErrorOutput),
    "retrieve_formats": TypeAdapter(FormatsOutputSuccess | ErrorOutput),
    "find_units": TypeAdapter(FindUnitsOutputSuccess | ErrorOutput),
    "update_user_preferences": TypeAdapter(UpdatePreferencesOutputSuccess | ErrorOutput),
    "crm_upsert_quote": TypeAdapter(CrmUpsertOutputSuccess | ErrorOutput),
    "detect_lead_capture_readiness": TypeAdapter(LeadCaptureReadinessOutputSuccess | ErrorOutput),
}


def validate_tool_input(tool_name: str, payload: dict[str, object]) -> dict[str, object]:
    model = _INPUT_MODELS.get(tool_name)
    if model is None:
        raise ToolValidationError(f"tool not allowed: {tool_name}")

    try:
        return model.model_validate(payload).model_dump(exclude_none=True)
    except ValidationError as exc:
        raise ToolValidationError(exc.json()) from exc


def validate_tool_output(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    adapter = _OUTPUT_ADAPTERS.get(tool_name)
    if adapter is None:
        raise ToolValidationError(f"tool not allowed: {tool_name}")

    try:
        validated = adapter.validate_python(payload)
        if isinstance(validated, BaseModel):
            return validated.model_dump(exclude_none=True)
        if isinstance(validated, dict):
            return validated
        raise ToolValidationError("validated output has unsupported type")
    except ValidationError as exc:
        raise ToolValidationError(exc.json()) from exc
