from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Mode = Literal["greeting", "discovery", "capture_completion", "support_answer", "handoff"]
AgentName = Literal["greeting", "core_capture", "knowledge", "inventory"]
Intent = Literal["support", "product_inquiry", "pricing", "purchase_intent", "unknown"]
Qualification = Literal["cold", "warm", "hot"]
LeadStatus = Literal["calificado", "en_revision", "no_calificado"]
MissingField = Literal["contact_name", "phone", "contact_schedule"]
NextAction = Literal["answer", "ask_question", "capture_lead", "handoff"]
PricingSource = Literal["unknown", "kb", "snapshot"]

CONTACT_FIELD_ORDER: tuple[MissingField, MissingField, MissingField] = (
    "contact_name",
    "phone",
    "contact_schedule",
)


class CapturedContact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contact_name: str | None = None
    phone: str | None = None
    contact_schedule: str | None = None


class LeadState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Intent = "unknown"
    qualification: Qualification = "cold"
    status: LeadStatus = "en_revision"


class PricingContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verified_summary: str | None = None
    source: PricingSource = "unknown"
    query: str | None = None
    updated_at: str | None = None


class ConversationState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Mode = "greeting"
    captured: CapturedContact = Field(default_factory=CapturedContact)
    lead: LeadState = Field(default_factory=LeadState)
    pricing_context: PricingContext = Field(default_factory=PricingContext)


class RouterDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Mode
    agent: AgentName
    intent: Intent
    qualification: Qualification
    missing_fields: list[MissingField] = Field(default_factory=list)
    next_action: NextAction


class OrchestrationState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str = ""
    requestor: str = ""
    last_user_message: str = ""
    conversation_state: ConversationState = Field(default_factory=ConversationState)
    router: RouterDecision | None = None
    draft_response: str = ""
    tool_result: dict[str, Any] | None = None
    tools_used: list[str] = Field(default_factory=list)
    final_response: str = ""
