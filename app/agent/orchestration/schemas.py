from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Stage = Literal[
    "greeting",
    "discovery",
    "opportunity",
    "value_mapping",
    "lead_capture",
    "demo_cta",
]
Intent = Literal["unknown", "exploring", "interested", "demo_requested"]
Qualification = Literal["cold", "warm", "hot"]
MissingField = Literal["contact_name", "phone", "contact_schedule"]

RouteName = Literal["repeat_handler", "memory_lookup", "discovery_value"]
NextAction = Literal["answer", "ask_question", "capture_lead", "demo_cta", "handoff"]
MemoryRouteMode = Literal["read", "update"]

CONTACT_FIELD_ORDER: tuple[MissingField, MissingField, MissingField] = (
    "contact_name",
    "phone",
    "contact_schedule",
)


class UserProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    phone: str | None = None
    role: str | None = None
    company: str | None = None


class BusinessContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    industry: str | None = None
    use_case: str | None = None
    pain_points: list[str] = Field(default_factory=list)
    channels: list[str] = Field(default_factory=list)
    volume_estimate: str | None = None


class SalesState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Intent = "unknown"
    qualification: Qualification = "cold"
    buying_signals: list[str] = Field(default_factory=list)
    objections: list[str] = Field(default_factory=list)


class LeadData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contact_name: str | None = None
    phone: str | None = None
    contact_schedule: str | None = None
    intent: str | None = None
    qualification: str | None = None


class QAItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_question: str
    answer_summary: str


class QAMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answered_questions: list[QAItem] = Field(default_factory=list)
    factual_cache: dict[str, Any] = Field(default_factory=dict)


class ToolingState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    last_memory_route_mode: MemoryRouteMode | None = None
    last_capture_result: dict[str, Any] | None = None


class TurnFlags(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_repeat_question: bool = False
    needs_factual_lookup: bool = False
    has_buying_signal: bool = False
    lead_capture_ready: bool = False


class ConversationState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: Stage = "greeting"
    user_profile: UserProfile = Field(default_factory=UserProfile)
    business_context: BusinessContext = Field(default_factory=BusinessContext)
    sales: SalesState = Field(default_factory=SalesState)
    lead_data: LeadData = Field(default_factory=LeadData)
    qa_memory: QAMemory = Field(default_factory=QAMemory)
    tooling: ToolingState = Field(default_factory=ToolingState)
    flags: TurnFlags = Field(default_factory=TurnFlags)


class RouterDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: RouteName
    stage: Stage
    intent: Intent
    qualification: Qualification
    missing_fields: list[MissingField] = Field(default_factory=list)
    next_action: NextAction
    flags: TurnFlags = Field(default_factory=TurnFlags)


class OrchestrationState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str = ""
    requestor: str = ""
    messages: list[dict[str, str]] = Field(default_factory=list)
    last_user_message: str = ""

    conversation_state: ConversationState = Field(default_factory=ConversationState)
    router: RouterDecision | None = None

    draft_response: str = ""
    tool_result: dict[str, Any] | None = None
    tools_used: list[str] = Field(default_factory=list)
    final_response: str = ""
