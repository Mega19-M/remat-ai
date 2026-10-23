"""Typed data structures shared by the agent, tools and UI (standard library only)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class Action(str, Enum):
    """The typed option set the Decider chooses from (tool-selection pattern)."""

    INVESTIGATE_RECOVERY = "investigate_recovery"
    CHECK_DISPOSAL = "check_disposal"
    CHECK_EVIDENCE = "check_evidence"
    ESCALATE = "escalate"
    REQUEST_INFO = "request_info"
    CONCLUDE = "conclude"


class Pathway(str, Enum):
    RECOVER = "RECOVER"
    DISPOSE = "DISPOSE"
    ESCALATE = "ESCALATE"
    VERIFY = "VERIFY"


ACTION_TO_PATHWAY = {
    Action.INVESTIGATE_RECOVERY: Pathway.RECOVER,
    Action.CHECK_DISPOSAL: Pathway.DISPOSE,
    Action.ESCALATE: Pathway.ESCALATE,
    Action.CHECK_EVIDENCE: Pathway.VERIFY,
    Action.REQUEST_INFO: Pathway.VERIFY,
}


class Label(str, Enum):
    """How much trust a displayed statement deserves."""

    REFERENCE = "SOURCE-BACKED"      # from the curated dataset, with a source pointer
    HYPOTHESIS = "HYPOTHESIS"        # generated or suggested; not verified
    UNKNOWN = "UNKNOWN"              # missing information
    SAFETY = "SAFETY RULE"           # deterministic safety statement


class Level(str, Enum):
    ROUTINE = "routine"
    REGULATED = "regulated"
    HIGH_RISK = "unknown_high_risk"


OBJECTIVES = {
    "recover": "Recover value (reuse / recycle)",
    "dispose": "Dispose safely",
    "both": "Investigate both",
}


@dataclass
class WasteCase:
    waste_name: str = ""
    description: str = ""
    category: str = ""
    location: str = ""
    danger_info: str = ""
    objective: str = "both"
    photo_attached: bool = False
    photo_confirmed_by_user: bool = False
    synthetic_scenario_id: Optional[str] = None  # set when loaded from a demo scenario


@dataclass
class SafetyAssessment:
    level: Level
    reasons: list[str] = field(default_factory=list)
    emergency: bool = False
    emergency_terms: list[str] = field(default_factory=list)
    missing_info: list[str] = field(default_factory=list)
    material_id: Optional[str] = None
    matched_terms: list[str] = field(default_factory=list)


@dataclass
class Finding:
    statement: str
    label: Label
    source_ids: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    next_step: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeciderDecision:
    """What the decision backend actually returned. Nothing here is invented."""

    backend: str                          # "strands-decider-2b (live, local server)" or "simulated-rules"
    live: bool                            # True only for a real Decider 2B response
    question: str
    options: list[str]
    selected: str
    confidence: Optional[float] = None    # only if the API returned one
    probabilities: Optional[dict[str, float]] = None  # only if the API returned them
    latency_ms: Optional[float] = None    # only if the API returned it
    selected_derived_from: str = ""       # e.g. "argmax of returned probabilities"
    raw_response: Optional[dict[str, Any]] = None
    rationale: str = ""                   # simulated backend only; live Decider gives none


@dataclass
class GuardrailResult:
    allowed: bool
    proposed: str
    final: str
    rule_id: str = ""
    reason: str = ""


@dataclass
class StepRecord:
    step: int
    available_options: list[str]
    state_sent: str
    decision: Optional[DeciderDecision]
    guardrail: Optional[GuardrailResult]
    executed_action: Optional[str]
    executed_by: str                      # "decider" or "guardrail_override"
    tool_inputs: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    error: str = ""


@dataclass
class GeneratedText:
    text: str
    source: str                           # e.g. "bedrock:<model id>" or "template (no AI)"
    label: Label
    ai_generated: bool
    note: str = ""


@dataclass
class Recommendation:
    pathway: Pathway
    headline: str
    reasons: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)


@dataclass
class CaseResult:
    status: str                           # ok | invalid_input | decider_error
    mode_label: str                       # banner text: LIVE or SIMULATED
    simulated: bool
    case: WasteCase
    assessment: Optional[SafetyAssessment] = None
    steps: list[StepRecord] = field(default_factory=list)
    recommendation: Optional[Recommendation] = None
    summary: Optional[GeneratedText] = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    missing_info: list[str] = field(default_factory=list)
    why_not_selected: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
