"""End-to-end decision workflow (tool-selection pattern).

Loop
----
1. Deterministic safety assessment.
2. Offer the Decider the typed option set.
3. The Decider (live or simulated) proposes the next action.
4. The safety guardrail may reject or replace the proposal.
5. The tool for the final action runs; its findings are recorded.
6. Repeat until `conclude` (or the step limit), then build a recommendation.

If the live Decider fails, the run STOPS with status "decider_error". The
workflow never silently substitutes simulated decisions for live ones.
"""
from __future__ import annotations

from typing import Callable, Optional

import tools as T
from agent import safety
from agent.decider import (
    ACTION_CRITERIA, DeciderBackend, DeciderError, SimulatedDeciderBackend, build_state,
)
from agent.generation import Generator, guarded_summary
from agent.schemas import (
    ACTION_TO_PATHWAY, Action, CaseResult, Finding, Label, Level, Pathway, Recommendation,
    SafetyAssessment, StepRecord, WasteCase,
)
from tools.disposal import EMERGENCY_WARNING

ToolFn = Callable[[WasteCase, SafetyAssessment, list[Finding]], list[Finding]]

# Registry: action -> deterministic tool. Tests may replace entries to simulate tool failures.
TOOL_REGISTRY: dict[str, ToolFn] = {
    Action.INVESTIGATE_RECOVERY.value: lambda c, a, prior: T.investigate_recovery(c, a),
    Action.CHECK_DISPOSAL.value: lambda c, a, prior: T.check_disposal_pathways(c, a),
    Action.CHECK_EVIDENCE.value: lambda c, a, prior: T.check_evidence(c, a, prior),
    Action.ESCALATE.value: lambda c, a, prior: T.escalate_to_professional(c, a),
    Action.REQUEST_INFO.value: lambda c, a, prior: T.request_missing_info(c, a),
}

SIMULATED_BANNER = "SIMULATED DEMO - decisions come from a fixed rule list, NOT from Strands Decider 2B or Amazon Bedrock"
LIVE_BANNER = "LIVE - decisions from Strands Decider 2B"


def available_options(executed: list[str], assessment: SafetyAssessment) -> list[str]:
    """The typed option set offered to the Decider at this step.

    Only loop hygiene is applied here: actions that already ran are removed,
    `request_info` is removed when nothing is missing, and `conclude` is only
    offered after at least one step. Safety decisions are NOT made here; they
    are made by the guardrail AFTER the Decider proposes.
    """
    opts = [a.value for a in Action if a.value not in executed]
    if not assessment.missing_info and Action.REQUEST_INFO.value in opts:
        opts.remove(Action.REQUEST_INFO.value)
    if not executed and Action.CONCLUDE.value in opts:
        opts.remove(Action.CONCLUDE.value)
    return opts


def _recommend(case: WasteCase, assessment: SafetyAssessment, executed: list[str]) -> Recommendation:
    ex = set(executed)
    esc, rec, dis = Action.ESCALATE.value, Action.INVESTIGATE_RECOVERY.value, Action.CHECK_DISPOSAL.value
    steps: list[str] = []
    reasons: list[str] = list(assessment.reasons)

    if esc in ex or assessment.level == Level.HIGH_RISK:
        if assessment.emergency:
            steps.append(EMERGENCY_WARNING)
        steps.append("Do not handle, open, move, mix or test the material; keep people away.")
        steps.append("Contact your State Pollution Control Board or a qualified hazardous-waste professional.")
        steps.append("Prepare what you can observe from a safe distance: labels, container, quantity, location.")
        return Recommendation(Pathway.ESCALATE, "ESCALATE: refer this material for qualified assessment.", reasons, steps)

    if case.objective == "dispose" and dis in ex:
        pathway = Pathway.DISPOSE
    elif case.objective == "recover" and rec in ex:
        pathway = Pathway.RECOVER
    elif rec in ex and dis in ex:
        pathway = Pathway.RECOVER if assessment.level == Level.ROUTINE else Pathway.DISPOSE
    elif rec in ex:
        pathway = Pathway.RECOVER
    elif dis in ex:
        pathway = Pathway.DISPOSE
    else:
        pathway = Pathway.VERIFY

    headline = {
        Pathway.RECOVER: "RECOVER: promising recovery routes were found in the reference data (not verified locally).",
        Pathway.DISPOSE: "DISPOSE: use an authorized disposal pathway (no local facility is verified by this app).",
        Pathway.VERIFY: "VERIFY: more information or evidence is needed before a pathway can be recommended.",
    }[pathway]
    if assessment.level == Level.REGULATED:
        steps.append("Use only authorized channels for this regulated material; no do-it-yourself processing.")
    if pathway in (Pathway.RECOVER, Pathway.DISPOSE):
        steps.append("Confirm any named authority or handler directly before acting; this app verifies no facility.")
    if pathway == Pathway.VERIFY:
        steps.append("Answer the open questions in the trace and run the assessment again.")
    return Recommendation(pathway, headline, reasons, steps)


def _why_not(steps: list[StepRecord]) -> list[str]:
    """Explain non-selected options ONLY from information the backend really returned."""
    out: list[str] = []
    for s in steps:
        d = s.decision
        if d is None:
            continue
        if d.probabilities:
            others = sorted(((o, p) for o, p in d.probabilities.items() if o != d.selected), key=lambda x: -x[1])
            txt = ", ".join(f"{o} ({p:.3f})" for o, p in others)
            out.append(f"Step {s.step}: the live Decider gave these other options lower probability: {txt}.")
        elif d.live:
            out.append(f"Step {s.step}: the Decider response contained no per-option probabilities, "
                       "so no explanation of the other options can be given.")
        else:
            out.append(f"Step {s.step}: simulated rule list - {d.rationale}")
        if s.guardrail and not s.guardrail.allowed:
            out.append(f"Step {s.step}: '{s.guardrail.proposed}' was rejected by safety rule "
                       f"{s.guardrail.rule_id}: {s.guardrail.reason}")
    return out


def run_case(
    case: WasteCase,
    backend: Optional[DeciderBackend] = None,
    generator: Optional[Generator] = None,
    max_steps: int = 5,
) -> CaseResult:
    """Run the full workflow. `backend=None` means simulated demo mode."""
    problems = safety.validate_case(case)
    simulated = backend is None or not getattr(backend, "live", False)
    banner = SIMULATED_BANNER if simulated else LIVE_BANNER
    if problems:
        return CaseResult(status="invalid_input", mode_label=banner, simulated=simulated, case=case, errors=problems)

    assessment = safety.assess(case)
    if backend is None:
        backend = SimulatedDeciderBackend(case, assessment)
        simulated = True
        banner = SIMULATED_BANNER

    result = CaseResult(
        status="ok", mode_label=banner, simulated=simulated, case=case,
        assessment=assessment, missing_info=list(assessment.missing_info),
    )
    executed: list[str] = []
    history: list[tuple[str, int]] = []
    findings_so_far: list[Finding] = []

    for step_no in range(1, max_steps + 1):
        options = available_options(executed, assessment)
        if not options:
            break
        state = build_state(case, assessment, history)
        try:
            decision = backend.choose(state, options)
        except DeciderError as exc:
            result.status = "decider_error"
            result.errors.append(str(exc))
            result.steps.append(StepRecord(step_no, options, state, None, None, None, "none", error=str(exc)))
            break

        verdict = safety.review_action(decision.selected, assessment, executed)
        final_action = verdict.final
        executed_by = "decider" if verdict.allowed else "guardrail_override"

        record = StepRecord(
            step=step_no, available_options=options, state_sent=state, decision=decision,
            guardrail=verdict, executed_action=None, executed_by=executed_by,
            tool_inputs={"action": final_action, "material_id": assessment.material_id},
        )
        result.steps.append(record)

        if final_action == Action.CONCLUDE.value:
            record.executed_action = final_action
            executed.append(final_action)
            break

        tool = TOOL_REGISTRY.get(final_action)
        record.executed_action = final_action
        executed.append(final_action)
        if tool is None:
            record.error = f"No tool is registered for action '{final_action}'."
            result.warnings.append(record.error)
            history.append((final_action, 0))
            continue
        try:
            found = tool(case, assessment, findings_so_far)
            record.findings = found
            findings_so_far.extend(found)
            history.append((final_action, len(found)))
        except Exception as exc:  # a tool failure must not crash the whole case
            record.error = f"Tool '{final_action}' failed: {exc}"
            result.warnings.append(record.error)
            history.append((final_action, 0))

    if result.status == "decider_error":
        result.summary = None
        result.why_not_selected = _why_not(result.steps)
        return result

    result.recommendation = _recommend(case, assessment, executed)
    result.why_not_selected = _why_not(result.steps)
    summary, warns = guarded_summary(generator, result, simulated=simulated)
    result.summary = summary
    result.warnings.extend(warns)
    return result
