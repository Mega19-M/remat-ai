"""Deterministic safety layer.

This module is deliberately separate from the Decider. The Decider proposes the
next investigation action; this module can REJECT or REPLACE that proposal.
Nothing in this file calls an AI model.
"""
from __future__ import annotations

import re
from typing import Iterable

from agent.schemas import (
    Action, GuardrailResult, Level, SafetyAssessment, WasteCase, OBJECTIVES,
)
from tools.material_lookup import lookup_material

MAX_FIELD_LEN = 2000

EMERGENCY_TERMS = [
    "leak", "leaking", "leaked", "fume", "fumes", "smoke", "smoking", "fire", "burning",
    "on fire", "flames", "hissing", "swollen", "swelling", "bulging", "overheating",
    "exposed", "exposure", "spill", "spilled", "spilling", "burn injury", "burns skin",
    "dizzy", "dizziness", "can't breathe", "difficulty breathing", "explosion", "exploded",
]

HIGH_RISK_TERMS = [
    "unlabelled", "unlabeled", "unmarked", "unknown", "mystery", "unidentified",
    "chemical smell", "corrosive", "toxic", "poison", "pesticide", "acid", "solvent",
    "reactive", "flammable", "explosive", "radioactive", "asbestos", "mercury",
    "biomedical", "infectious", "pcb", "chlorinated", "cyanide", "mixed with",
    "contaminated with", "unknown contents", "contents unknown",
]

_NEGATORS = ("no", "not", "without", "never", "none", "n't", "zero")


def _mentions(term: str, text: str) -> bool:
    text = text.lower()
    for m in re.finditer(r"(?<![a-z0-9])" + re.escape(term) + r"(?:s|ed|ing)?(?![a-z0-9])", text):
        window = text[max(0, m.start() - 24):m.start()]
        words = re.findall(r"[a-z']+", window)[-3:]
        if any(w in _NEGATORS or w.endswith("n't") for w in words):
            continue
        return True
    return False


def _hits(terms: Iterable[str], text: str) -> list[str]:
    return [t for t in terms if _mentions(t, text)]


def validate_case(case: WasteCase) -> list[str]:
    problems: list[str] = []
    name = (case.waste_name or "").strip()
    desc = (case.description or "").strip()
    if len(name) < 3 and len(desc) < 3:
        problems.append("Please describe the waste: add a name or a short description (at least 3 characters).")
    for label, value in (("Waste name", case.waste_name), ("Description", case.description),
                         ("Category", case.category), ("Location", case.location),
                         ("Danger information", case.danger_info)):
        if value and len(value) > MAX_FIELD_LEN:
            problems.append(f"{label} is too long (maximum {MAX_FIELD_LEN} characters).")
    if case.objective not in OBJECTIVES:
        problems.append(f"Objective must be one of: {', '.join(OBJECTIVES)}.")
    return problems


def missing_information(case: WasteCase, has_material_match: bool) -> list[str]:
    missing: list[str] = []
    if not (case.category or "").strip() and not has_material_match:
        missing.append("Waste category is not stated and could not be matched to the reference dataset.")
    if not (case.location or "").strip():
        missing.append("Approximate location (town or district) is not given; local authorities cannot be pointed out.")
    if not (case.danger_info or "").strip():
        missing.append("No information about contamination or danger was given (for example leaks, labels, mixing).")
    if case.photo_attached and not case.photo_confirmed_by_user:
        missing.append("A photo was attached but it is NOT analyzed by AI and you have not confirmed what it shows.")
    return missing


def assess(case: WasteCase) -> SafetyAssessment:
    full_text = " ".join([case.waste_name, case.description, case.category, case.danger_info])
    match = lookup_material(case)
    reasons: list[str] = []
    level = Level.ROUTINE

    if match is None:
        level = Level.HIGH_RISK
        reasons.append("The item is not in the small reference dataset, so it is treated cautiously as unidentified.")
    else:
        level = Level(match.material["hazard_class"])
        reasons.append(f"Matched reference material '{match.material['name']}' (class: {level.value}).")

    emergency_hits = _hits(EMERGENCY_TERMS, full_text)
    high_hits = _hits(HIGH_RISK_TERMS, full_text)
    if high_hits:
        level = Level.HIGH_RISK
        reasons.append("High-risk or unknown-material wording found: " + ", ".join(sorted(set(high_hits))) + ".")
    if emergency_hits:
        level = Level.HIGH_RISK
        reasons.append("Possible immediate-danger wording found: " + ", ".join(sorted(set(emergency_hits))) + ".")
    if level == Level.REGULATED:
        reasons.append("Regulated stream: only authorized channels may handle it; no do-it-yourself processing.")

    return SafetyAssessment(
        level=level,
        reasons=reasons,
        emergency=bool(emergency_hits),
        emergency_terms=sorted(set(emergency_hits)),
        missing_info=missing_information(case, match is not None),
        material_id=match.material["id"] if match else None,
        matched_terms=match.matched_terms if match else [],
    )


def review_action(
    proposed: str,
    assessment: SafetyAssessment,
    executed: list[str],
) -> GuardrailResult:
    """Check an action proposed by the Decider. May reject or replace it."""
    esc = Action.ESCALATE.value
    already = set(executed)

    def block(rule: str, reason: str, final: str) -> GuardrailResult:
        return GuardrailResult(False, proposed, final, rule, reason)

    # S0 - never repeat an action
    if proposed in already:
        fallback = Action.CONCLUDE.value if already else Action.CHECK_EVIDENCE.value
        if assessment.level == Level.HIGH_RISK and esc not in already:
            fallback = esc
        return block("S0_NO_REPEAT", f"'{proposed}' already ran; repeating adds no evidence.", fallback)

    # S1 - emergency wording: escalate first
    if assessment.emergency and proposed != esc and esc not in already:
        return block(
            "S1_EMERGENCY_FIRST",
            "Possible immediate danger was described (" + ", ".join(assessment.emergency_terms) +
            "). Escalation to qualified responders must come first.",
            esc,
        )

    # S2 - unknown / high-risk: recovery is never investigated
    if assessment.level == Level.HIGH_RISK and proposed == Action.INVESTIGATE_RECOVERY.value:
        return block(
            "S2_NO_RECOVERY_FOR_HIGH_RISK",
            "Recovery cannot be assessed for unidentified or high-risk material; "
            "a qualified person must identify it first.",
            esc if esc not in already else Action.CONCLUDE.value,
        )

    # S3 - unknown / high-risk: a disposal lookup is meaningless before escalation
    if assessment.level == Level.HIGH_RISK and proposed == Action.CHECK_DISPOSAL.value and esc not in already:
        return block(
            "S3_ESCALATE_BEFORE_DISPOSAL",
            "Disposal routes depend on identifying the material; refer it for qualified assessment first.",
            esc,
        )

    # S4 - do not conclude before the mandatory escalation or any investigation
    if proposed == Action.CONCLUDE.value:
        if assessment.level == Level.HIGH_RISK and esc not in already:
            return block("S4_CONCLUDE_NEEDS_ESCALATION",
                         "High-risk cases cannot be concluded without an escalation step.", esc)
        if not already:
            return block("S4_CONCLUDE_NEEDS_EVIDENCE",
                         "Nothing has been investigated yet.", Action.CHECK_EVIDENCE.value)

    # S5 - requesting info only makes sense when something is missing
    if proposed == Action.REQUEST_INFO.value and not assessment.missing_info:
        fallback = Action.CHECK_EVIDENCE.value if Action.CHECK_EVIDENCE.value not in already else Action.CONCLUDE.value
        return block("S5_NOTHING_MISSING", "No information is missing, so there is nothing to request.", fallback)

    # S6 - routine material without immediate danger should not escalate solely
    # because the Decider assigned escalation a high score.
    # Keep the ordinary return below; it remains the default for all other cases.
    if proposed == esc and assessment.level == Level.ROUTINE and not assessment.emergency:
        if Action.INVESTIGATE_RECOVERY.value not in already:
            fallback = Action.INVESTIGATE_RECOVERY.value
        elif Action.CHECK_DISPOSAL.value not in already:
            fallback = Action.CHECK_DISPOSAL.value
        elif Action.CHECK_EVIDENCE.value not in already:
            fallback = Action.CHECK_EVIDENCE.value
        else:
            fallback = Action.CONCLUDE.value
        return block(
            "S6_NO_UNWARRANTED_ESCALATION",
            "Routine material with no immediate-danger indicators does not warrant escalation by itself.",
            fallback,
        )

    return GuardrailResult(True, proposed, proposed, "", "")


UNSAFE_PATTERNS = [
    r"\b(burn|incinerat\w*|set fire)\b",
    r"\b(mix|mixing|combine|blend)\b.{0,30}\b(chemical|acid|solvent|oil|bleach|ammonia)\w*",
    r"\bneutrali[sz]\w*",
    r"\b(dismantl\w*|disassembl\w*|take apart|pry open|crack open|break open|puncture|drill)\b",
    r"\b(heat|boil|distil\w*|evaporat\w*)\b.{0,30}\b(chemical|oil|solvent|battery|acid)\w*",
    r"\b(dissolv\w*|extract\w*)\b.{0,30}\b(metal|gold|copper|lithium)\b",
    r"\bat home\b",
    r"\bdiy\b|\bdo[- ]it[- ]yourself\b",
    r"\b(pour|dump|flush)\b.{0,25}\b(drain|ground|soil|water|sewer)\b",
    r"\b(sniff|taste|touch)\b",
]
_WARNING_PREFIX = r"(?:do not|don't|never|must not|should not|avoid|without|no|not)\s+(?:[\w,'/-]+\s+){0,6}"


def scan_unsafe_instructions(text: str) -> list[str]:
    found: list[str] = []
    lowered = (text or "").lower()
    for pat in UNSAFE_PATTERNS:
        for m in re.finditer(pat, lowered):
            before = lowered[max(0, m.start() - 40):m.start()]
            if re.search(_WARNING_PREFIX + r"$", before):
                continue
            found.append(m.group(0).strip())
    return sorted(set(found))
