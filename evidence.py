"""Investigation tools: evidence check, missing-information request, escalation referral."""
from __future__ import annotations

from agent.schemas import Finding, Label, Level, SafetyAssessment, WasteCase
from tools.disposal import EMERGENCY_WARNING
from tools.material_lookup import get_material, load_sources


def check_evidence(case: WasteCase, assessment: SafetyAssessment, prior_findings: list[Finding]) -> list[Finding]:
    """Audit what the investigation has so far: which statements have sources, what is unknown."""
    sources = load_sources()
    findings: list[Finding] = []
    material = get_material(assessment.material_id)

    if material is None:
        findings.append(Finding(
            statement="No reference material matched: nothing in the curated dataset supports a pathway for this item.",
            label=Label.UNKNOWN,
            next_step="Provide more detail or refer the item to a qualified assessor.",
        ))
    else:
        findings.append(Finding(
            statement=f"Matched reference material: {material['name']} (terms: {', '.join(assessment.matched_terms)}). "
                      "The match is keyword-based and should be confirmed by you.",
            label=Label.REFERENCE,
            source_ids=sorted({s for opt in material.get("recovery_options", []) + material.get("disposal_pathways", [])
                               for s in opt.get("source_ids", [])}),
            uncertainties=["Keyword matching can be wrong; confirm that the description fits the reference material."],
        ))

    unsourced = [f for f in prior_findings if f.label == Label.HYPOTHESIS and not f.source_ids]
    sourced = [f for f in prior_findings if f.source_ids]
    findings.append(Finding(
        statement=f"Evidence audit: {len(sourced)} earlier finding(s) cite a source; "
                  f"{len(unsourced)} earlier finding(s) are unsourced hypotheses.",
        label=Label.SAFETY,
    ))

    cited = sorted({s for f in prior_findings for s in f.source_ids})
    for sid in cited:
        src = sources.get(sid)
        if not src:
            continue
        checked = src.get("date_checked") or "not checked by this project"
        findings.append(Finding(
            statement=f"Source {sid}: {src['title']} - {src['authority']}. Date checked: {checked}. "
                      f"Check method: {src['check_method']}.",
            label=Label.REFERENCE,
            source_ids=[sid],
            uncertainties=["Clause-level wording is not verified by this project."] if src.get("date_checked") else
                          ["This source was not checked by this project."],
            next_step=src.get("find_at", ""),
        ))

    for item in assessment.missing_info:
        findings.append(Finding(statement=item, label=Label.UNKNOWN))
    return findings


def request_missing_info(case: WasteCase, assessment: SafetyAssessment) -> list[Finding]:
    questions = []
    if any("category" in m.lower() for m in assessment.missing_info):
        questions.append("What kind of waste is it (for example electronic, agricultural, oil, chemical)?")
    if any("location" in m.lower() for m in assessment.missing_info):
        questions.append("Which town or district are you in?")
    if any("contamination" in m.lower() or "danger" in m.lower() for m in assessment.missing_info):
        questions.append("Is there any leak, label, smell, mixing or contamination you can observe from a safe distance?")
    if any("photo" in m.lower() for m in assessment.missing_info):
        questions.append("Does the attached photo show the item described? (The photo is not analyzed by AI.)")
    if not questions:
        questions.append("No further information is required for this case.")
    return [Finding(statement="Question for the user: " + q, label=Label.UNKNOWN) for q in questions]


def escalate_to_professional(case: WasteCase, assessment: SafetyAssessment) -> list[Finding]:
    """Referral guidance. Deliberately contains no handling or treatment instructions."""
    findings: list[Finding] = []
    if assessment.emergency:
        findings.append(Finding(statement=EMERGENCY_WARNING, label=Label.SAFETY, source_ids=["SRC_EMERGENCY"]))
    findings.append(Finding(
        statement="Refer this material for qualified assessment. Do not handle, open, move, mix, test or "
                  "treat it yourself, and keep other people away from it.",
        label=Label.SAFETY,
        source_ids=["SRC_SPCB", "SRC_HOWM_RULES_2016"],
        next_step="Contact your State Pollution Control Board or a qualified hazardous-waste professional.",
    ))
    findings.append(Finding(
        statement="Information worth preparing for the assessor (observed from a safe distance only): what the "
                  "container looks like, any labels or markings, approximate quantity, exact location, and "
                  "how long it has been there.",
        label=Label.SAFETY,
        next_step="Write these details down; do not approach or open the container to collect them.",
    ))
    if assessment.level == Level.HIGH_RISK:
        findings.append(Finding(
            statement="Whether the material can be recovered, reused or disposed of through a specific route is "
                      "UNKNOWN until it has been identified.",
            label=Label.UNKNOWN,
        ))
    return findings
