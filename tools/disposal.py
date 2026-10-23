"""Investigation tool: authorized disposal pathways.

The app bundles NO facility directory. It never states that a facility exists,
is nearby, is open, is licensed or accepts a waste type. It points to the
authority that can verify that.
"""
from __future__ import annotations

from agent.schemas import Finding, Label, SafetyAssessment, WasteCase
from tools.material_lookup import get_material

EMERGENCY_WARNING = (
    "EMERGENCY: if there is a leak, fumes, smoke, fire, swelling or any exposure right now, move away "
    "if it is safe to do so and contact local emergency services or qualified responders immediately "
    "(112 is the unified emergency number in India)."
)


def check_disposal_pathways(case: WasteCase, assessment: SafetyAssessment) -> list[Finding]:
    material = get_material(assessment.material_id)
    findings: list[Finding] = []

    if material is None:
        findings.append(Finding(
            statement="The material is not in the reference dataset, so no disposal pathway can be stated.",
            label=Label.UNKNOWN,
            uncertainties=["Material not identified."],
            next_step="Contact your State Pollution Control Board or local municipal authority and describe the item.",
        ))
    else:
        for p in material["disposal_pathways"]:
            findings.append(Finding(
                statement=f"Disposal pathway: {p['pathway']}",
                label=Label.REFERENCE if p.get("source_ids") else Label.HYPOTHESIS,
                source_ids=list(p.get("source_ids", [])),
                uncertainties=list(p.get("uncertainties", [])),
                next_step=p.get("next_step", ""),
                extra={"authority": p.get("authority", ""), "evidence_level": p.get("evidence_level", "")},
            ))

    where = case.location.strip() or "your area"
    findings.append(Finding(
        statement=f"No verified facility for {where}: this app has no facility directory and does not claim "
                  "that any facility is nearby, open, licensed, or accepting this waste.",
        label=Label.UNKNOWN,
        source_ids=["SRC_SPCB", "SRC_CPCB"],
        uncertainties=["Facility existence, licence status, opening hours and acceptance are all unverified."],
        next_step="Ask the State Pollution Control Board (or the equivalent body) for currently authorized "
                  "handlers, then confirm acceptance with the handler before transporting anything.",
        extra={"facility_status": "NOT_VERIFIED"},
    ))
    if assessment.emergency:
        findings.insert(0, Finding(statement=EMERGENCY_WARNING, label=Label.SAFETY, source_ids=["SRC_EMERGENCY"]))
    return findings
