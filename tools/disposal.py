"""Investigation tool: context-aware disposal pathways.

The app bundles NO facility directory. Routine agricultural residue is routed to
local agricultural guidance; the generic facility warning is reserved for
regulated or unidentified/high-risk materials where authorized handling matters.
"""
from __future__ import annotations

from agent.schemas import Finding, Label, Level, SafetyAssessment, WasteCase
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
            source_ids = list(p.get("source_ids", []))
            # CPCB/SPCB entries in this project's source registry describe
            # hazardous waste, e-waste and used-oil authorization—not crop-straw
            # management. Do not attach those citations to agricultural residue.
            if material.get("id") == "agri_residue":
                source_ids = [
                    sid for sid in source_ids
                    if sid not in {"SRC_CPCB", "SRC_SPCB"}
                ]
            findings.append(Finding(
                statement=f"Disposal pathway: {p['pathway']}",
                label=Label.REFERENCE if source_ids else Label.HYPOTHESIS,
                source_ids=source_ids,
                uncertainties=list(p.get("uncertainties", [])),
                next_step=p.get("next_step", ""),
                extra={"authority": p.get("authority", ""), "evidence_level": p.get("evidence_level", "")},
            ))

        # A generic authorized-facility check is useful for regulated or
        # unidentified/high-risk waste, but misleading for routine crop residue.
        if material.get("hazard_class") != "routine":
            where = case.location.strip() or "your area"
            findings.append(Finding(
                statement=f"No verified authorized handler for {where}: this app has no facility directory "
                          "and does not claim that any facility is nearby, open, licensed, or accepting this waste.",
                label=Label.UNKNOWN,
                source_ids=["SRC_SPCB", "SRC_CPCB"],
                uncertainties=[
                    "Handler existence, authorization status, opening hours and acceptance are unverified."
                ],
                next_step="Ask the State Pollution Control Board (or the equivalent body) for currently "
                          "authorized handlers, then confirm acceptance with the handler before transporting anything.",
                extra={"facility_status": "NOT_VERIFIED"},
            ))

    if assessment.emergency:
        findings.insert(0, Finding(
            statement=EMERGENCY_WARNING,
            label=Label.SAFETY,
            source_ids=["SRC_EMERGENCY"],
        ))
    return findings
