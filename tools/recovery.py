"""Investigation tool: recovery options (reuse / recycling) from the curated dataset."""
from __future__ import annotations

from agent.schemas import Finding, Label, Level, SafetyAssessment, WasteCase
from tools.material_lookup import get_material


def investigate_recovery(case: WasteCase, assessment: SafetyAssessment) -> list[Finding]:
    """Return source-backed recovery options for the matched material.

    Never returns do-it-yourself processing instructions. For high-risk or
    unidentified material it returns a single refusal finding.
    """
    material = get_material(assessment.material_id)
    if assessment.level == Level.HIGH_RISK or material is None or not material.get("recovery_options"):
        return [Finding(
            statement="No recovery option can be assessed for unidentified or high-risk material. "
                      "It must be identified by a qualified person before any reuse or recycling is considered.",
            label=Label.SAFETY,
            source_ids=["SRC_SPCB"],
            uncertainties=["The material has not been identified."],
            next_step="Refer the material for qualified assessment (see the escalation step).",
        )]

    findings: list[Finding] = []
    for opt in material["recovery_options"]:
        regulated = assessment.level == Level.REGULATED
        stmt = f"Potential recovery option: {opt['option']}."
        if regulated:
            stmt += " Only through authorized channels; no do-it-yourself processing."
        findings.append(Finding(
            statement=stmt,
            label=Label.REFERENCE if opt.get("source_ids") else Label.HYPOTHESIS,
            source_ids=list(opt.get("source_ids", [])),
            uncertainties=list(opt.get("uncertainties", [])),
            next_step=opt.get("next_step", ""),
            extra={
                "material_compatibility": opt.get("compatibility", ""),
                "professional_processing": opt.get("professional_processing", ""),
                "evidence_level": opt.get("evidence_level", ""),
            },
        ))
    return findings
