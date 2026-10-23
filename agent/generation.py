"""Generative step: Amazon Bedrock through the Strands Agents SDK.

The generated text is ALWAYS labelled as model-generated and unverified, it is
scanned by the safety layer, and it never replaces the source-backed findings.

Imports of `strands` / `boto3` are lazy so the rest of the app (and the demo
mode) works even if those packages are not installed.
"""
from __future__ import annotations

from typing import Protocol

from agent.safety import scan_unsafe_instructions
from agent.schemas import CaseResult, GeneratedText, Label

SYSTEM_PROMPT = (
    "You write a short plain-language summary for a waste-handling decision-support tool.\n"
    "Rules you must follow:\n"
    "1. Use ONLY the facts provided. Do not add facts, numbers, names of companies or facilities.\n"
    "2. Never say that a facility exists, is nearby, is open, is licensed, or accepts a waste type.\n"
    "3. Never give instructions to handle, open, dismantle, mix, heat, burn, neutralise or treat "
    "hazardous or unknown materials. For such materials only tell the reader to refer to qualified "
    "professionals or the State Pollution Control Board.\n"
    "4. Say clearly which parts are source-backed and which are only hypotheses or unknown.\n"
    "5. Maximum 120 words. No headings."
)


class GeneratorUnavailable(RuntimeError):
    """Bedrock generation could not run; `args[0]` is a user-readable reason."""


class Generator(Protocol):
    name: str

    def summarize(self, facts: str) -> GeneratedText: ...


def facts_for_prompt(result: CaseResult) -> str:
    lines = [f"Item: {result.case.waste_name}",
             f"Safety level: {result.assessment.level.value if result.assessment else 'unknown'}",
             f"Recommended pathway: {result.recommendation.pathway.value if result.recommendation else 'none'}",
             "Findings:"]
    for step in result.steps:
        for f in step.findings:
            lines.append(f"- [{f.label.value}] {f.statement}")
    if result.missing_info:
        lines.append("Unknown / missing: " + "; ".join(result.missing_info))
    return "\n".join(lines)


def _explain_exception(exc: Exception) -> str:
    """Map common AWS / network failures to messages a beginner can act on."""
    name = type(exc).__name__
    text = str(exc)
    low = text.lower()
    if name in ("NoCredentialsError", "PartialCredentialsError") or "unable to locate credentials" in low:
        return ("AWS credentials were not found. Run `aws configure` (or set AWS_PROFILE) and try again. "
                "See README step 6.")
    if "accessdenied" in low or "not authorized" in low:
        return ("AWS denied access. Your IAM user/role needs bedrock:InvokeModel (and bedrock:InvokeModelWithResponseStream) "
                "and the model must be enabled for your account. See README step 7.")
    if "resourcenotfound" in low or "model identifier is invalid" in low or "validationexception" in low:
        return ("Bedrock does not recognise that model id in this region, or the request was invalid. "
                "Run `python scripts/check_bedrock.py` to list model ids you can use.")
    if "throttl" in low or "too many requests" in low:
        return "Bedrock is throttling requests. Wait a minute and try again."
    if "endpointconnection" in name.lower() or "could not connect" in low or "connect timeout" in low:
        return "Network problem: could not reach Amazon Bedrock. Check your internet connection and region."
    return f"Bedrock generation failed ({name}): {text[:200]}"


class BedrockGenerator:
    """Amazon Bedrock model driven through a Strands Agent."""

    def __init__(self, model_id: str, region: str):
        self.model_id = model_id
        self.region = region
        self.name = f"Amazon Bedrock via Strands Agents SDK ({model_id})"

    def summarize(self, facts: str) -> GeneratedText:
        if not self.model_id:
            raise GeneratorUnavailable(
                "No Bedrock model id is configured. Set REMAT_BEDROCK_MODEL_ID (see README step 7)."
            )
        try:
            from strands import Agent  # type: ignore
            from strands.models import BedrockModel  # type: ignore
        except ImportError as exc:
            raise GeneratorUnavailable(
                "The Strands Agents SDK is not installed. Run: pip install -r requirements.txt"
            ) from exc
        try:
            model = BedrockModel(model_id=self.model_id, region_name=self.region)
            agent = Agent(model=model, system_prompt=SYSTEM_PROMPT, callback_handler=None)
            reply = agent(facts)
            text = str(reply).strip()
        except Exception as exc:  # network, credentials, throttling, access, bad model id...
            raise GeneratorUnavailable(_explain_exception(exc)) from exc
        if not text:
            raise GeneratorUnavailable("Bedrock returned an empty response.")
        return GeneratedText(
            text=text,
            source=f"bedrock:{self.model_id}",
            label=Label.HYPOTHESIS,
            ai_generated=True,
            note="Generated by an AI model from the findings above. Not verified; do not treat as fact.",
        )


def template_summary(facts: str, reason: str) -> GeneratedText:
    """Deterministic fallback; NOT AI-generated."""
    return GeneratedText(
        text="Template summary (no AI generation was used). See the findings and the decision trace for details.",
        source="template (no AI)",
        label=Label.SAFETY,
        ai_generated=False,
        note=reason,
    )


def guarded_summary(
    generator: Generator | None, result: CaseResult, simulated: bool = False
) -> tuple[GeneratedText, list[str]]:
    """Run the generator (if any), apply the safety scan, and return (summary, warnings)."""
    warnings: list[str] = []
    facts = facts_for_prompt(result)
    if simulated:
        return template_summary(facts, "Demo mode: no generative model was called."), warnings
    if generator is None:
        reason = "No Bedrock generator is configured, so no AI summary was generated."
        warnings.append(reason)
        return template_summary(facts, reason), warnings
    try:
        gen = generator.summarize(facts)
    except GeneratorUnavailable as exc:
        warnings.append(f"Generative summary unavailable: {exc}")
        return template_summary(facts, str(exc)), warnings
    flagged = scan_unsafe_instructions(gen.text)
    if flagged:
        warnings.append("The AI-generated summary was discarded by the safety filter (matched: "
                        + ", ".join(flagged) + ").")
        return template_summary(facts, "Generated text discarded by the safety filter."), warnings
    return gen, warnings
