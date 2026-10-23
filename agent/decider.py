"""Decision backends: live Decider client and clearly labelled simulation.

This module supports:
- Hosted Decider POST /v1/decide
- Local Strands Decider POST /v1/systemone
- Explicitly simulated rules for demos

The hosted endpoint previously returned a validation error indicating that it
expects NoulQuestion requests (type="noul", criteria keys "true"/"false").
The hosted client below sends one Noul question per available action and
supports several common answer shapes. If the server's actual response schema
differs, it raises a safe diagnostic error containing answer shapes only.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional, Protocol

from agent.schemas import Action, DeciderDecision, Level, SafetyAssessment, WasteCase

LOCAL_QUESTION_NAME = "next_action"
HOSTED_QUESTION_NAME = "action"

QUESTION_TEXT = (
    "Given the waste case and the investigation so far, "
    "which single investigation action should run next?"
)

ACTION_CRITERIA: dict[str, str] = {
    Action.INVESTIGATE_RECOVERY.value:
        "Look up possible reuse or recycling options for this material.",
    Action.CHECK_DISPOSAL.value:
        "Look up the authorized collection, treatment or disposal pathway.",
    Action.CHECK_EVIDENCE.value:
        "Check the sources behind the findings and list what is unknown.",
    Action.ESCALATE.value:
        "Refer the case to a qualified professional because it may be hazardous.",
    Action.REQUEST_INFO.value:
        "Ask the user for missing information before going further.",
    Action.CONCLUDE.value:
        "Stop investigating because enough has been checked for a recommendation.",
}


class DeciderError(RuntimeError):
    """Base class for decision-backend failures."""


class DeciderUnavailable(DeciderError):
    """The server could not be reached."""


class DeciderResponseError(DeciderError):
    """The server responded, but the response was unsuccessful or invalid."""

    def __init__(self, message: str, raw: Any = None):
        super().__init__(message)
        self.raw = raw


class DeciderBackend(Protocol):
    name: str
    live: bool

    def choose(self, state: str, options: list[str]) -> DeciderDecision: ...


def build_state(
    case: WasteCase,
    assessment: SafetyAssessment,
    history: list[tuple[str, int]],
) -> str:
    """Build a plain-text description of the case and current investigation."""
    lines = [
        "WASTE CASE (user-provided, unverified)",
        f"Item: {case.waste_name or '(not given)'}",
        f"Description: {case.description or '(not given)'}",
        f"Category stated by user: {case.category or '(not given)'}",
        f"Location: {case.location or '(not given)'}",
        f"Danger or contamination information: {case.danger_info or '(none given)'}",
        f"User objective: {case.objective}",
    ]

    if getattr(case, "photo_attached", False):
        lines.append("Photo: attached but not analyzed by this backend.")

    lines.extend([
        "",
        "DETERMINISTIC SAFETY ASSESSMENT",
        f"Level: {assessment.level.value}",
        f"Possible immediate danger: {'yes' if assessment.emergency else 'no'}",
    ])
    lines.extend(f"- {reason}" for reason in assessment.reasons)
    lines.extend([
        f"Missing information items: {len(assessment.missing_info)}",
        "",
        "INVESTIGATION SO FAR",
    ])

    if not history:
        lines.append("Nothing has been investigated yet.")

    for action, finding_count in history:
        lines.append(f"- {action}: {finding_count} finding(s) recorded")

    return "\n".join(lines)


def _match_option(value: Any, options: list[str]) -> Optional[str]:
    if isinstance(value, str):
        cleaned = value.strip().lower()
        for option in options:
            if cleaned == option.lower():
                return option
    return None


def _answer_shape(value: Any) -> dict[str, Any]:
    """Return safe shape metadata, not arbitrary server payload contents."""
    if isinstance(value, dict):
        return {"type": "object", "keys": list(value.keys())[:30]}
    if isinstance(value, list):
        return {"type": "array", "length": len(value)}
    if isinstance(value, str):
        return {"type": "string", "sample": value[:80]}
    if value is None:
        return {"type": "null"}
    return {"type": type(value).__name__, "sample": str(value)[:80]}


def _parse_bool(answer: Any) -> Optional[bool]:
    """Interpret common yes/no response structures without guessing."""
    if isinstance(answer, bool):
        return answer

    if isinstance(answer, (int, float)) and not isinstance(answer, bool):
        if answer == 1:
            return True
        if answer == 0:
            return False

    if isinstance(answer, str):
        value = answer.strip().lower()
        if value in {"true", "yes", "1", "positive"}:
            return True
        if value in {"false", "no", "0", "negative"}:
            return False

    if isinstance(answer, dict):
        # Common wrapper fields
        for key in ("answer", "value", "prediction", "selected", "label", "choice", "result", "noul"):
            if key in answer:
                parsed = _parse_bool(answer[key])
                if parsed is not None:
                    return parsed

        # Some services return a criterion-score mapping.
        for true_key, false_key in (
            ("true", "false"),
            ("True", "False"),
            ("yes", "no"),
        ):
            if true_key in answer and false_key in answer:
                true_val, false_val = answer[true_key], answer[false_key]
                if isinstance(true_val, bool) and isinstance(false_val, bool):
                    if true_val != false_val:
                        return true_val
                if (
                    isinstance(true_val, (int, float))
                    and isinstance(false_val, (int, float))
                ):
                    if true_val != false_val:
                        return true_val > false_val

        # Some answers use a nested list of candidates with a name/score.
        for key in ("scores", "probabilities", "distribution", "options", "predictions"):
            value = answer.get(key)
            if isinstance(value, dict) and "true" in value and "false" in value:
                t, f = value["true"], value["false"]
                if isinstance(t, (int, float)) and isinstance(f, (int, float)) and t != f:
                    return t > f

    return None


def parse_choice_answer(answer: Any, options: list[str]) -> dict[str, Any]:
    """Extract a multi-choice answer from common response shapes."""
    selected: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Optional[dict[str, float]] = None
    derived_from = ""

    if isinstance(answer, str):
        selected = _match_option(answer, options)

    elif isinstance(answer, dict):
        inner = answer.get("result")
        if not isinstance(inner, dict):
            inner = answer

        for key in (
            "choice", "selected", "label", "option", "answer",
            "top", "value", "prediction", "predicted",
        ):
            if key in inner:
                selected = _match_option(inner[key], options)
                if selected is not None:
                    break

        raw_confidence = inner.get("confidence")
        if isinstance(raw_confidence, (int, float)) and not isinstance(raw_confidence, bool):
            confidence = float(raw_confidence)

        for key in ("probabilities", "probs", "distribution", "scores", "options"):
            candidate = inner.get(key)
            if isinstance(candidate, dict):
                parsed: dict[str, float] = {}
                for label, value in candidate.items():
                    option = _match_option(label, options)
                    if option is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
                        parsed[option] = float(value)
                if parsed:
                    probabilities = parsed
                    break

        if selected is None and probabilities:
            selected = max(probabilities, key=probabilities.get)
            derived_from = "Selected from the probability distribution in the response."

    if selected is None:
        raise DeciderResponseError(
            "Could not determine the selected option from the Decider response.",
            raw={"answer_shape": _answer_shape(answer)},
        )

    return {
        "selected": selected,
        "confidence": confidence,
        "probabilities": probabilities,
        "derived_from": derived_from,
    }


class HttpDeciderBackend:
    """HTTP client for local or hosted Decider services."""

    name = "Strands Decider 2B (live)"
    live = True

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout_s: float = 60.0,
        model_field: str = "",
        api_key: str = "",
    ):
        self.base_url = base_url.strip().rstrip("/")
        self.timeout_s = timeout_s
        self.model_field = model_field.strip()
        self.api_key = api_key.strip()
        self.hosted = self.base_url.lower().startswith("https://") or bool(self.api_key)

        if self.hosted:
            self.endpoint = "/v1/decide"
            self.question_name = HOSTED_QUESTION_NAME
        else:
            self.endpoint = "/v1/systemone"
            self.question_name = LOCAL_QUESTION_NAME

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": "REMAT-AI/1.0"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def health(self) -> dict[str, Any]:
        request = urllib.request.Request(
            self.base_url + "/health",
            headers=self._headers(),
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=min(self.timeout_s, 10)) as response:
                response_text = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise DeciderResponseError(
                f"Health check returned HTTP {exc.code}. {detail}",
                raw={"status": exc.code, "body": detail},
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DeciderUnavailable(f"Cannot connect to Decider at {self.base_url}: {exc}") from exc

        try:
            result = json.loads(response_text)
        except ValueError as exc:
            raise DeciderResponseError(
                "Health endpoint returned invalid JSON.",
                raw={"body_sample": response_text[:200]},
            ) from exc
        if not isinstance(result, dict):
            raise DeciderResponseError("Health endpoint did not return a JSON object.")
        return result

    def request_body(self, state: str, options: list[str]) -> dict[str, Any]:
        unknown = [option for option in options if option not in ACTION_CRITERIA]
        if unknown:
            raise ValueError(f"Unknown decision options: {unknown}")
        if not options:
            raise ValueError("At least one decision option is required.")

        if not self.hosted:
            questions: dict[str, Any] = {
                self.question_name: {
                    "type": "choice",
                    "instructions": QUESTION_TEXT,
                    "criteria": {option: ACTION_CRITERIA[option] for option in options},
                }
            }
        else:
            # The hosted validator complained that questions.action must be
            # NoulQuestion. Each candidate therefore gets a binary question.
            questions = {}
            for index, option in enumerate(options, start=1):
                questions[f"action_{index}"] = {
                    "type": "noul",
                    "instructions": (
                        "Answer true only if this is the single best next "
                        "investigation action for the case. Otherwise answer false.\n"
                        f"Candidate: {option}\n"
                        f"Description: {ACTION_CRITERIA[option]}\n"
                        f"Other available actions: {', '.join(x for x in options if x != option)}"
                    ),
                    "criteria": {
                        "true": f"{option} is the best next action.",
                        "false": f"{option} is not the best next action.",
                    },
                }

        body: dict[str, Any] = {"state": state, "questions": questions}
        if self.model_field:
            body["model"] = self.model_field
        return body

    def choose(self, state: str, options: list[str]) -> DeciderDecision:
        if not options:
            raise ValueError("At least one decision option is required.")

        body = json.dumps(self.request_body(state, options)).encode("utf-8")
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + self.endpoint,
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                response_text = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:1500]
            raise DeciderResponseError(
                f"Decision request returned HTTP {exc.code}. {detail or 'No details supplied.'}",
                raw={"status": exc.code, "body": detail},
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DeciderUnavailable(f"Cannot connect to Decider at {self.base_url}: {exc}") from exc

        try:
            raw = json.loads(response_text)
        except ValueError as exc:
            raise DeciderResponseError(
                "Decider server returned invalid JSON.",
                raw={"body_sample": response_text[:200]},
            ) from exc

        if not isinstance(raw, dict) or not isinstance(raw.get("answers"), dict):
            raise DeciderResponseError(
                "Decider response has no answers object.",
                raw={"response_shape": _answer_shape(raw)},
            )

        answers = raw["answers"]

        if self.hosted:
            # The hosted API returns Noul answers as probabilities, e.g.
            # {"type": "noul", "noul": 0.93}. These are not booleans.
            # Compare the probability for each candidate and select the
            # highest-scoring candidate; requiring exactly one score >= 0.5
            # is incorrect because multiple candidates can exceed 0.5.
            scores: dict[str, float] = {}
            unknown: dict[str, Any] = {}

            for index, option in enumerate(options, start=1):
                question_key = f"action_{index}"
                answer = answers.get(question_key)

                if answer is None and index == 1:
                    answer = answers.get(HOSTED_QUESTION_NAME)

                if answer is None:
                    unknown[question_key] = {"type": "missing"}
                    continue

                probability = None
                if isinstance(answer, dict) and "noul" in answer:
                    candidate = answer["noul"]
                    if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                        if 0.0 <= float(candidate) <= 1.0:
                            probability = float(candidate)

                if probability is None:
                    parsed = _parse_bool(answer)
                    if parsed is not None:
                        probability = 1.0 if parsed else 0.0

                if probability is None:
                    unknown[question_key] = _answer_shape(answer)
                else:
                    scores[option] = probability

            if unknown:
                raise DeciderResponseError(
                    "Hosted Decider response could not be interpreted. "
                    f"Answer shapes: {unknown}",
                    raw={"answer_shapes": unknown},
                )

            if not scores:
                raise DeciderResponseError(
                    "Hosted Decider returned no usable Noul probabilities.",
                    raw={"answer_keys": list(answers.keys())[:30]},
                )

            selected = max(scores, key=scores.get)
            confidence = scores[selected]
            probabilities = scores
            derived_from = (
                "Selected the action with the highest hosted Decider Noul "
                "probability across the available candidates."
            )
        else:
            answer = answers.get(self.question_name)
            if answer is None:
                raise DeciderResponseError(
                    f"Decider response has no answer for '{self.question_name}'.",
                    raw={"answer_keys": list(answers.keys())[:30]},
                )
            parsed = parse_choice_answer(answer, options)
            selected = parsed["selected"]
            confidence = parsed["confidence"]
            probabilities = parsed["probabilities"]
            derived_from = parsed["derived_from"]

        latency = raw.get("latency_ms")
        model_name = raw.get("model") if isinstance(raw.get("model"), str) else ""
        if self.hosted:
            backend_name = f"Strands Decider (hosted: {model_name})" if model_name else "Strands Decider (hosted endpoint)"
        else:
            backend_name = f"Strands Decider (local: {model_name})" if model_name else "Strands Decider 2B (live, local server)"

        return DeciderDecision(
            backend=backend_name,
            live=True,
            question=QUESTION_TEXT,
            options=list(options),
            selected=selected,
            confidence=confidence,
            probabilities=probabilities,
            latency_ms=(
                float(latency)
                if isinstance(latency, (int, float)) and not isinstance(latency, bool)
                else None
            ),
            selected_derived_from=derived_from,
            raw_response=raw,
        )


class SimulatedDeciderBackend:
    """Deterministic demo rules; this is not a machine-learning model."""

    name = "SIMULATED rules (not Strands Decider 2B)"
    live = False

    def __init__(self, case: WasteCase, assessment: SafetyAssessment):
        self.case = case
        self.assessment = assessment

    def choose(self, state: str, options: list[str]) -> DeciderDecision:
        if not options:
            raise ValueError("At least one decision option is required.")

        if self.case.objective == "recover":
            order = [
                Action.INVESTIGATE_RECOVERY.value,
                Action.CHECK_DISPOSAL.value,
                Action.CHECK_EVIDENCE.value,
                Action.REQUEST_INFO.value,
                Action.CONCLUDE.value,
            ]
        elif self.case.objective == "dispose":
            order = [
                Action.CHECK_DISPOSAL.value,
                Action.CHECK_EVIDENCE.value,
                Action.INVESTIGATE_RECOVERY.value,
                Action.REQUEST_INFO.value,
                Action.CONCLUDE.value,
            ]
        else:
            order = [
                Action.INVESTIGATE_RECOVERY.value,
                Action.CHECK_DISPOSAL.value,
                Action.CHECK_EVIDENCE.value,
                Action.REQUEST_INFO.value,
                Action.CONCLUDE.value,
            ]

        if self.assessment.emergency or self.assessment.level == Level.HIGH_RISK:
            order = [Action.ESCALATE.value] + order

        selected = next((option for option in order if option in options), options[0])
        return DeciderDecision(
            backend=self.name,
            live=False,
            question=QUESTION_TEXT,
            options=list(options),
            selected=selected,
            rationale="Simulated fixed-order rules. No AI model was run.",
        )
