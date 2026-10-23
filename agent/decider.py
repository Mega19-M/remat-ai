
"""Decision backends: real Decider client and clearly labelled simulation.

Hosted endpoint:
    POST https://<host>/v1/decide

Local Strands Decider:
    POST http://127.0.0.1:8000/v1/systemone

The hosted API must support the request format documented by its provider.
The simulation is not a real AI model.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional, Protocol

from agent.schemas import (
    Action,
    DeciderDecision,
    Level,
    SafetyAssessment,
    WasteCase,
)

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

    def choose(
        self, state: str, options: list[str]
    ) -> DeciderDecision: ...


# ------------------------------------------------------------------
# Build the decision state
# ------------------------------------------------------------------

def build_state(
    case: WasteCase,
    assessment: SafetyAssessment,
    history: list[tuple[str, int]],
) -> str:
    """Build a plain-text description containing known case information."""
    lines = [
        "WASTE CASE (user-provided, unverified)",
        f"Item: {case.waste_name or '(not given)'}",
    ]

    if case.description:
        lines.append(f"Description: {case.description}")

    lines.extend([
        f"Category stated by user: {case.category or '(not given)'}",
        f"Location: {case.location or '(not given)'}",
        (
            "Danger or contamination information: "
            f"{case.danger_info or '(none given)'}"
        ),
        f"User objective: {case.objective}",
    ])

    if case.photo_attached:
        lines.append(
            "Photo: attached but NOT analyzed by any AI model."
        )

    lines.extend([
        "",
        "DETERMINISTIC SAFETY ASSESSMENT",
        f"Level: {assessment.level.value}",
        (
            "Possible immediate danger: "
            f"{'yes' if assessment.emergency else 'no'}"
        ),
    ])

    for reason in assessment.reasons:
        lines.append(f"- {reason}")

    lines.extend([
        f"Missing information items: {len(assessment.missing_info)}",
        "",
        "INVESTIGATION SO FAR",
    ])

    if not history:
        lines.append("Nothing has been investigated yet.")

    for action, finding_count in history:
        lines.append(
            f"- {action}: {finding_count} finding(s) recorded"
        )

    return "\n".join(lines)


# ------------------------------------------------------------------
# Parse a choice returned by the server
# ------------------------------------------------------------------

_SELECTED_KEYS = (
    "choice",
    "selected",
    "label",
    "option",
    "answer",
    "top",
    "value",
    "prediction",
    "predicted",
)

_PROB_KEYS = (
    "probabilities",
    "probs",
    "distribution",
    "scores",
    "options",
)


def _match_option(
    value: Any, options: list[str]
) -> Optional[str]:
    if isinstance(value, str):
        for option in options:
            if value.strip().lower() == option.lower():
                return option

    return None


def parse_choice_answer(
    answer: Any, options: list[str]
) -> dict[str, Any]:
    """Extract a selected option and optional confidence/probabilities."""
    selected: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Optional[dict[str, float]] = None
    derived_from = ""

    if isinstance(answer, str):
        selected = _match_option(answer, options)

    elif isinstance(answer, dict):
        inner = (
            answer["result"]
            if isinstance(answer.get("result"), dict)
            else answer
        )

        for key in _SELECTED_KEYS:
            if key in inner:
                selected = _match_option(inner[key], options)
                if selected is not None:
                    break

        raw_confidence = inner.get("confidence")
        if (
            isinstance(raw_confidence, (int, float))
            and not isinstance(raw_confidence, bool)
        ):
            confidence = float(raw_confidence)

        for key in _PROB_KEYS:
            candidate = inner.get(key)

            if isinstance(candidate, dict) and candidate:
                parsed: dict[str, float] = {}

                for label, value in candidate.items():
                    option = _match_option(label, options)

                    if (
                        option is not None
                        and isinstance(value, (int, float))
                        and not isinstance(value, bool)
                    ):
                        parsed[option] = float(value)

                if parsed:
                    probabilities = parsed
                    break

        if selected is None and probabilities:
            selected = max(
                probabilities,
                key=probabilities.get,
            )
            derived_from = (
                "Selected from the probabilities returned by "
                "the server; no explicit selected field was found."
            )

    if selected is None:
        raise DeciderResponseError(
            "Could not determine the selected option from "
            "the Decider response.",
            raw=answer,
        )

    return {
        "selected": selected,
        "confidence": confidence,
        "probabilities": probabilities,
        "derived_from": derived_from,
    }


# ------------------------------------------------------------------
# Real HTTP backend
# ------------------------------------------------------------------

class HttpDeciderBackend:
    """HTTP client for a local or hosted Decider service."""

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

        self.hosted = (
            self.base_url.lower().startswith("https://")
            or bool(self.api_key)
        )

        if self.hosted:
            self.endpoint = "/v1/decide"
            self.question_name = HOSTED_QUESTION_NAME
        else:
            self.endpoint = "/v1/systemone"
            self.question_name = LOCAL_QUESTION_NAME

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": "REMAT-AI/1.0",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    def health(self) -> dict[str, Any]:
        """Check /health and distinguish HTTP rejection from connection errors."""
        request = urllib.request.Request(
            self.base_url + "/health",
            headers=self._headers(),
            method="GET",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=min(self.timeout_s, 10),
            ) as response:
                raw_text = response.read().decode(
                    "utf-8", "replace"
                )

                try:
                    result = json.loads(raw_text)
                except ValueError as exc:
                    raise DeciderResponseError(
                        "Health endpoint returned invalid JSON.",
                        raw=raw_text[:1000],
                    ) from exc

                if not isinstance(result, dict):
                    raise DeciderResponseError(
                        "Health endpoint did not return a JSON object.",
                        raw=result,
                    )

                return result

        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(
                "utf-8", "replace"
            )[:500]

            raise DeciderResponseError(
                f"Health check returned HTTP {exc.code}. "
                f"{detail or 'No error details supplied.'}",
                raw={"status": exc.code, "body": detail},
            ) from exc

        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DeciderUnavailable(
                f"Cannot connect to Decider at {self.base_url}: "
                f"{exc}"
            ) from exc

    def request_body(
        self, state: str, options: list[str]
    ) -> dict[str, Any]:
        unknown_options = [
            option
            for option in options
            if option not in ACTION_CRITERIA
        ]

        if unknown_options:
            raise ValueError(
                f"Unknown decision options: {unknown_options}"
            )

        body: dict[str, Any] = {
            "state": state,
            "questions": {
                self.question_name: {
                    "type": "choice",
                    "instructions": QUESTION_TEXT,
                    "criteria": {
                        option: ACTION_CRITERIA[option]
                        for option in options
                    },
                }
            },
        }

        if self.model_field:
            body["model"] = self.model_field

        return body

    def choose(
        self, state: str, options: list[str]
    ) -> DeciderDecision:
        if not options:
            raise ValueError(
                "At least one decision option is required."
            )

        body = json.dumps(
            self.request_body(state, options)
        ).encode("utf-8")

        headers = self._headers()
        headers["Content-Type"] = "application/json"

        request = urllib.request.Request(
            self.base_url + self.endpoint,
            data=body,
            method="POST",
            headers=headers,
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_s,
            ) as response:
                response_text = response.read().decode(
                    "utf-8", "replace"
                )

        except urllib.error.HTTPError as exc:
            # Do not expose request headers or the API key.
            detail = exc.read().decode(
                "utf-8", "replace"
            )[:500]

            raise DeciderResponseError(
                f"Decision request returned HTTP {exc.code}. "
                f"{detail or 'The server supplied no details.'}",
                raw={
                    "status": exc.code,
                    "body": detail,
                },
            ) from exc

        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DeciderUnavailable(
                f"Cannot connect to Decider at {self.base_url}: "
                f"{exc}"
            ) from exc

        try:
            raw = json.loads(response_text)
        except ValueError as exc:
            raise DeciderResponseError(
                "Decider server returned invalid JSON.",
                raw=response_text[:1000],
            ) from exc

        if not isinstance(raw, dict):
            raise DeciderResponseError(
                "Decider response is not a JSON object.",
                raw=raw,
            )

        answers = raw.get("answers")

        if not isinstance(answers, dict):
            raise DeciderResponseError(
                "Decider response has no answers object.",
                raw=raw,
            )

        answer = answers.get(self.question_name)

        if answer is None:
            answer = answers.get(LOCAL_QUESTION_NAME)

        if answer is None:
            answer = answers.get(HOSTED_QUESTION_NAME)

        if answer is None:
            raise DeciderResponseError(
                "Decider response has no answer for the question.",
                raw=raw,
            )

        parsed = parse_choice_answer(answer, options)

        latency = raw.get("latency_ms")
        model_name = (
            raw.get("model")
            if isinstance(raw.get("model"), str)
            else ""
        )

        if self.hosted:
            backend_name = (
                f"Strands Decider (hosted: {model_name})"
                if model_name
                else "Strands Decider (hosted endpoint)"
            )
        else:
            backend_name = (
                f"Strands Decider (local: {model_name})"
                if model_name
                else "Strands Decider 2B (live, local server)"
            )

        return DeciderDecision(
            backend=backend_name,
            live=True,
            question=QUESTION_TEXT,
            options=list(options),
            selected=parsed["selected"],
            confidence=parsed["confidence"],
            probabilities=parsed["probabilities"],
            latency_ms=(
                float(latency)
                if isinstance(latency, (int, float))
                and not isinstance(latency, bool)
                else None
            ),
            selected_derived_from=parsed["derived_from"],
            raw_response=raw,
        )


# ------------------------------------------------------------------
# Simulation backend — never represented as a real model
# ------------------------------------------------------------------

class SimulatedDeciderBackend:
    """Deterministic demonstration rules, not a machine-learning model."""

    name = "SIMULATED rules (not Strands Decider 2B)"
    live = False

    def __init__(
        self,
        case: WasteCase,
        assessment: SafetyAssessment,
    ):
        self.case = case
        self.assessment = assessment

    def choose(
        self, state: str, options: list[str]
    ) -> DeciderDecision:
        if not options:
            raise ValueError(
                "At least one decision option is required."
            )

        objective = self.case.objective

        if objective == "recover":
            order = [
                "investigate_recovery",
                "check_disposal",
                "check_evidence",
                "request_info",
                "conclude",
            ]
        elif objective == "dispose":
            order = [
                "check_disposal",
                "check_evidence",
                "investigate_recovery",
                "request_info",
                "conclude",
            ]
        else:
            order = [
                "investigate_recovery",
                "check_disposal",
                "check_evidence",
                "request_info",
                "conclude",
            ]

        if self.assessment.emergency:
            order = ["escalate"] + order
        elif (
            self.assessment.level == Level.HIGH_RISK
            and objective != "recover"
        ):
            order = ["escalate"] + order

        selected = next(
            (option for option in order if option in options),
            options[0],
        )

        return DeciderDecision(
            backend=self.name,
            live=False,
            question=QUESTION_TEXT,
            options=list(options),
            selected=selected,
            rationale=(
                "Simulated fixed-order rules based on the user's "
                "objective. No AI model was run."
            ),
        )
