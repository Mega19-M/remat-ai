"""Decision backends: the REAL Strands Decider 2B client and a clearly-labelled simulation.

What is verified (official model card / README, checked 2026-10-19)
-------------------------------------------------------------------
* `pip install strands-decider`
* `strands-decider serve <model-id> --port 8000` starts a local HTTP server
  (binds to 127.0.0.1, no authentication, for local experiments only).
* `POST /v1/systemone` with JSON
  {"state": "...", "questions": {"<name>": {"type": "choice", "instructions": "...",
   "criteria": {"<option>": "<description>", ...}}}}
  returns JSON with `model`, `answers` (keyed by question name), `usage`, `latency_ms`.
* A `choice` answer carries the chosen option, a confidence and a probability per option.
* `GET /health` returns model/device information.

What is NOT documented (so this client is defensive and never guesses)
----------------------------------------------------------------------
The exact JSON field names inside a `choice` answer are not spelled out in the
public docs we could read. `parse_choice_answer` therefore accepts several
plausible shapes, and raises `DeciderResponseError` (carrying the raw body)
when it cannot determine the selected option. It never invents a selection,
confidence or probability. Run `python scripts/check_decider.py` to see the raw
response from your own server.
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
    "Given the waste case and the investigation so far, which single investigation action "
    "should run next?"
)

# One short, distinct description per option. These are the typed option set.
ACTION_CRITERIA: dict[str, str] = {
    Action.INVESTIGATE_RECOVERY.value: "look up possible reuse or recycling options for this material",
    Action.CHECK_DISPOSAL.value: "look up the authorized collection, treatment or disposal pathway for this material",
    Action.CHECK_EVIDENCE.value: "check the sources behind the current findings and list what is unknown",
    Action.ESCALATE.value: "refer the case to a qualified professional because it is unknown, hazardous or dangerous",
    Action.REQUEST_INFO.value: "ask the user for missing information before going further",
    Action.CONCLUDE.value: "stop investigating because enough has been checked to give a recommendation",
}


class DeciderError(RuntimeError):
    """Base class for decision-backend failures."""


class DeciderUnavailable(DeciderError):
    """The Decider server could not be reached (not running, wrong URL, timeout)."""


class DeciderResponseError(DeciderError):
    """The server answered but the answer could not be understood."""

    def __init__(self, message: str, raw: Any = None):
        super().__init__(message)
        self.raw = raw


class DeciderBackend(Protocol):
    name: str
    live: bool

    def choose(self, state: str, options: list[str]) -> DeciderDecision: ...


# ------------------------------------------------------------------ state text
def build_state(
    case: WasteCase,
    assessment: SafetyAssessment,
    history: list[tuple[str, int]],
) -> str:
    """Plain-text case description given to the Decider. Contains only known facts."""
    lines = ["WASTE CASE (user-provided, unverified)"]
    lines.append(f"Item: {case.waste_name or '(not given)'}")
    if case.description:
        lines.append(f"Description: {case.description}")
    lines.append(f"Category stated by user: {case.category or '(not given)'}")
    lines.append(f"Location: {case.location or '(not given)'}")
    lines.append(f"Danger or contamination information: {case.danger_info or '(none given)'}")
    lines.append(f"User objective: {case.objective}")
    if case.photo_attached:
        lines.append("Photo: attached but NOT analyzed by any AI model.")
    lines.append("")
    lines.append("DETERMINISTIC SAFETY ASSESSMENT")
    lines.append(f"Level: {assessment.level.value}")
    lines.append(f"Possible immediate danger: {'yes' if assessment.emergency else 'no'}")
    for r in assessment.reasons:
        lines.append(f"- {r}")
    lines.append(f"Missing information items: {len(assessment.missing_info)}")
    lines.append("")
    lines.append("INVESTIGATION SO FAR")
    if not history:
        lines.append("Nothing has been investigated yet.")
    for action, n_findings in history:
        lines.append(f"- {action}: {n_findings} finding(s) recorded")
    return "\n".join(lines)


# ------------------------------------------------------------------ response parsing
_SELECTED_KEYS = ("choice", "selected", "label", "option", "answer", "top", "value", "prediction", "predicted")
_PROB_KEYS = ("probabilities", "probs", "distribution", "scores", "options")


def _match_option(value: Any, options: list[str]) -> Optional[str]:
    if isinstance(value, str):
        for o in options:
            if value.strip().lower() == o.lower():
                return o
    return None


def parse_choice_answer(answer: Any, options: list[str]) -> dict[str, Any]:
    """Extract (selected, confidence, probabilities) from one `choice` answer.

    Returns a dict with keys selected, confidence, probabilities, derived_from.
    Raises DeciderResponseError if the selected option cannot be determined.
    """
    selected: Optional[str] = None
    confidence: Optional[float] = None
    probs: Optional[dict[str, float]] = None
    derived = ""

    if isinstance(answer, str):
        selected = _match_option(answer, options)
    elif isinstance(answer, dict):
        # nested result object, e.g. {"type": "choice", "result": {...}}
        inner = answer.get("result") if isinstance(answer.get("result"), dict) else answer
        for key in _SELECTED_KEYS:
            if key in inner:
                selected = _match_option(inner[key], options)
                if selected:
                    break
        conf = inner.get("confidence")
        if isinstance(conf, (int, float)) and not isinstance(conf, bool):
            confidence = float(conf)
        for key in _PROB_KEYS:
            cand = inner.get(key)
            if isinstance(cand, dict) and cand:
                parsed = {}
                for k, v in cand.items():
                    opt = _match_option(k, options)
                    if opt is not None and isinstance(v, (int, float)) and not isinstance(v, bool):
                        parsed[opt] = float(v)
                if parsed:
                    probs = parsed
                    break
        if selected is None and probs:
            selected = max(probs, key=probs.get)
            derived = "argmax of the probabilities returned by the server (no explicit selected field found)"

    if selected is None:
        raise DeciderResponseError("Could not determine the selected option from the Decider response.", raw=answer)
    return {"selected": selected, "confidence": confidence, "probabilities": probs, "derived_from": derived}


# ------------------------------------------------------------------ real backend
class HttpDeciderBackend:
    """Talks to a local `strands-decider serve` process."""

    name = "Strands Decider 2B (live, local server)"
    live = True

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout_s: float = 60.0,
                 model_field: str = "", api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.model_field = model_field
        self.api_key = api_key.strip()
        # The hosted endpoint previously tested by this project uses /v1/decide and
        # bearer authentication. Local strands-decider uses /v1/systemone without auth.
        self.hosted = self.base_url.startswith("https://") or bool(self.api_key)
        self.endpoint = "/v1/decide" if self.hosted else "/v1/systemone"
        self.question_name = HOSTED_QUESTION_NAME if self.hosted else LOCAL_QUESTION_NAME

    # -- helpers
    def health(self) -> dict[str, Any]:
        try:
            headers = {"authorization": f"Bearer {self.api_key}"} if self.hosted and self.api_key else {}
            req = urllib.request.Request(self.base_url + "/health", headers=headers)
            with urllib.request.urlopen(req, timeout=min(self.timeout_s, 10)) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise DeciderUnavailable(
                f"Cannot reach the Decider server at {self.base_url} ({exc}). "
                "Start it with: strands-decider serve StrandsAgents/strands-decider-2B-hobson-v21 --port 8000"
            ) from exc

    def request_body(self, state: str, options: list[str]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "state": state,
            "questions": {
                self.question_name: {
                    "type": "choice",
                    "instructions": QUESTION_TEXT,
                    "criteria": {o: ACTION_CRITERIA[o] for o in options},
                }
            },
        }
        if self.model_field:
            body["model"] = self.model_field
        return body

    def choose(self, state: str, options: list[str]) -> DeciderDecision:
        body = json.dumps(self.request_body(state, options)).encode("utf-8")
        headers = {"content-type": "application/json"}
        if self.hosted and self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.base_url + self.endpoint, data=body, method="POST", headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:  # pragma: no cover - best effort
                pass
            raise DeciderResponseError(f"Decider server returned HTTP {exc.code}. {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DeciderUnavailable(
                f"Cannot reach the Decider server at {self.base_url} ({exc}). Is `strands-decider serve` running?"
            ) from exc
        except ValueError as exc:
            raise DeciderResponseError("Decider server returned invalid JSON.") from exc

        answers = raw.get("answers") if isinstance(raw, dict) else None
        if not isinstance(answers, dict):
            raise DeciderResponseError("Decider response has no answers object.", raw=raw)
        answer = answers.get(self.question_name)
        # Accept either known question key to tolerate provider-side aliases, but
        # never infer a choice from missing/invalid answer content.
        if answer is None:
            answer = answers.get(LOCAL_QUESTION_NAME) or answers.get(HOSTED_QUESTION_NAME)
        if answer is None:
            raise DeciderResponseError("Decider response has no answer for the question.", raw=raw)
        parsed = parse_choice_answer(answer, options)
        latency = raw.get("latency_ms")
        model_name = raw.get("model") if isinstance(raw.get("model"), str) else ""
        backend_name = (f"Strands Decider (live, hosted: {model_name})" if self.hosted and model_name
                        else "Strands Decider (live, hosted endpoint)" if self.hosted
                        else f"Strands Decider (live, local: {model_name})" if model_name
                        else self.name)
        return DeciderDecision(
            backend=backend_name,
            live=True,
            question=QUESTION_TEXT,
            options=list(options),
            selected=parsed["selected"],
            confidence=parsed["confidence"],
            probabilities=parsed["probabilities"],
            latency_ms=float(latency) if isinstance(latency, (int, float)) and not isinstance(latency, bool) else None,
            selected_derived_from=parsed["derived_from"],
            raw_response=raw,
        )


# ------------------------------------------------------------------ simulated backend
class SimulatedDeciderBackend:
    """Deterministic stand-in used ONLY for demo mode.

    It is a plain rule list. It is NOT Strands Decider 2B, produces no
    confidence or probabilities, and every decision is flagged live=False.
    The policy is intentionally a little naive (it follows the user's stated
    objective first) so the demo can show the safety guardrail rejecting it.
    """

    name = "SIMULATED rules (not Strands Decider 2B)"
    live = False

    def __init__(self, case: WasteCase, assessment: SafetyAssessment):
        self.case = case
        self.assessment = assessment

    def choose(self, state: str, options: list[str]) -> DeciderDecision:
        order: list[str]
        obj = self.case.objective
        if obj == "recover":
            order = ["investigate_recovery", "check_disposal", "check_evidence", "request_info", "conclude"]
        elif obj == "dispose":
            order = ["check_disposal", "check_evidence", "investigate_recovery", "request_info", "conclude"]
        else:
            order = ["investigate_recovery", "check_disposal", "check_evidence", "request_info", "conclude"]
        if self.assessment.emergency:
            order = ["escalate"] + order
        elif self.assessment.level == Level.HIGH_RISK and obj != "recover":
            order = ["escalate"] + order
        pick = next((o for o in order if o in options), options[0])
        return DeciderDecision(
            backend=self.name,
            live=False,
            question=QUESTION_TEXT,
            options=list(options),
            selected=pick,
            rationale="Simulated fixed-order rule list based on the user's objective. No model was run.",
        )
