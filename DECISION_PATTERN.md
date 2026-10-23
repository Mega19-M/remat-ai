# Reusable decision pattern: tool selection with a veto

**Declared pattern:** `#tool-selection` (tag with `#strands-decider-2b`).

## The typed question
- Type: `choice` (Strands Decider question types are `noul`, `choice`, `score`).
- Question: *"Given the waste case and the investigation so far, which single investigation action should run next?"*
- Options (`agent/decider.py::ACTION_CRITERIA`): `investigate_recovery`, `check_disposal`, `check_evidence`, `escalate`, `request_info`, `conclude`, each with a one-line description sent as the option `criteria`.
- State: plain text built by `build_state` (case facts, deterministic safety level, what has already been investigated).

## The wiring
```
state, options ──► Decider.choose ──► proposed action ──► safety.review_action ──► final action ──► TOOL_REGISTRY[final]()
```
The Decider's answer decides which tool executes. The guardrail may veto; a veto is recorded as `guardrail_override`, never as a Decider decision.
This hackathon pattern is wired as an explicit loop in `agent/workflow.py`. The Strands blog describes an `InterventionHandler.before_tool_call`
integration for gating tool calls; this project did not use it because its exact API could not be verified here.

## How to reuse
1. Replace `ACTION_CRITERIA` and `TOOL_REGISTRY` with your own tools.
2. Write `build_state` for your domain.
3. Replace `safety.review_action` with your own policy (it must be deterministic and able to say no).
4. Keep `HttpDeciderBackend` as is.

## Known caveats (from the model card)
Questions are read less closely than documents; keep option descriptions distinct. Calibration is per primitive, so measure thresholds on your own data
before acting on confidence. Long multi-step documents are the weakest area; this project keeps the state short.
