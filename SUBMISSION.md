# Hackathon submission pack

Rules summary (from the Builder Center rules page, checked 2026-10-19): projects due **October 23, 2026, 11:59 PM PDT**; need an original agent using Decider 2B
with Amazon Bedrock, a Builder Center article, a public repo with at least one reusable decision pattern, a public demo video under 3 minutes, and a declared pattern.
Re-read the official rules before submitting; they may change.

## Checklists

**Required tags**
- [ ] `#strands-decider-2b` on the article
- [ ] `#tool-selection` on the article

**Before you submit (fill in honestly)**
- [ ] `python scripts/check_decider.py` ran against a real Decider server; raw response looked right (fix the parser if not)
- [ ] Live run completed with Bedrock summary; screenshot taken
- [ ] Demo video recorded in Live mode (or clearly says it is simulated), under 3 minutes, set to public
- [ ] Article published on Builder Center with the tags above
- [ ] Repo is public and contains the decision-pattern files (`agent/decider.py`, `docs/DECISION_PATTERN.md`)

**Public repository**
- [ ] No `.env`, AWS keys, tokens or `.aws` folder (search: `git grep -i -E "AKIA|secret|token"`)
- [ ] `LICENSE` present (Apache-2.0)
- [ ] README states what is tested and what is not
- [ ] No claims of endorsement, winning, measured impact or accuracy you have not measured

## Judging-focused explanation (rubric: creativity 30, Decider + Bedrock use 30, communication 25, originality 15)
- **Creativity:** an agent whose job is deciding the *next investigation step* for waste, including refusing to recover unknown material.
- **Actual Decider use:** the Decider's `choice` answer selects which tool executes next. Replacing it with a random or fixed choice changes the outcome (tested with a fake server).
  The simulated demo backend is labelled and never reported as Decider output.
- **Pattern quality:** typed option set + state builder + deterministic veto + trace, reusable via `docs/DECISION_PATTERN.md`.
- **Bedrock:** a Strands `Agent` with `BedrockModel` writes the labelled summary; unsafe output is discarded.
- **Bonus (only if you measure it):** record real Decider latency from the `latency_ms` field in the trace. Do not quote numbers you have not measured.
- **Originality:** the safety-veto-over-small-decision-model design applied to hazardous-waste triage.

## Builder Center article draft
**Title:** RE:MAT AI: letting a 2B decision model choose the next step in waste triage — with a safety veto
**Tags:** #strands-decider-2b #tool-selection

**The problem.** Waste apps usually collect reports. The harder question is what to do next: recover, dispose, escalate, or check the evidence. For unknown or hazardous
material the wrong "helpful" answer can be dangerous, and invented facility names are worse than no answer.

**The decision handed to Decider 2B.** At each step the Decider answers one typed choice question over six actions (investigate recovery, check disposal, check evidence,
escalate, request information, conclude). Its answer decides which deterministic tool runs next.

**Architecture.** Streamlit UI → deterministic safety assessment → Decider proposes → guardrail may reject or replace → tool runs → findings labelled
(source-backed / hypothesis / unknown / safety) → recommendation → Bedrock summary (via Strands Agents SDK), scanned for unsafe instructions. [Insert the Mermaid diagram from docs/ARCHITECTURE.md.]

**Why a veto.** A small decision model can be wrong. The guardrail encodes rules that must hold regardless: no recovery for unknown material, emergency wording escalates first,
no conclusion for high-risk cases without escalation. Overrides appear in the trace as overrides.

**Implementation notes.** The Decider runs as a local server (`strands-decider serve`) and the app calls `POST /v1/systemone`. Confidence and probabilities are stored only if the server returns them.
If the Decider is down, the run stops rather than silently simulating.

**Evaluation.** [Fill in only what you measured: e.g. results of running the five scenarios live, observed latency from `latency_ms`, how often the guardrail overrode the Decider.]
Automated tests cover the safety rules, workflow, failure handling and the client contract against a fake server.

**Limitations.** Small India-focused dataset; source entries are pointers not verified at clause level; no facility directory; photos not analyzed; keyword matching; no measured accuracy or impact.

**Lessons learned.** [Write your own. Suggested: how question wording affected the Decider; what the guardrail caught; what surprised you.]
