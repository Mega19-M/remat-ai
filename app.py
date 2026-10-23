"""RE:MAT AI - Streamlit dashboard.

Run with: streamlit run app.py
"""
from __future__ import annotations

import html
import json
import os

import streamlit as st

import config
from agent.decider import HttpDeciderBackend
from agent.generation import BedrockGenerator
from agent.schemas import OBJECTIVES, CaseResult, Label, WasteCase
from agent.workflow import run_case
from tools.material_lookup import load_scenarios, load_sources


# ------------------------------------------------------------------ settings
config.load_dotenv_if_present()

# Copy Streamlit Cloud Secrets into environment variables before loading settings.
_SECRET_ENV_NAMES = (
    "REMAT_DECIDER_URL",
    "DECIDER_API_KEY",
    "REMAT_DECIDER_MODEL_FIELD",
    "REMAT_DECIDER_TIMEOUT_S",
    "REMAT_BEDROCK_MODEL_ID",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "REMAT_MAX_STEPS",
)

for _name in _SECRET_ENV_NAMES:
    try:
        if _name in st.secrets and _name not in os.environ:
            os.environ[_name] = str(st.secrets[_name])
    except Exception:
        # Demo mode should work even when secrets are not configured.
        pass

SETTINGS = config.Settings.from_env()

st.set_page_config(
    page_title="RE:MAT AI",
    page_icon="♻️",
    layout="wide",
)

PAGES = [
    "Overview",
    "Waste assessment",
    "Recommendations",
    "Decision trace",
    "Demo scenarios",
    "About & methodology",
]

CATEGORIES = [
    "Not sure",
    "Agricultural residue",
    "Electronic waste",
    "Batteries",
    "Used oil / industrial waste",
    "Chemical waste",
    "Other",
]

MODE_DEMO = "Demo (simulated, no AWS needed)"
MODE_LIVE = "Live (Strands Decider; optional Bedrock summary)"

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
.remat-hero {background: linear-gradient(120deg,#0b3d2e 0%,#14684a 60%,#1c8a5f 100%); color:#f3fbf6;
  padding: 1.6rem 1.8rem; border-radius: 16px; margin-bottom: 1rem;}
.remat-hero h1 {margin:0; font-size:2.1rem; color:#fff;}
.remat-hero p {margin:.4rem 0 0 0; font-size:1.05rem; opacity:.95;}
.remat-card {border:1px solid rgba(120,140,130,.35); border-radius:12px; padding:.9rem 1.1rem; margin:.5rem 0;}
.badge {display:inline-block; padding:.1rem .55rem; border-radius:999px; font-size:.75rem; font-weight:600; margin-right:.4rem;}
.b-ref {background:#d8f3e3; color:#0b5a37;}
.b-hyp {background:#fff0c9; color:#7a5200;}
.b-unk {background:#e6e8ee; color:#394150;}
.b-saf {background:#fde0e0; color:#8a1c1c;}
.b-sim {background:#ffe3b8; color:#7a3d00;}
.b-live {background:#cfeedd; color:#0b5a37;}
.banner-sim {background:#fff1d6; border-left:6px solid #e08a00; padding:.7rem 1rem; border-radius:8px; color:#5a3300; margin-bottom:.8rem;}
.banner-live {background:#e3f6ec; border-left:6px solid #1c8a5f; padding:.7rem 1rem; border-radius:8px; color:#0b4a30; margin-bottom:.8rem;}
.emergency {background:#fde0e0; border-left:6px solid #c62828; padding:.8rem 1rem; border-radius:8px; color:#6d1212; font-weight:600;}
.small {font-size:.85rem; opacity:.85;}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

LABEL_CLASS = {
    Label.REFERENCE: "b-ref",
    Label.HYPOTHESIS: "b-hyp",
    Label.UNKNOWN: "b-unk",
    Label.SAFETY: "b-saf",
}


def esc(x) -> str:
    return html.escape(str(x))


def badge(text: str, cls: str) -> str:
    return f'<span class="badge {cls}">{esc(text)}</span>'


# ------------------------------------------------------------------ state
def init_state() -> None:
    defaults = {
        "page": PAGES[0],
        "f_name": "",
        "f_desc": "",
        "f_cat": CATEGORIES[0],
        "f_loc": "",
        "f_danger": "",
        "f_obj": "both",
        "f_photo_ok": False,
        "result": None,
        "scenario_id": None,
    }

    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def load_scenario(sid: str) -> None:
    """Load a synthetic scenario into the form."""
    sc = next(s for s in load_scenarios() if s["id"] == sid)
    c = sc["case"]

    st.session_state.update({
        "f_name": c["waste_name"],
        "f_desc": c["description"],
        "f_cat": c["category"] if c["category"] in CATEGORIES else CATEGORIES[0],
        "f_loc": c["location"],
        "f_danger": c["danger_info"],
        "f_obj": c["objective"],
        "f_photo_ok": False,
        "scenario_id": sid,
        "result": None,
        "page": "Waste assessment",
    })


def go(page: str) -> None:
    st.session_state["page"] = page


init_state()


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## ♻️ RE:MAT AI")
    st.caption("The Safe Recovery & Disposal Intelligence Agent")

    st.radio("Navigate", PAGES, key="page")
    st.divider()

    mode = st.radio(
        "Decision mode",
        [MODE_DEMO, MODE_LIVE],
        key="mode",
    )

    if mode == MODE_DEMO:
        st.markdown(
            '<div class="banner-sim"><b>SIMULATED.</b> A fixed rule list makes '
            "the decisions. Strands Decider 2B and Amazon Bedrock are NOT used.</div>",
            unsafe_allow_html=True,
        )

    else:
        with st.expander("Live settings", expanded=True):
            st.text_input(
                "Decider server URL",
                value=SETTINGS.decider_url,
                key="decider_url",
            )

            st.text_input(
                "Bedrock model id",
                value=SETTINGS.bedrock_model_id,
                key="bedrock_model",
                help="A model id you have access to.",
            )

            st.text_input(
                "AWS region",
                value=SETTINGS.aws_region,
                key="region",
            )

            # Do not assume that a hosted endpoint exposes a public GET /health.
            # The actual authenticated POST /v1/decide request is tested when
            # the user runs an assessment.
            if st.button("Check Decider server"):
                st.info(
                    "This button does not call /health because the hosted "
                    "service may not expose that endpoint. Run an assessment "
                    "to test the actual authenticated decision request."
                )

        if st.session_state.get(
            "decider_url", SETTINGS.decider_url
        ).startswith("https://"):
            st.caption(
                "Using the hosted HTTPS Decider endpoint. "
                "The API key is loaded from app settings and is not displayed."
            )
        else:
            st.caption(
                "Using a local Decider server. Start `strands-decider serve` "
                "in a separate terminal. AWS credentials for an optional "
                "Bedrock summary come from your AWS configuration."
            )


# ------------------------------------------------------------------ case input
def current_case() -> WasteCase:
    photo = st.session_state.get("f_photo")
    cat = st.session_state["f_cat"]

    return WasteCase(
        waste_name=st.session_state["f_name"].strip(),
        description=st.session_state["f_desc"].strip(),
        category="" if cat == CATEGORIES[0] else cat,
        location=st.session_state["f_loc"].strip(),
        danger_info=st.session_state["f_danger"].strip(),
        objective=st.session_state["f_obj"],
        photo_attached=photo is not None,
        photo_confirmed_by_user=(
            bool(st.session_state["f_photo_ok"]) and photo is not None
        ),
        synthetic_scenario_id=st.session_state.get("scenario_id"),
    )


def mode_banner(result: CaseResult) -> None:
    cls = "banner-sim" if result.simulated else "banner-live"

    st.markdown(
        f'<div class="{cls}"><b>{esc(result.mode_label)}</b></div>',
        unsafe_allow_html=True,
    )

    if result.case.synthetic_scenario_id:
        st.markdown(
            badge("SYNTHETIC SCENARIO - not a real incident or facility", "b-sim"),
            unsafe_allow_html=True,
        )


def render_finding(f) -> None:
    src = ""

    if f.source_ids:
        names = load_sources()
        src = "<div class='small'>Sources: " + "; ".join(
            esc(f"{sid} ({names[sid]['title']})")
            if sid in names else esc(sid)
            for sid in f.source_ids
        ) + "</div>"

    unc = (
        "<div class='small'><b>Uncertainties:</b> "
        + " ".join(esc(u) for u in f.uncertainties)
        + "</div>"
    ) if f.uncertainties else ""

    nxt = (
        f"<div class='small'><b>Next step:</b> {esc(f.next_step)}</div>"
    ) if f.next_step else ""

    extra = ""

    if f.extra.get("material_compatibility"):
        extra += (
            "<div class='small'><b>Material compatibility:</b> "
            f"{esc(f.extra['material_compatibility'])}</div>"
        )

    if f.extra.get("professional_processing"):
        extra += (
            "<div class='small'><b>Professional processing needed:</b> "
            f"{esc(f.extra['professional_processing'])}</div>"
        )

    cls = "emergency" if f.statement.startswith("EMERGENCY") else "remat-card"

    st.markdown(
        f"<div class='{cls}'>"
        f"{badge(f.label.value, LABEL_CLASS[f.label])}"
        f"{esc(f.statement)}{extra}{unc}{nxt}{src}</div>",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------ pages
def page_overview() -> None:
    st.markdown(
        '<div class="remat-hero"><h1>RE:MAT AI</h1>'
        "<p><i>Every waste item deserves the right next decision — "
        "not just another report.</i></p></div>",
        unsafe_allow_html=True,
    )

    st.write(
        "RE:MAT AI investigates what to do with a waste item. A small decision "
        "model (Strands Decider 2B) chooses which investigation tool to run next, "
        "a deterministic safety layer can overrule it, and every finding is "
        "labelled as source-backed, hypothesis or unknown."
    )

    cols = st.columns(4)

    for col, (heading, description) in zip(
        cols,
        [
            ("RECOVER", "Investigate plausible reuse or recycling routes."),
            ("DISPOSE", "Point to authorized collection or disposal channels."),
            ("ESCALATE", "Refer unknown, hazardous or dangerous material to qualified people."),
            ("VERIFY", "Check the evidence and list what is missing."),
        ],
    ):
        with col:
            st.markdown(
                f"<div class='remat-card'><b>{heading}</b><br>"
                f"<span class='small'>{description}</span></div>",
                unsafe_allow_html=True,
            )

    st.markdown("#### What it will not do")
    st.markdown(
        "- Invent facilities or claim that any facility is nearby, licensed or accepting your waste.\n"
        "- Give do-it-yourself handling, dismantling, mixing, burning or neutralising instructions for hazardous material.\n"
        "- Pretend that a photo was analyzed by AI (photo analysis is not implemented).\n"
        "- Present simulated demo decisions as live Decider 2B or Bedrock results."
    )

    st.button(
        "Start an assessment",
        on_click=go,
        args=("Waste assessment",),
        type="primary",
    )
    st.button(
        "Try a demo scenario",
        on_click=go,
        args=("Demo scenarios",),
    )


def page_assess() -> None:
    st.header("Waste assessment")

    if st.session_state.get("scenario_id"):
        st.markdown(
            badge("SYNTHETIC SCENARIO LOADED - invented example data", "b-sim"),
            unsafe_allow_html=True,
        )

    left, right = st.columns([3, 2])

    with left:
        st.text_input(
            "Waste name",
            key="f_name",
            placeholder="e.g. Rice straw, old laptops, unlabelled drum",
        )
        st.text_area("Description (what you can observe)", key="f_desc", height=110)

        c1, c2 = st.columns(2)
        c1.selectbox("Waste category (if known)", CATEGORIES, key="f_cat")
        c2.text_input("Approximate location (town or district)", key="f_loc")

        st.text_area(
            "Known contamination or danger information",
            key="f_danger",
            height=80,
            placeholder="Leaks, labels, smells, mixing, damage... or 'none known'",
        )

        st.radio(
            "Your objective",
            list(OBJECTIVES),
            key="f_obj",
            format_func=lambda k: OBJECTIVES[k],
            horizontal=True,
        )

    with right:
        photo = st.file_uploader(
            "Optional photo",
            type=["png", "jpg", "jpeg"],
            key="f_photo",
        )

        if photo is not None:
            st.image(photo, use_container_width=True)
            st.warning(
                "Prototype notice: photo analysis is NOT implemented. "
                "This photo is not analyzed by any AI. It is only noted in the case."
            )
            st.checkbox(
                "I confirm this photo shows the item I described",
                key="f_photo_ok",
            )

        st.info(
            "Never open, move or test unknown containers to answer these questions. "
            "Describe only what you can see from a safe distance."
        )

    if st.button("Run assessment", type="primary"):
        case = current_case()

        with st.spinner("Investigating..."):
            if mode == MODE_DEMO:
                result = run_case(
                    case,
                    backend=None,
                    max_steps=SETTINGS.max_steps,
                )
            else:
                backend = HttpDeciderBackend(
                    st.session_state["decider_url"],
                    SETTINGS.decider_timeout_s,
                    SETTINGS.decider_model_field,
                    SETTINGS.decider_api_key,
                )

                gen = (
                    BedrockGenerator(
                        st.session_state["bedrock_model"],
                        st.session_state["region"],
                    )
                    if st.session_state["bedrock_model"]
                    else None
                )

                result = run_case(
                    case,
                    backend=backend,
                    generator=gen,
                    max_steps=SETTINGS.max_steps,
                )

        st.session_state["result"] = result

        if result.status == "invalid_input":
            for error in result.errors:
                st.error(error)

        elif result.status == "decider_error":
            st.error(
                "The Decider could not be used, so no decision was made. "
                + " ".join(result.errors)
            )
            st.caption(
                "Nothing was substituted: switch to Demo mode in the sidebar "
                "if you want to explore the app with simulated decisions."
            )

        else:
            st.success("Assessment complete.")
            st.button(
                "View recommendations",
                on_click=go,
                args=("Recommendations",),
            )
            st.button(
                "View decision trace",
                on_click=go,
                args=("Decision trace",),
            )


def need_result() -> CaseResult | None:
    result = st.session_state.get("result")

    if result is None or result.status != "ok":
        st.info(
            "No completed assessment yet. Run one on the Waste assessment page, "
            "or load a demo scenario."
        )
        st.button(
            "Go to Waste assessment",
            on_click=go,
            args=("Waste assessment",),
        )
        return None

    return result


def page_recs() -> None:
    st.header("Recovery & disposal recommendations")
    result = need_result()

    if not result:
        return

    mode_banner(result)
    rec = result.recommendation

    if rec.pathway.value == "ESCALATE":
        st.markdown(
            f"<div class='emergency'>{esc(rec.headline)}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.subheader(rec.headline)

    st.markdown("**Next steps**")
    for step in rec.next_steps:
        st.markdown(f"- {step}")

    st.markdown("**Findings by investigation step**")
    for step in result.steps:
        if step.findings:
            st.markdown(
                f"##### {step.executed_action.replace('_', ' ').title()}"
            )
            for finding in step.findings:
                render_finding(finding)

    if result.missing_info:
        st.markdown("**Unknowns / missing information**")
        for item in result.missing_info:
            st.markdown(
                f"- {badge('UNKNOWN', 'b-unk')} {esc(item)}",
                unsafe_allow_html=True,
            )

    st.markdown("**Summary**")

    if result.summary:
        tag = (
            "AI-GENERATED HYPOTHESIS"
            if result.summary.ai_generated
            else "NOT AI-GENERATED"
        )

        st.markdown(
            f"{badge(tag, 'b-hyp' if result.summary.ai_generated else 'b-unk')} "
            f"<span class='small'>source: {esc(result.summary.source)}</span>",
            unsafe_allow_html=True,
        )
        st.write(result.summary.text)

        if result.summary.note:
            st.caption(result.summary.note)

    for warning in result.warnings:
        st.warning(warning)


def page_trace() -> None:
    st.header("Decision trace & evidence")
    result = need_result()

    if not result:
        return

    mode_banner(result)

    st.markdown("**1. Waste information (as entered)**")
    st.json({
        k: v
        for k, v in result.case.__dict__.items()
        if v not in ("", None, False)
    })

    assessment = result.assessment

    st.markdown("**2. Safety assessment and uncertainty**")
    st.write(
        f"Level: `{assessment.level.value}` · Possible immediate danger: "
        f"`{'yes' if assessment.emergency else 'no'}`"
    )

    for reason in assessment.reasons:
        st.markdown(f"- {reason}")

    for item in assessment.missing_info:
        st.markdown(
            f"- {badge('UNKNOWN', 'b-unk')} {esc(item)}",
            unsafe_allow_html=True,
        )

    st.markdown("**3-5. Steps: options, selected action, executed tool, findings**")

    for step in result.steps:
        decision = step.decision
        guardrail = step.guardrail

        title = (
            f"Step {step.step}: "
            + (
                f"Decider selected `{decision.selected}`"
                if decision
                else "no decision"
            )
        )

        if guardrail and not guardrail.allowed:
            title += (
                f" → overridden by {guardrail.rule_id} "
                f"→ ran `{guardrail.final}`"
            )

        with st.expander(title, expanded=step.step == 1):
            if decision:
                kind = "b-live" if decision.live else "b-sim"

                st.markdown(
                    f"{badge('LIVE' if decision.live else 'SIMULATED', kind)} "
                    f"{esc(decision.backend)}",
                    unsafe_allow_html=True,
                )

                st.write(
                    "**Available options:** "
                    + ", ".join(f"`{option}`" for option in step.available_options)
                )
                st.write(
                    f"**Selected by decision backend:** `{decision.selected}`"
                )

                if decision.confidence is not None:
                    st.write(
                        f"**Confidence returned by the Decider:** "
                        f"{decision.confidence:.3f}"
                    )
                else:
                    st.caption("No confidence value was returned for this decision.")

                if decision.probabilities:
                    st.write("**Per-option probabilities returned by the Decider:**")
                    st.json(decision.probabilities)

                if decision.latency_ms is not None:
                    st.caption(
                        f"Decider latency reported by server: {decision.latency_ms} ms"
                    )

                if decision.selected_derived_from:
                    st.caption("Note: " + decision.selected_derived_from)

                if decision.rationale:
                    st.caption(decision.rationale)

            if guardrail:
                if guardrail.allowed:
                    st.success("Safety guardrail: proposal allowed.")
                else:
                    st.error(
                        f"Safety guardrail REJECTED `{guardrail.proposed}` "
                        f"(rule {guardrail.rule_id}): {guardrail.reason} "
                        f"Replaced by deterministic action `{guardrail.final}` "
                        "(this replacement is not a Decider decision)."
                    )

            st.write(
                f"**Executed:** `{step.executed_action}` "
                f"(chosen by: {step.executed_by})"
            )
            st.write("**Tool inputs:**")
            st.json(step.tool_inputs)

            if step.error:
                st.error(step.error)

            for finding in step.findings:
                render_finding(finding)

            with st.expander("State text sent to the decision backend"):
                st.code(step.state_sent)

            if decision and decision.live and decision.raw_response is not None:
                with st.expander("Raw Decider response"):
                    st.code(json.dumps(decision.raw_response, indent=2)[:6000])

    st.markdown("**6. Final recommendation**")
    st.write(result.recommendation.headline)

    st.markdown("**7. Why other actions were not selected**")
    if result.why_not_selected:
        for explanation in result.why_not_selected:
            st.markdown(f"- {explanation}")
    else:
        st.caption("No explanation is available from the decision data.")

    st.markdown("**8. Next steps**")
    for step in result.recommendation.next_steps:
        st.markdown(f"- {step}")

    st.download_button(
        "Download trace as JSON",
        data=json.dumps(result.to_dict(), indent=2, default=str),
        file_name="remat_trace.json",
        mime="application/json",
    )


def page_scenarios() -> None:
    st.header("Demo scenarios")

    st.markdown(
        badge("ALL SCENARIOS ARE SYNTHETIC", "b-sim")
        + " They are invented examples, not real incidents, locations or verified facilities.",
        unsafe_allow_html=True,
    )

    for scenario in load_scenarios():
        with st.container(border=True):
            st.markdown(f"#### {scenario['title']}")
            st.write(scenario["summary"])
            st.caption("Expected behaviour: " + scenario["expected_behavior"])

            st.button(
                "Load this scenario",
                key=f"load_{scenario['id']}",
                on_click=load_scenario,
                args=(scenario["id"],),
            )


def page_about() -> None:
    st.header("About & methodology")

    st.markdown("""
**Decision pattern: tool selection.** At each step the Strands Decider 2B model answers one typed *choice* question
("which investigation action should run next?") over a fixed option set. The chosen action decides which deterministic
tool runs next.

**Separation of duties.** The Decider proposes. A deterministic guardrail may reject or replace the proposal (for example,
recovery is never investigated for unknown material, and possible emergencies are escalated first). Replacements are shown
as safety overrides, never as Decider decisions.

**Evidence labels.** `SOURCE-BACKED` = from the small curated dataset with a source pointer. `HYPOTHESIS` = AI-generated or
suggested, not verified. `UNKNOWN` = missing information. `SAFETY RULE` = deterministic safety statement.

**Generative step.** After the investigation, an Amazon Bedrock model (through the Strands Agents SDK) writes a short summary
from the findings. It is labelled as an unverified hypothesis and discarded if the safety filter finds unsafe instructions.

**Limits.** The dataset is small and India-focused. Source entries are pointers to public rules; clause-level wording was
not verified by this project. No facility directory is included, so no facility is ever named. Photos are not analyzed.
Matching is keyword-based and can be wrong. This is decision support, not professional advice.
""")

    st.markdown("#### Source registry")

    for source_id, source in load_sources().items():
        st.markdown(
            f"- **{source_id}**: {source['title']} ({source['authority']}). "
            f"Date checked: {source.get('date_checked') or 'not checked'}. "
            f"Method: {source['check_method']}."
        )

    st.markdown(
        '<div class="emergency">In an emergency (leak, fumes, smoke, fire, swelling or exposure): '
        "move away if it is safe and contact local emergency services immediately (112 in India).</div>",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------ render page
{
    "Overview": page_overview,
    "Waste assessment": page_assess,
    "Recommendations": page_recs,
    "Decision trace": page_trace,
    "Demo scenarios": page_scenarios,
    "About & methodology": page_about,
}[st.session_state["page"]]()
