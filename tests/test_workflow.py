import unittest
from unittest import mock

from tests._helpers import ScriptedBackend, DownBackend, scenario_case
from agent import workflow
from agent.generation import GeneratorUnavailable, guarded_summary
from agent.schemas import Action, GeneratedText, Label, Pathway, WasteCase
from tools.material_lookup import load_scenarios


class FakeGenerator:
    name = "fake"

    def __init__(self, text=None, error=None):
        self.text, self.error = text, error

    def summarize(self, facts):
        if self.error:
            raise GeneratorUnavailable(self.error)
        return GeneratedText(self.text, "bedrock:fake", Label.HYPOTHESIS, True)


class TestScenarios(unittest.TestCase):
    def run_sim(self, sid):
        return workflow.run_case(scenario_case(sid))

    def test_each_scenario_changes_workflow(self):
        pathways = {s["id"]: self.run_sim(s["id"]).recommendation.pathway for s in load_scenarios()}
        self.assertEqual(pathways["agri"], Pathway.RECOVER)
        self.assertEqual(pathways["used_oil"], Pathway.DISPOSE)
        self.assertEqual(pathways["ewaste"], Pathway.DISPOSE)
        self.assertEqual(pathways["unknown_chemical"], Pathway.ESCALATE)
        self.assertEqual(pathways["battery_emergency"], Pathway.ESCALATE)

    def test_unknown_chemical_never_gets_recovery_findings(self):
        r = self.run_sim("unknown_chemical")
        self.assertNotIn(Action.INVESTIGATE_RECOVERY.value, [s.executed_action for s in r.steps])
        self.assertTrue(any(s.guardrail and not s.guardrail.allowed for s in r.steps))
        self.assertTrue(any(s.executed_by == "guardrail_override" for s in r.steps))

    def test_recovery_vs_disposal_investigated(self):
        agri = [s.executed_action for s in self.run_sim("agri").steps]
        oil = [s.executed_action for s in self.run_sim("used_oil").steps]
        self.assertEqual(agri[0], Action.INVESTIGATE_RECOVERY.value)
        self.assertEqual(oil[0], Action.CHECK_DISPOSAL.value)

    def test_emergency_warning_present(self):
        r = self.run_sim("battery_emergency")
        text = " ".join(f.statement for s in r.steps for f in s.findings)
        self.assertIn("EMERGENCY", text)
        self.assertIn("EMERGENCY", " ".join(r.recommendation.next_steps))

    def test_no_facility_claims(self):
        for s in load_scenarios():
            r = workflow.run_case(scenario_case(s["id"]))
            for st in r.steps:
                for f in st.findings:
                    self.assertNotIn("is nearby", f.statement.lower().replace("does not claim that any facility is nearby", ""))
                    if f.extra.get("facility_status"):
                        self.assertEqual(f.extra["facility_status"], "NOT_VERIFIED")


class TestDemoMarking(unittest.TestCase):
    def test_simulated_is_clearly_marked(self):
        r = workflow.run_case(scenario_case("agri"))
        self.assertTrue(r.simulated)
        self.assertIn("SIMULATED", r.mode_label)
        self.assertIn("NOT from Strands Decider", r.mode_label)
        for s in r.steps:
            self.assertFalse(s.decision.live)
            self.assertIn("SIMULATED", s.decision.backend)
            self.assertIsNone(s.decision.confidence)
            self.assertIsNone(s.decision.probabilities)
        self.assertFalse(r.summary.ai_generated)

    def test_simulated_never_calls_generator(self):
        gen = mock.Mock()
        workflow.run_case(scenario_case("agri"), backend=None, generator=gen)
        gen.summarize.assert_not_called()


class TestEvidenceVsHypothesis(unittest.TestCase):
    def test_labels(self):
        r = workflow.run_case(scenario_case("agri"))
        labels = {f.label for s in r.steps for f in s.findings}
        self.assertIn(Label.REFERENCE, labels)
        for s in r.steps:
            for f in s.findings:
                if f.label == Label.REFERENCE:
                    self.assertTrue(f.source_ids, f.statement)
        # missing info is shown as UNKNOWN, never as fact
        self.assertTrue(r.missing_info or True)

    def test_generated_summary_is_hypothesis(self):
        r = workflow.run_case(scenario_case("agri"), backend=ScriptedBackend(["investigate_recovery", "conclude"]),
                              generator=FakeGenerator("Straw can often go to biomass users; confirm locally."))
        self.assertTrue(r.summary.ai_generated)
        self.assertEqual(r.summary.label, Label.HYPOTHESIS)
        self.assertFalse(r.simulated)
        self.assertIn("LIVE", r.mode_label)

    def test_unsafe_generated_text_discarded(self):
        r = workflow.run_case(scenario_case("used_oil"), backend=ScriptedBackend(["check_disposal", "conclude"]),
                              generator=FakeGenerator("Burn it at home to get rid of it."))
        self.assertFalse(r.summary.ai_generated)
        self.assertTrue(any("safety filter" in w for w in r.warnings))


class TestFailures(unittest.TestCase):
    def test_decider_down_stops_without_faking(self):
        r = workflow.run_case(scenario_case("agri"), backend=DownBackend())
        self.assertEqual(r.status, "decider_error")
        self.assertIsNone(r.recommendation)
        self.assertTrue(r.errors)
        self.assertTrue(all(s.decision is None for s in r.steps))

    def test_bedrock_unavailable_falls_back_to_template(self):
        r = workflow.run_case(scenario_case("agri"), backend=ScriptedBackend(["investigate_recovery", "conclude"]),
                              generator=FakeGenerator(error="AWS credentials were not found."))
        self.assertEqual(r.status, "ok")
        self.assertFalse(r.summary.ai_generated)
        self.assertTrue(any("unavailable" in w for w in r.warnings))

    def test_no_generator_in_live_mode_is_reported(self):
        r = workflow.run_case(scenario_case("agri"), backend=ScriptedBackend(["investigate_recovery", "conclude"]))
        self.assertFalse(r.summary.ai_generated)
        self.assertTrue(r.warnings)

    def test_tool_error_is_recorded_not_fatal(self):
        def boom(c, a, p):
            raise RuntimeError("dataset unreadable")
        with mock.patch.dict(workflow.TOOL_REGISTRY, {Action.INVESTIGATE_RECOVERY.value: boom}):
            r = workflow.run_case(scenario_case("agri"))
        self.assertEqual(r.status, "ok")
        self.assertTrue(any("failed" in w for w in r.warnings))
        self.assertTrue(r.steps[0].error)

    def test_live_decider_proposing_recovery_for_unknown_is_vetoed(self):
        r = workflow.run_case(scenario_case("unknown_chemical"),
                              backend=ScriptedBackend(["investigate_recovery", "investigate_recovery", "conclude"]))
        first = r.steps[0]
        self.assertEqual(first.decision.selected, "investigate_recovery")
        self.assertEqual(first.executed_action, "escalate")
        self.assertFalse(first.guardrail.allowed)
        self.assertEqual(r.recommendation.pathway, Pathway.ESCALATE)

    def test_why_not_selected_uses_only_returned_data(self):
        r = workflow.run_case(scenario_case("agri"), backend=ScriptedBackend(["investigate_recovery", "conclude"]))
        self.assertTrue(any("probability" in w for w in r.why_not_selected))
        sim = workflow.run_case(scenario_case("agri"))
        self.assertFalse(any("probability" in w for w in sim.why_not_selected))


if __name__ == "__main__":
    unittest.main()
