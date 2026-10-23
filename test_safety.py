import unittest

from tests._helpers import scenario_case  # noqa: F401  (sets sys.path)
from agent import safety
from agent.schemas import Action, Level, WasteCase
from agent.workflow import run_case
from tools.material_lookup import load_materials, load_scenarios


class TestAssessment(unittest.TestCase):
    def test_unknown_chemical_is_high_risk(self):
        a = safety.assess(scenario_case("unknown_chemical"))
        self.assertEqual(a.level, Level.HIGH_RISK)
        self.assertFalse(a.emergency)

    def test_unmatched_item_is_treated_cautiously(self):
        a = safety.assess(WasteCase(waste_name="strange glowing rock", description="found it"))
        self.assertEqual(a.level, Level.HIGH_RISK)

    def test_emergency_wording_detected(self):
        a = safety.assess(scenario_case("battery_emergency"))
        self.assertTrue(a.emergency)
        self.assertEqual(a.level, Level.HIGH_RISK)

    def test_negated_danger_words_do_not_trigger(self):
        a = safety.assess(scenario_case("ewaste"))
        self.assertFalse(a.emergency)
        self.assertEqual(a.level, Level.REGULATED)
        b = safety.assess(scenario_case("used_oil"))
        self.assertEqual(b.level, Level.REGULATED)

    def test_agri_is_routine(self):
        self.assertEqual(safety.assess(scenario_case("agri")).level, Level.ROUTINE)


class TestValidation(unittest.TestCase):
    def test_empty_input_rejected(self):
        self.assertTrue(safety.validate_case(WasteCase()))
        r = run_case(WasteCase())
        self.assertEqual(r.status, "invalid_input")
        self.assertTrue(r.errors)
        self.assertEqual(r.steps, [])

    def test_bad_objective_rejected(self):
        self.assertTrue(safety.validate_case(WasteCase(waste_name="straw", objective="burn")))

    def test_too_long_rejected(self):
        self.assertTrue(safety.validate_case(WasteCase(waste_name="straw", description="x" * 5000)))

    def test_valid_input_ok(self):
        self.assertEqual(safety.validate_case(WasteCase(waste_name="rice straw")), [])


class TestGuardrail(unittest.TestCase):
    def setUp(self):
        self.high = safety.assess(scenario_case("unknown_chemical"))
        self.emerg = safety.assess(scenario_case("battery_emergency"))
        self.routine = safety.assess(scenario_case("agri"))

    def test_blocks_recovery_for_unknown(self):
        v = safety.review_action(Action.INVESTIGATE_RECOVERY.value, self.high, [])
        self.assertFalse(v.allowed)
        self.assertEqual(v.final, Action.ESCALATE.value)
        self.assertEqual(v.rule_id, "S2_NO_RECOVERY_FOR_HIGH_RISK")

    def test_emergency_forces_escalation_first(self):
        v = safety.review_action(Action.CHECK_EVIDENCE.value, self.emerg, [])
        self.assertEqual(v.final, Action.ESCALATE.value)
        self.assertEqual(v.rule_id, "S1_EMERGENCY_FIRST")

    def test_cannot_conclude_high_risk_without_escalation(self):
        v = safety.review_action(Action.CONCLUDE.value, self.high, [Action.CHECK_EVIDENCE.value])
        self.assertEqual(v.final, Action.ESCALATE.value)

    def test_no_repeat(self):
        v = safety.review_action(Action.CHECK_EVIDENCE.value, self.routine, [Action.CHECK_EVIDENCE.value])
        self.assertFalse(v.allowed)

    def test_allows_normal_action(self):
        v = safety.review_action(Action.INVESTIGATE_RECOVERY.value, self.routine, [])
        self.assertTrue(v.allowed)


class TestUnsafeScanner(unittest.TestCase):
    def test_flags_diy_instructions(self):
        for bad in ["Burn the casing to get the copper out.",
                    "You can neutralise the acid with baking soda.",
                    "Dismantle the battery at home.",
                    "Mix the solvent with the oil and pour it down the drain."]:
            self.assertTrue(safety.scan_unsafe_instructions(bad), bad)

    def test_ignores_negated_warnings(self):
        self.assertEqual(safety.scan_unsafe_instructions(
            "Do not open, puncture, burn or throw in general waste. Never pour on ground or into drains."), [])

    def test_no_dataset_text_gives_diy_advice(self):
        """Every string in the dataset and every tool output must pass the scanner."""
        texts = []

        def walk(x):
            if isinstance(x, str):
                texts.append(x)
            elif isinstance(x, dict):
                [walk(v) for v in x.values()]
            elif isinstance(x, list):
                [walk(v) for v in x]
        walk(load_materials())
        for t in texts:
            self.assertEqual(safety.scan_unsafe_instructions(t), [], t)
        for s in load_scenarios():
            r = run_case(WasteCase(**s["case"]))
            for st in r.steps:
                for f in st.findings:
                    self.assertEqual(safety.scan_unsafe_instructions(f.statement + " " + f.next_step), [], f.statement)
            for n in r.recommendation.next_steps:
                self.assertEqual(safety.scan_unsafe_instructions(n), [], n)


if __name__ == "__main__":
    unittest.main()
