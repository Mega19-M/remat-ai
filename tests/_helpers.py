"""Shared test helpers: path setup and test doubles (clearly NOT the real Decider)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.decider import DeciderUnavailable  # noqa: E402
from agent.schemas import DeciderDecision, WasteCase  # noqa: E402
from tools.material_lookup import load_scenarios  # noqa: E402


class ScriptedBackend:
    """Test double: proposes a fixed sequence of actions. live=True so the workflow treats it
    like a live backend, but it is a TEST DOUBLE and not Strands Decider 2B."""
    name = "scripted test double"
    live = True

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def choose(self, state, options):
        pick = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return DeciderDecision(self.name, True, "q", list(options), pick,
                               confidence=0.5, probabilities={o: 1 / len(options) for o in options})


class DownBackend:
    name = "down"
    live = True

    def choose(self, state, options):
        raise DeciderUnavailable("server not running")


def scenario_case(sid):
    s = next(s for s in load_scenarios() if s["id"] == sid)
    return WasteCase(**s["case"], synthetic_scenario_id=sid)
