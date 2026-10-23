"""Decider client tests.

1. CONTRACT tests run against a tiny local fake HTTP server that follows the
   documented request schema. They prove our client builds the documented
   request and handles several response shapes. They do NOT prove anything about
   the real model.
2. LIVE tests talk to a real `strands-decider serve` process and a real Amazon
   Bedrock model. They are SKIPPED unless those are available.
"""
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from tests._helpers import scenario_case
from agent.decider import (
    DeciderResponseError, DeciderUnavailable, HttpDeciderBackend, parse_choice_answer,
)
from agent.workflow import run_case

OPTIONS = ["investigate_recovery", "check_disposal", "escalate"]


def make_server(response_for):
    seen = {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"model": "fake", "device": "cpu"}).encode())

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            seen["path"], seen["body"], seen["authorization"] = self.path, body, self.headers.get("authorization")
            payload = response_for(body)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


class TestContract(unittest.TestCase):
    def test_request_follows_documented_schema(self):
        srv, seen = make_server(lambda b: {"answers": {"next_action": {"type": "choice", "choice": "escalate",
                                                                       "confidence": 0.7,
                                                                       "probabilities": {"escalate": 0.8, "check_disposal": 0.2}}},
                                           "latency_ms": 12})
        self.addCleanup(srv.server_close); self.addCleanup(srv.shutdown)
        d = HttpDeciderBackend(f"http://127.0.0.1:{srv.server_port}").choose("state text", OPTIONS)
        self.assertEqual(seen["path"], "/v1/systemone")
        q = seen["body"]["questions"]["next_action"]
        self.assertEqual(seen["body"]["state"], "state text")
        self.assertEqual(q["type"], "choice")
        self.assertEqual(set(q["criteria"]), set(OPTIONS))
        self.assertEqual((d.selected, d.confidence, d.latency_ms, d.live), ("escalate", 0.7, 12.0, True))

    def test_hosted_endpoint_uses_bearer_auth_and_action_question(self):
        srv, seen = make_server(lambda b: {"model": "hosted-test-model", "answers": {"action": {
            "type": "choice", "choice": "conclude", "confidence": 0.88,
            "probabilities": {"conclude": 0.94, "escalate": 0.06}}}, "latency_ms": 25})
        self.addCleanup(srv.server_close); self.addCleanup(srv.shutdown)
        backend = HttpDeciderBackend(f"https://127.0.0.1:{srv.server_port}", api_key="test-secret")
        # Avoid TLS in the test while still exercising hosted URL mode.
        backend.base_url = f"http://127.0.0.1:{srv.server_port}"
        d = backend.choose("state", ["conclude", "escalate"])
        self.assertEqual(seen["path"], "/v1/decide")
        self.assertEqual(seen["authorization"], "Bearer test-secret")
        self.assertIn("action", seen["body"]["questions"])
        self.assertEqual(d.selected, "conclude")
        self.assertIn("hosted: hosted-test-model", d.backend)

    def test_selected_derived_from_probabilities_is_flagged(self):
        r = parse_choice_answer({"probabilities": {"escalate": 0.2, "check_disposal": 0.8}}, OPTIONS)
        self.assertEqual(r["selected"], "check_disposal")
        self.assertTrue(r["derived_from"])
        self.assertIsNone(r["confidence"])

    def test_unparseable_answer_raises_instead_of_guessing(self):
        with self.assertRaises(DeciderResponseError):
            parse_choice_answer({"weird": 1}, OPTIONS)
        with self.assertRaises(DeciderResponseError):
            parse_choice_answer({"choice": "not_an_option"}, OPTIONS)

    def test_missing_answer_key_raises(self):
        srv, _ = make_server(lambda b: {"answers": {}})
        self.addCleanup(srv.server_close); self.addCleanup(srv.shutdown)
        with self.assertRaises(DeciderResponseError):
            HttpDeciderBackend(f"http://127.0.0.1:{srv.server_port}").choose("s", OPTIONS)

    def test_server_not_running(self):
        with self.assertRaises(DeciderUnavailable):
            HttpDeciderBackend("http://127.0.0.1:9", timeout_s=2).choose("s", OPTIONS)
        with self.assertRaises(DeciderUnavailable):
            HttpDeciderBackend("http://127.0.0.1:9", timeout_s=2).health()

    def test_workflow_uses_server_choice(self):
        """The server's answer, not our code, decides which tool runs first."""
        srv, _ = make_server(lambda b: {"answers": {"next_action": {"choice": "check_disposal"}}})
        self.addCleanup(srv.server_close); self.addCleanup(srv.shutdown)
        r = run_case(scenario_case("agri"), backend=HttpDeciderBackend(f"http://127.0.0.1:{srv.server_port}"), max_steps=1)
        self.assertEqual(r.steps[0].executed_action, "check_disposal")
        self.assertEqual(r.steps[0].executed_by, "decider")


def _decider_url():
    return os.environ.get("REMAT_DECIDER_URL", "http://127.0.0.1:8000")


def _decider_up():
    try:
        HttpDeciderBackend(_decider_url(), timeout_s=3).health()
        return True
    except DeciderUnavailable:
        return False


@unittest.skipUnless(_decider_up(), "No live Strands Decider server reachable (start `strands-decider serve ...`).")
class TestLiveDecider(unittest.TestCase):
    def test_real_decider_picks_a_valid_option(self):
        r = run_case(scenario_case("unknown_chemical"), backend=HttpDeciderBackend(_decider_url()), max_steps=1)
        self.assertEqual(r.status, "ok", r.errors)
        d = r.steps[0].decision
        self.assertTrue(d.live)
        self.assertIn(d.selected, d.options)
        self.assertIsNotNone(d.raw_response)


@unittest.skipUnless(os.environ.get("REMAT_BEDROCK_MODEL_ID"), "REMAT_BEDROCK_MODEL_ID not set; skipping live Bedrock test.")
class TestLiveBedrock(unittest.TestCase):
    def test_real_bedrock_summary(self):
        from agent.generation import BedrockGenerator, GeneratorUnavailable
        from config import Settings
        s = Settings.from_env()
        try:
            out = BedrockGenerator(s.bedrock_model_id, s.aws_region).summarize("Item: rice straw\nFindings:\n- [SOURCE-BACKED] Ex-situ use")
        except GeneratorUnavailable as exc:
            self.skipTest(f"Bedrock not usable here: {exc}")
        self.assertTrue(out.ai_generated)


if __name__ == "__main__":
    unittest.main()
