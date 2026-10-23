"""Check the local Strands Decider server and print the RAW response for one sample question.

Usage:  python scripts/check_decider.py [http://127.0.0.1:8000]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.decider import ACTION_CRITERIA, DeciderError, HttpDeciderBackend  # noqa: E402

url = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("REMAT_DECIDER_URL", "http://127.0.0.1:8000")
backend = HttpDeciderBackend(url, 60)
try:
    print("Health:", json.dumps(backend.health(), indent=2))
    options = [o for o in ACTION_CRITERIA if o != "conclude"]
    d = backend.choose("WASTE CASE\nItem: unlabelled blue drum\nDanger: unlabelled, chemical smell", options)
    print("\nSelected:", d.selected, "| confidence:", d.confidence, "| probabilities:", d.probabilities)
    print("\nRAW RESPONSE (send this to whoever helps you debug):")
    print(json.dumps(d.raw_response, indent=2))
except DeciderError as exc:
    print("FAILED:", exc)
    if getattr(exc, "raw", None) is not None:
        print("Raw:", json.dumps(exc.raw, indent=2)[:2000])
    sys.exit(1)
