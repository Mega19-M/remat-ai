# RE:MAT AI — The Safe Recovery & Disposal Intelligence Agent

> *Every waste item deserves the right next decision — not just another report.*

RE:MAT AI investigates what to do with a waste item. **Strands Decider 2B** chooses which investigation tool runs next
(tool-selection pattern), a **deterministic safety guardrail** can overrule it, **Amazon Bedrock** (through the **Strands Agents SDK**)
writes a short unverified summary, and every finding is labelled `SOURCE-BACKED`, `HYPOTHESIS`, `UNKNOWN` or `SAFETY RULE`.

Built for the AWS Builder Center *Agent Experience Hackathon: Strands Decider*. Tags: `#strands-decider-2b` `#tool-selection`.
Not endorsed by AWS. Decision support only, not professional advice.

## Status (read this first)

| Part | State |
|---|---|
| Workflow, safety guardrail, tools, datasets, Streamlit UI | Written. Workflow/safety tests pass (see below). |
| Strands Decider client | Supports local `POST /v1/systemone` and hosted HTTPS `POST /v1/decide` with bearer auth. Contract-tested; your hosted credentials should be verified from your machine. |
| Bedrock summary via Strands Agents SDK | Written, **not yet run** (no AWS credentials or package access when built). |
| Demo mode | Works with no AWS and no Decider. Always labelled **SIMULATED**. |

First thing to do on your machine: run `python scripts/check_decider.py`. It prints the raw Decider response. The public docs do not spell out
the field names inside a `choice` answer, so the client parses several shapes and **raises an error instead of guessing**. If it errors, send the raw output
and the parser can be adjusted in `agent/decider.py::parse_choice_answer`.

## Windows setup (PowerShell or Command Prompt)

**Python version.** Package metadata could not be checked when this was written, so this is advice, not a verified fact:
PyTorch (needed by the Decider) usually lags new Python releases, so **Python 3.15 will probably not work**. Try 3.14.7 first; if `pip` fails, install
**Python 3.12** from python.org and use `py -3.12` below. Check what you have with `py -0`.

1. **Extract the ZIP.** Right-click `remat-ai.zip` → *Extract All…* (e.g. to `C:\Users\you\remat-ai`).
2. **Open in VS Code.** `cd C:\Users\you\remat-ai` then `code .`
3. **Create a virtual environment for the app:**
   ```
   py -3.14 -m venv .venv
   ```
4. **Activate it.**
   - PowerShell: `.\.venv\Scripts\Activate.ps1`
   - If PowerShell says scripts are disabled: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (then retry), **or** use Command Prompt: `.venv\Scripts\activate.bat`
5. **Install the app dependencies:** `python -m pip install --upgrade pip` then `pip install -r requirements.txt`
6. **Configure AWS credentials safely** (only needed for live mode):
   - Install the AWS CLI, then run `aws configure` and paste an IAM user's access key (they are stored in `%UserProfile%\.aws\`, outside this folder).
   - Or use `aws configure sso` / `AWS_PROFILE`. Never paste keys into code, `.env.example` or Git.
7. **Verify Bedrock model access:**
   - In the AWS console → Amazon Bedrock → *Model access*, enable a text model in your region.
   - IAM permissions needed: `bedrock:InvokeModel`, `bedrock:InvokeModelWithResponseStream` (generation) plus, for the check script, `sts:GetCallerIdentity`, `bedrock:ListFoundationModels`, `bedrock:ListInferenceProfiles`.
   - Run `python scripts/check_bedrock.py` to list ids, then `python scripts/check_bedrock.py <model-id>` to test one.
   - Put the working id in `.env` (copy `.env.example`) as `REMAT_BEDROCK_MODEL_ID=...`, or type it in the app sidebar. No default is shipped because I could not verify a current id for your account.
8. **Start the Decider server** (second terminal, separate environment, ~4.5 GB first-run download):
   ```
   py -3.12 -m venv .venv-decider
   .\.venv-decider\Scripts\Activate.ps1
   pip install -r requirements-decider.txt
   strands-decider serve StrandsAgents/strands-decider-2B-hobson-v21 --port 8000
   ```
   (Per the model card, `--device cpu` is available; it binds to 127.0.0.1 with no authentication, so keep it local.) Check it: `python scripts/check_decider.py`
9. **Configure the Decider endpoint.** For a local server, leave `REMAT_DECIDER_URL=http://127.0.0.1:8000`. For a hosted HTTPS endpoint, set `REMAT_DECIDER_URL=https://YOUR-HOSTED-DECIDER` and keep its API key in your private `.env` as `DECIDER_API_KEY=YOUR_ROTATED_KEY`. Never commit or paste the real key. The hosted client calls `/v1/decide` and sends `Authorization: Bearer …`; local mode calls `/v1/systemone`. In the decision trace, the model name is taken from the API response when provided; a confidence value is displayed only when the API returns one.
10. **Run the app** (first terminal, `.venv` active): `streamlit run app.py`  → opens http://localhost:8501.
   Choose **Demo** in the sidebar to explore without AWS/Decider, or **Live**.
11. **Run the tests:** `python -m unittest discover -s tests -t . -v` (works without extra packages; `pytest` also works if installed).

### Troubleshooting
| Problem | Fix |
|---|---|
| `py` not found | Reinstall Python from python.org with *py launcher* ticked. |
| `Activate.ps1 cannot be loaded` | See step 4. |
| `No matching distribution` / build errors | Wrong Python version for torch/strands. Use Python 3.12. |
| `Cannot reach the Decider server` | Start `strands-decider serve …`; check the URL in the sidebar. |
| `AWS credentials were not found` | Step 6. |
| `AccessDenied` / `ResourceNotFound` | Enable the model (step 7); use an id from `check_bedrock.py`; check the region. |
| Throttling | Wait and retry. |
| Decider response error | Run `scripts/check_decider.py` and read the raw JSON. |
| `ModuleNotFoundError: agent` | Run commands from the project folder. |

## How it works
See `docs/ARCHITECTURE.md` (Mermaid diagram), `docs/DECISION_PATTERN.md` (reusable pattern), `docs/DEMO_SCRIPT.md`, `docs/SUBMISSION.md`.

## Tests (what they do and do not prove)
- `tests/test_safety.py`, `tests/test_workflow.py`: unknown chemical → guardrail blocks recovery and escalates; emergency wording → escalation first; no DIY wording in any dataset or tool output; invalid input; recovery vs disposal; tool errors; Bedrock unavailable; demo mode marked SIMULATED.
- `tests/test_decider_integration.py`: **contract tests** against a local fake HTTP server (these prove the client follows the documented request format, **not** that the real model behaves well) plus **live tests that are skipped** unless a Decider server is running / `REMAT_BEDROCK_MODEL_ID` is set.

## Limitations
Small India-focused dataset; sources are pointers whose clause-level wording was not verified; no facility directory (so no facility is ever named); photos are not analyzed; keyword matching can be wrong; no measured latency, accuracy or impact figures exist yet.

## License
Apache-2.0 (see `LICENSE`), matching the Decider's license.
