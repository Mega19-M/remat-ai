"""Central configuration for RE:MAT AI.

Everything is read from environment variables (optionally loaded from a local
.env file by the Streamlit app). No secrets live in the code. AWS credentials
come from the standard AWS credential chain (aws configure / AWS_PROFILE /
AWS_ACCESS_KEY_ID ...), never from this file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def load_dotenv_if_present(path: Path | None = None) -> None:
    """Tiny .env loader (KEY=VALUE lines). Existing environment variables win."""
    path = path or (ROOT / ".env")
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    # Strands Decider 2B is served locally with:  strands-decider serve <model> --port 8000
    decider_url: str = "http://127.0.0.1:8000"
    # Optional "model" field for the local /v1/systemone request body.
    decider_model_field: str = ""
    # Hosted Decider token. Keep in .env / environment only; never show in the UI.
    decider_api_key: str = ""
    decider_timeout_s: float = 60.0
    # Amazon Bedrock (used through the Strands Agents SDK) for the generative summary.
    aws_region: str = "us-east-1"
    bedrock_model_id: str = ""  # you must choose a model you have access to (see README)
    max_steps: int = 5

    @staticmethod
    def from_env() -> "Settings":
        return Settings(
            decider_url=os.environ.get("REMAT_DECIDER_URL", "http://127.0.0.1:8000").rstrip("/"),
            decider_model_field=os.environ.get("REMAT_DECIDER_MODEL_FIELD", ""),
            decider_api_key=os.environ.get("DECIDER_API_KEY", ""),
            decider_timeout_s=float(os.environ.get("REMAT_DECIDER_TIMEOUT_S", "60")),
            aws_region=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1",
            bedrock_model_id=os.environ.get("REMAT_BEDROCK_MODEL_ID", ""),
            max_steps=int(os.environ.get("REMAT_MAX_STEPS", "5")),
        )
