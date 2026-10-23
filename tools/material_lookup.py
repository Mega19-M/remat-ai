"""Deterministic material lookup against the curated dataset (no AI involved)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

from config import DATA_DIR
from agent.schemas import WasteCase


@lru_cache(maxsize=1)
def load_materials() -> list[dict[str, Any]]:
    return json.loads((DATA_DIR / "materials.json").read_text(encoding="utf-8"))["materials"]


@lru_cache(maxsize=1)
def load_sources() -> dict[str, dict[str, Any]]:
    return json.loads((DATA_DIR / "sources.json").read_text(encoding="utf-8"))["sources"]


@lru_cache(maxsize=1)
def load_scenarios() -> list[dict[str, Any]]:
    return json.loads((DATA_DIR / "demo_scenarios.json").read_text(encoding="utf-8"))["scenarios"]


@dataclass
class MaterialMatch:
    material: dict[str, Any]
    matched_terms: list[str]


def _text_of(case: WasteCase) -> str:
    return " ".join([case.waste_name, case.description, case.category]).lower()


def _has_term(term: str, text: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(term.lower()) + r"(?![a-z0-9])", text) is not None


def lookup_material(case: WasteCase) -> Optional[MaterialMatch]:
    """Return the best dataset match for the case, or None.

    Scoring: the number of distinct alias hits. Ties are broken by dataset order
    EXCEPT that the more dangerous class wins, so a mixed description is never
    classified as the safer material.
    """
    text = _text_of(case)
    rank = {"unknown_high_risk": 0, "regulated": 1, "routine": 2}
    best: Optional[tuple[int, int, MaterialMatch]] = None
    for material in load_materials():
        hits = [a for a in material["aliases"] + [material["name"]] if _has_term(a, text)]
        if not hits:
            continue
        key = (len(set(hits)), -rank.get(material["hazard_class"], 2))
        if best is None or key > (best[0], best[1]):
            best = (key[0], key[1], MaterialMatch(material, sorted(set(hits))))
    return best[2] if best else None


def get_material(material_id: Optional[str]) -> Optional[dict[str, Any]]:
    if not material_id:
        return None
    return next((m for m in load_materials() if m["id"] == material_id), None)
