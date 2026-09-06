"""Mock medical decision-support tool (symptom -> condition / risk / next steps).

Stands in for a real clinical decision support system. Backed by
app/mock_data/medical_guidelines.json.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from app.json_utils import load_json

_DATA_PATH = Path(__file__).resolve().parent.parent / "mock_data" / "medical_guidelines.json"
_RULES: List[Dict[str, Any]] = load_json(_DATA_PATH)["condition_rules"]

_RISK_ORDER = {"low": 0, "moderate": 1, "high": 2, "critical": 3}


def lookup_symptoms(symptoms: List[str]) -> List[Dict[str, Any]]:
    """
    Cross-reference reported symptoms against the synthetic guideline
    database. Returns matching condition rules, ranked by risk level
    (highest first), each including the recommended tests/action.
    """
    normalized = {s.strip().lower() for s in symptoms}
    matches = []
    for rule in _RULES:
        overlap = normalized.intersection(s.lower() for s in rule["trigger_symptoms"])
        if len(overlap) >= rule["min_matches"]:
            matches.append({**rule, "matched_symptoms": sorted(overlap)})
    matches.sort(key=lambda r: _RISK_ORDER.get(r["risk_level"], 0), reverse=True)
    return matches


def highest_risk_level(symptoms: List[str]) -> str:
    matches = lookup_symptoms(symptoms)
    return matches[0]["risk_level"] if matches else "unknown"
