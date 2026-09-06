"""Mock clinical ontology tool (symptom -> standardized concept -> candidate conditions).

Stands in for a SNOMED-CT / UMLS lookup service. Backed by
app/mock_data/clinical_ontology.json.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from app.json_utils import load_json

_DATA_PATH = Path(__file__).resolve().parent.parent / "mock_data" / "clinical_ontology.json"
_ONTOLOGY: Dict[str, Any] = load_json(_DATA_PATH)


def map_to_ontology(symptoms: List[str]) -> List[Dict[str, str]]:
    """Map free-text symptoms to standardized ontology concepts."""
    concepts = []
    for s in symptoms:
        key = s.strip().lower()
        if key in _ONTOLOGY["symptom_concepts"]:
            concepts.append({"symptom": key, **_ONTOLOGY["symptom_concepts"][key]})
    return concepts


def candidate_conditions(concepts: List[Dict[str, str]]) -> List[str]:
    """Given mapped concepts, return the union of candidate conditions."""
    conditions: set[str] = set()
    for c in concepts:
        conditions.update(_ONTOLOGY["concept_to_condition"].get(c["concept_id"], []))
    return sorted(conditions)


def recent_updates() -> List[Dict[str, str]]:
    return _ONTOLOGY.get("recent_updates", [])
