"""Mock Electronic Health Record (EHR) tool.

Stands in for a real EHR/FHIR integration. Backed by app/mock_data/ehr_records.json.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from app.json_utils import load_json

_DATA_PATH = Path(__file__).resolve().parent.parent / "mock_data" / "ehr_records.json"
_RECORDS: Dict[str, Any] = load_json(_DATA_PATH)


def query_ehr(patient_id: str) -> Optional[Dict[str, Any]]:
    """Return the synthetic EHR record for a patient_id, or None if unknown."""
    return _RECORDS.get(patient_id)


def list_patient_ids() -> list[str]:
    return list(_RECORDS.keys())
