from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PatientQuery(BaseModel):
    patient_id: Optional[str] = Field(
        default=None,
        description="Optional synthetic EHR patient id, e.g. 'P-1001'. See /patients for valid ids.",
    )
    symptoms: List[str] = Field(
        ..., description="Reported symptoms as free text, e.g. ['chest discomfort', 'shortness of breath']"
    )
    notes: Optional[str] = Field(default=None, description="Free-text additional context from the patient/query.")


class PatternResult(BaseModel):
    pattern: str
    trace: List[Dict[str, Any]]
    final_output: Dict[str, Any]
