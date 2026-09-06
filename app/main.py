"""
AI-Powered Clinical Assistant for Emergency Triage & Decision Support
----------------------------------------------------------------------
FastAPI service implementing Demo-2 of the "Agentic AI Architectures and
Design Patterns" module: Reflection, Tool Use, Planning, ReAct, ReWOO, and
the Multi-Agent pattern, each as its own LangGraph graph, plus a combined
end-to-end pipeline that chains them together.

*** EDUCATIONAL / PORTFOLIO DEMO ONLY ***
All patient data, guidelines, and ontology mappings are synthetic and
included only under app/mock_data/. This is NOT a medical device, has not
been validated against real clinical data, and must never be used to make
real triage or treatment decisions.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.patterns import multi_agent, planning, react, reflection, rewoo, tool_use
from app.schemas import PatientQuery, PatternResult
from app.tools import ehr_tool

app = FastAPI(
    title="AI-Powered Clinical Assistant — Agentic Design Patterns Demo",
    description=(
        "Demo-2 implementation: Reflection, Tool Use, Planning, ReAct, ReWOO, "
        "and Multi-Agent patterns for emergency triage & decision support. "
        "Educational demo only — synthetic data, not a real medical device."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_PATTERNS = {
    "reflection": reflection.run,
    "tool-use": tool_use.run,
    "planning": planning.run,
    "react": react.run,
    "rewoo": rewoo.run,
    "multi-agent": multi_agent.run,
}


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "AI-Powered Clinical Assistant — Agentic Design Patterns Demo",
        "disclaimer": "Educational demo only. Synthetic data. Not a medical device.",
        "patterns_available": list(_PATTERNS.keys()),
        "endpoints": [
            "POST /patterns/{pattern_name}",
            "POST /pipeline/full",
            "GET  /patients",
            "GET  /patients/{patient_id}",
        ],
    }


@app.get("/patients")
def list_patients() -> Dict[str, Any]:
    return {"patient_ids": ehr_tool.list_patient_ids()}


@app.get("/patients/{patient_id}")
def get_patient(patient_id: str) -> Dict[str, Any]:
    record = ehr_tool.query_ehr(patient_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown synthetic patient_id '{patient_id}'")
    return record


@app.post("/patterns/{pattern_name}", response_model=PatternResult)
def run_pattern(pattern_name: str, query: PatientQuery) -> PatternResult:
    fn = _PATTERNS.get(pattern_name)
    if fn is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown pattern '{pattern_name}'. Choose one of: {list(_PATTERNS.keys())}",
        )
    result = fn(symptoms=query.symptoms, patient_id=query.patient_id or "", notes=query.notes or "")
    return PatternResult(**result)


@app.post("/pipeline/full")
def run_full_pipeline(query: PatientQuery) -> Dict[str, Any]:
    """
    Chains all six patterns into one end-to-end run, roughly mirroring how a
    real deployment would combine them (see README "How the patterns
    combine"):

      1. ReWOO       - structure the raw complaint into ontology-linked concepts
      2. ReAct        - reason + act in an interleaved loop to reach a working diagnosis
      3. Tool Use     - confirm findings against the decision-support tool
      4. Planning     - lay out the multi-step care plan
      5. Multi-Agent  - fan out to Triage/Admin agents and coordinate the response
      6. Reflection   - check the outcome against the prior-case audit log and
                        revise if this symptom pattern has a history of slow response
    """
    kwargs = dict(symptoms=query.symptoms, patient_id=query.patient_id or "", notes=query.notes or "")
    steps = {
        "1_rewoo": rewoo.run(**kwargs),
        "2_react": react.run(**kwargs),
        "3_tool_use": tool_use.run(**kwargs),
        "4_planning": planning.run(**kwargs),
        "5_multi_agent": multi_agent.run(**kwargs),
        "6_reflection": reflection.run(**kwargs),
    }
    return {
        "disclaimer": "Educational demo only. Synthetic data. Not a medical device.",
        "patient_id": query.patient_id,
        "symptoms": query.symptoms,
        "steps": steps,
        "summary": {
            "final_recommendation": steps["6_reflection"]["final_output"]["recommendation"],
            "escalated": steps["5_multi_agent"]["final_output"]["escalation_status"],
            "referral": steps["5_multi_agent"]["final_output"]["referral"],
        },
    }
