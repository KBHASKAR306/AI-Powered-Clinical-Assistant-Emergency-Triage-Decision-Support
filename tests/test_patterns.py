"""
Offline test suite. No ANTHROPIC_API_KEY required — reasoning nodes fall
back to the deterministic heuristic in app/llm.py, so every branch of every
pattern's graph can be exercised in CI.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

from fastapi.testclient import TestClient

from app.json_utils import load_json
from app.llm import get_completion
from app.main import app
from app.patterns import multi_agent, planning, react, reflection, rewoo, tool_use

client = TestClient(app)


def test_json_loader_accepts_utf8_bom(tmp_path):
    json_path = tmp_path / "sample.json"
    json_path.write_text('\ufeff{"status": "ok"}', encoding="utf-8")

    assert load_json(json_path) == {"status": "ok"}


def test_llm_completion_is_valid_json():
    value = get_completion("chest discomfort and shortness of breath")
    parsed = json.loads(value)
    assert isinstance(parsed, dict)
    assert "text" in parsed


def test_trace_details_are_json_safe_objects():
    result = tool_use.run(symptoms=["chest discomfort", "shortness of breath"], patient_id="P-1001")
    assert all(not isinstance(step["detail"], str) for step in result["trace"])

CRITICAL_SYMPTOMS = ["chest discomfort", "shortness of breath"]
LOW_RISK_SYMPTOMS = ["sore throat", "runny nose"]


# --------------------------------------------------------------------------
# Reflection
# --------------------------------------------------------------------------
def test_reflection_triggers_on_audited_symptom():
    result = reflection.run(symptoms=["difficulty in breathing"])
    assert result["final_output"]["reflection_triggered"] is True
    assert "REVISED" in result["final_output"]["recommendation"]


def test_reflection_passthrough_on_unaudited_symptom():
    result = reflection.run(symptoms=["mild headache"])
    assert result["final_output"]["reflection_triggered"] is False


# --------------------------------------------------------------------------
# Tool Use
# --------------------------------------------------------------------------
def test_tool_use_escalates_on_critical_symptoms():
    result = tool_use.run(symptoms=CRITICAL_SYMPTOMS, patient_id="P-1001")
    assert result["final_output"]["clinically_significant"] is True
    assert "ALERT" in result["final_output"]["response"]


def test_tool_use_logs_routine_on_low_risk_symptoms():
    result = tool_use.run(symptoms=LOW_RISK_SYMPTOMS)
    assert result["final_output"]["clinically_significant"] is False


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------
def test_planning_escalates_for_critical_case():
    result = planning.run(symptoms=CRITICAL_SYMPTOMS)
    assert result["final_output"]["escalated"] is True
    assert len(result["trace"]) > 0


def test_planning_completes_for_low_risk_case():
    result = planning.run(symptoms=LOW_RISK_SYMPTOMS)
    assert result["final_output"]["escalated"] is False


# --------------------------------------------------------------------------
# ReAct
# --------------------------------------------------------------------------
def test_react_full_trace_present():
    result = react.run(symptoms=CRITICAL_SYMPTOMS, patient_id="P-1001")
    step_names = [t["step"] for t in result["trace"]]
    assert step_names == [
        "Consider possible causes",
        "Query EHR for history and check recent vitals",
        "Evaluate differential diagnosis",
        "Order tests (e.g., ECG, troponin) and alert provider",
        "Recommendation and reasoning trace",
    ]
    assert "recommendation" in result["final_output"]


# --------------------------------------------------------------------------
# ReWOO
# --------------------------------------------------------------------------
def test_rewoo_maps_known_symptoms_to_ontology():
    result = rewoo.run(symptoms=CRITICAL_SYMPTOMS)
    linked = result["final_output"]["linked_concepts"]
    assert len(linked) == 2
    assert any("acute coronary syndrome" in c for c in result["final_output"]["candidate_conditions"])


def test_rewoo_handles_unknown_symptoms_gracefully():
    result = rewoo.run(symptoms=["an unmapped made-up symptom"])
    assert result["final_output"]["linked_concepts"] == []
    assert result["final_output"]["triggered_tools"] == ["routine_advice_workflow"]


# --------------------------------------------------------------------------
# Multi-Agent
# --------------------------------------------------------------------------
def test_multi_agent_escalates_and_aggregates():
    result = multi_agent.run(symptoms=CRITICAL_SYMPTOMS, patient_id="P-1001")
    out = result["final_output"]
    assert out["referral"] == "Emergency department"
    assert out["escalation_status"]["emergency_team_notified"] is True
    assert out["admin_status"]["patient_on_file"] is True
    step_names = {t["step"] for t in result["trace"]}
    assert {"Triage Agent", "Admin Agent", "Coordinator / Mediator Aggregates Agent Outputs"} <= step_names


def test_multi_agent_no_escalation_for_low_risk():
    result = multi_agent.run(symptoms=LOW_RISK_SYMPTOMS)
    out = result["final_output"]
    assert out["referral"] == "Standard follow-up"
    assert out["escalation_status"]["emergency_team_notified"] is False


# --------------------------------------------------------------------------
# FastAPI endpoints
# --------------------------------------------------------------------------
def test_root_lists_patterns():
    resp = client.get("/")
    assert resp.status_code == 200
    assert set(resp.json()["patterns_available"]) == {
        "reflection", "tool-use", "planning", "react", "rewoo", "multi-agent",
    }


def test_list_and_get_patient():
    resp = client.get("/patients")
    assert resp.status_code == 200
    ids = resp.json()["patient_ids"]
    assert "P-1001" in ids

    resp = client.get(f"/patients/{ids[0]}")
    assert resp.status_code == 200
    assert "name" in resp.json()


def test_get_unknown_patient_404():
    resp = client.get("/patients/NOT-REAL")
    assert resp.status_code == 404


def test_run_pattern_endpoint():
    resp = client.post(
        "/patterns/tool-use",
        json={"patient_id": "P-1001", "symptoms": CRITICAL_SYMPTOMS},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["pattern"] == "Tool Use"
    assert body["final_output"]["clinically_significant"] is True


def test_run_unknown_pattern_404():
    resp = client.post("/patterns/does-not-exist", json={"symptoms": ["x"]})
    assert resp.status_code == 404


def test_full_pipeline_endpoint():
    resp = client.post(
        "/pipeline/full",
        json={"patient_id": "P-1003", "symptoms": ["palpitations", "dizziness"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["steps"].keys()) == {
        "1_rewoo", "2_react", "3_tool_use", "4_planning", "5_multi_agent", "6_reflection",
    }
    assert "final_recommendation" in body["summary"]
