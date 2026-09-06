"""
ReAct (Reasoning and Acting) Pattern
====================================
Matches the flow on p.6-7 of the source document:

  Patient Input/Query
        -> Consider possible causes                 [Think]
        -> Query EHR for history and check vitals    [Action]
        -> Evaluate differential diagnosis           [Think]
        -> Order tests (e.g. ECG, troponin) and
           alert provider                            [Action]
        -> Recommendation and reasoning trace         [Decision]

Cognitive Module -> the Think steps.
Action Module    -> the Action steps, executed immediately after each
                    reasoning step (interleaved, not batched).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from app.llm import get_completion
from app.tools import ehr_tool, medical_db_tool


class ReactState(TypedDict, total=False):
    patient_id: str
    symptoms: List[str]
    notes: str
    possible_causes: str
    ehr_snapshot: Dict[str, Any]
    differential_diagnosis: str
    tool_result: List[Dict[str, Any]]
    recommendation: str
    trace: List[Dict[str, Any]]


def _log(state: ReactState, step: str, detail: Any, tag: str) -> None:
    if isinstance(detail, str):
        detail = {"value": detail}
    state.setdefault("trace", []).append({"step": step, "tag": tag, "detail": detail})


def think_consider_causes(state: ReactState) -> ReactState:
    prompt = f"Patient reports: {state['symptoms']}. Notes: {state.get('notes', '')}. What are the possible causes?"
    state["possible_causes"] = get_completion(prompt)
    _log(
        state,
        "Consider possible causes",
        {"raw_response": json.loads(state["possible_causes"]) if isinstance(state["possible_causes"], str) else state["possible_causes"]},
        tag="Think",
    )
    return state


def act_query_ehr(state: ReactState) -> ReactState:
    record = ehr_tool.query_ehr(state.get("patient_id", "")) if state.get("patient_id") else None
    state["ehr_snapshot"] = record or {"note": "no EHR record on file for this patient_id"}
    _log(state, "Query EHR for history and check recent vitals", {"ehr_snapshot": state["ehr_snapshot"]}, tag="Action")
    return state


def think_evaluate_diagnosis(state: ReactState) -> ReactState:
    prompt = (
        f"Possible causes: {state['possible_causes']}\n"
        f"EHR snapshot: {state['ehr_snapshot']}\n"
        "Evaluate the differential diagnosis and the risk of a life-threatening condition."
    )
    state["differential_diagnosis"] = get_completion(prompt)
    _log(
        state,
        "Evaluate differential diagnosis",
        {"raw_response": json.loads(state["differential_diagnosis"]) if isinstance(state["differential_diagnosis"], str) else state["differential_diagnosis"]},
        tag="Think",
    )
    return state


def act_order_tests_alert(state: ReactState) -> ReactState:
    matches = medical_db_tool.lookup_symptoms(state["symptoms"])
    state["tool_result"] = matches
    if matches:
        top = matches[0]
        detail = {
            "tests_ordered": top["recommended_tests"],
            "provider_alerted": top["risk_level"] in {"critical", "high"},
            "risk_level": top["risk_level"],
        }
    else:
        detail = {"tests_ordered": [], "provider_alerted": False, "risk_level": "unknown"}
    _log(state, "Order tests (e.g., ECG, troponin) and alert provider", detail, tag="Action")
    return state


def recommendation_with_trace(state: ReactState) -> ReactState:
    matches = state.get("tool_result") or []
    if matches:
        top = matches[0]
        state["recommendation"] = (
            f"Working impression: {top['condition']} (risk={top['risk_level']}). "
            f"{top['recommended_action']}"
        )
    else:
        state["recommendation"] = "No high-confidence pattern match; recommend routine clinical review."
    _log(
        state,
        "Recommendation and reasoning trace",
        {"recommendation": state["recommendation"]},
        tag="Decision",
    )
    return state


def build_graph():
    graph = StateGraph(ReactState)
    graph.add_node("think_consider_causes", think_consider_causes)
    graph.add_node("act_query_ehr", act_query_ehr)
    graph.add_node("think_evaluate_diagnosis", think_evaluate_diagnosis)
    graph.add_node("act_order_tests_alert", act_order_tests_alert)
    graph.add_node("recommendation_with_trace", recommendation_with_trace)

    graph.set_entry_point("think_consider_causes")
    graph.add_edge("think_consider_causes", "act_query_ehr")
    graph.add_edge("act_query_ehr", "think_evaluate_diagnosis")
    graph.add_edge("think_evaluate_diagnosis", "act_order_tests_alert")
    graph.add_edge("act_order_tests_alert", "recommendation_with_trace")
    graph.add_edge("recommendation_with_trace", END)
    return graph.compile()


_APP = build_graph()


def run(symptoms: List[str], patient_id: str = "", notes: str = "", **_ignored) -> Dict[str, Any]:
    state: ReactState = {"symptoms": symptoms, "patient_id": patient_id, "notes": notes, "trace": []}
    result = _APP.invoke(state)
    return {
        "pattern": "ReAct",
        "trace": result["trace"],
        "final_output": {"recommendation": result["recommendation"]},
    }
