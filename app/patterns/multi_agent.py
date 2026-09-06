"""
Multi-Agent Pattern
====================
Matches the flow on p.10 of the source document:

  Patient Input/Query
        -> Broadcast to Specialized Agents
             -> Triage Agent   \\
             -> Admin Agent     >  run concurrently
        -> Critical Symptoms Detected?
             -> [yes] Escalate to Emergency Team -> Notify Human Clinician -> Continue to Coordination
             -> [no]  Continue to Coordination
        -> Coordinator / Mediator Aggregates Agent Outputs
        -> Final Output (Recommendation, Plan, Referral)

Collaboration Module -> the broadcast + fan-in around the two specialized
                         agents and the coordinator/mediator.
Action Module         -> each agent's own task (triage assessment / admin
                         record-keeping).

Implementation note: Triage Agent and Admin Agent run as a genuine
LangGraph fan-out/fan-in (both fire off "broadcast" in the same superstep).
Because they execute concurrently, every node in this module returns only
the keys it *changes* (never the whole state), and `trace` is declared as
an Annotated/operator.add channel so both agents' log entries merge instead
of clobbering each other.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from app.tools import ehr_tool, medical_db_tool

_SIGNIFICANT_LEVELS = {"critical", "high"}


class MultiAgentState(TypedDict, total=False):
    patient_id: str
    symptoms: List[str]
    notes: str
    triage_output: Dict[str, Any]
    admin_output: Dict[str, Any]
    critical_detected: bool
    escalation: Dict[str, Any]
    aggregated_output: Dict[str, Any]
    trace: Annotated[List[Dict[str, Any]], operator.add]


def _entry(step: str, detail: Any) -> List[Dict[str, Any]]:
    return [{"step": step, "detail": detail}]


def broadcast(state: MultiAgentState) -> Dict[str, Any]:
    return {"trace": _entry("Broadcast to Specialized Agents", {"agents": ["Triage Agent", "Admin Agent"]})}


def triage_agent(state: MultiAgentState) -> Dict[str, Any]:
    matches = medical_db_tool.lookup_symptoms(state["symptoms"])
    output = {
        "assessment": matches[0] if matches else {"condition": "unclear", "risk_level": "unknown"},
        "all_matches": matches,
    }
    return {"triage_output": output, "trace": _entry("Triage Agent", output)}


def admin_agent(state: MultiAgentState) -> Dict[str, Any]:
    record = ehr_tool.query_ehr(state.get("patient_id", "")) if state.get("patient_id") else None
    output = {
        "ehr_updated": True,
        "patient_on_file": record is not None,
        "billing_code_drafted": "TRIAGE-VISIT-GENERIC",
    }
    return {"admin_output": output, "trace": _entry("Admin Agent", output)}


def check_critical(state: MultiAgentState) -> Dict[str, Any]:
    risk = state["triage_output"]["assessment"].get("risk_level", "unknown")
    critical = risk in _SIGNIFICANT_LEVELS
    detail = {"risk_level": risk, "critical_detected": critical}
    return {"critical_detected": critical, "trace": _entry("Critical Symptoms Detected?", detail)}


def escalate_emergency_team(state: MultiAgentState) -> Dict[str, Any]:
    escalation = {"emergency_team_notified": True, "reason": state["triage_output"]["assessment"]}
    return {"escalation": escalation, "trace": _entry("Escalate to Emergency Team", escalation)}


def notify_clinician(state: MultiAgentState) -> Dict[str, Any]:
    escalation = {**state.get("escalation", {}), "clinician_notified": True}
    return {"escalation": escalation, "trace": _entry("Notify Human Clinician", escalation)}


def continue_coordination(state: MultiAgentState) -> Dict[str, Any]:
    escalation = state.get("escalation") or {"emergency_team_notified": False, "clinician_notified": False}
    return {"escalation": escalation, "trace": _entry("Continue to Coordination", escalation)}


def route_critical(state: MultiAgentState) -> str:
    return "escalate_emergency_team" if state["critical_detected"] else "continue_coordination"


def coordinator_aggregate(state: MultiAgentState) -> Dict[str, Any]:
    aggregated = {
        "recommendation": state["triage_output"]["assessment"].get("recommended_action", "Routine care advice."),
        "plan": [m["condition"] for m in state["triage_output"].get("all_matches", [])] or ["No specific condition matched"],
        "referral": "Emergency department" if state["critical_detected"] else "Standard follow-up",
        "admin_status": state["admin_output"],
        "escalation_status": state.get("escalation", {}),
    }
    return {
        "aggregated_output": aggregated,
        "trace": _entry("Coordinator / Mediator Aggregates Agent Outputs", aggregated),
    }


def build_graph():
    graph = StateGraph(MultiAgentState)
    graph.add_node("broadcast", broadcast)
    graph.add_node("triage_agent", triage_agent)
    graph.add_node("admin_agent", admin_agent)
    graph.add_node("check_critical", check_critical)
    graph.add_node("escalate_emergency_team", escalate_emergency_team)
    graph.add_node("notify_clinician", notify_clinician)
    graph.add_node("continue_coordination", continue_coordination)
    graph.add_node("coordinator_aggregate", coordinator_aggregate)

    graph.set_entry_point("broadcast")
    # Fan-out: both specialized agents run off the same broadcast step.
    graph.add_edge("broadcast", "triage_agent")
    graph.add_edge("broadcast", "admin_agent")
    # Fan-in: check_critical waits for both agents to complete.
    graph.add_edge("triage_agent", "check_critical")
    graph.add_edge("admin_agent", "check_critical")
    graph.add_conditional_edges(
        "check_critical",
        route_critical,
        {"escalate_emergency_team": "escalate_emergency_team", "continue_coordination": "continue_coordination"},
    )
    graph.add_edge("escalate_emergency_team", "notify_clinician")
    graph.add_edge("notify_clinician", "coordinator_aggregate")
    graph.add_edge("continue_coordination", "coordinator_aggregate")
    graph.add_edge("coordinator_aggregate", END)
    return graph.compile()


_APP = build_graph()


def run(symptoms: List[str], patient_id: str = "", notes: str = "", **_ignored) -> Dict[str, Any]:
    state: MultiAgentState = {"symptoms": symptoms, "patient_id": patient_id, "notes": notes, "trace": []}
    result = _APP.invoke(state)
    return {
        "pattern": "Multi-Agent",
        "trace": result["trace"],
        "final_output": result["aggregated_output"],
    }
