"""
Tool Use Pattern
================
Matches the flow on p.4 of the source document:

  Patient Input/Query
        -> Intent Recognition and Context Parsing (parse symptoms, extract medication data)
        -> Identify Relevant Tool/API
        -> Call External Tool API
        -> External API/Tool Call (medical database)
        -> Is the outcome clinically significant?
             -> [yes] Alert User and Escalate to Clinician
             -> [no]  Log Result and Continue Routine Advice
        -> Respond to User and Forward to Action Module

Action Module     -> connects to the mock medical decision-support tool.
Perception Module -> parses the raw patient input before the tool call.
"""
from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from app.tools import ehr_tool, medical_db_tool

_SIGNIFICANT_LEVELS = {"critical", "high"}


class ToolUseState(TypedDict, total=False):
    patient_id: str
    symptoms: List[str]
    notes: str
    parsed_context: Dict[str, Any]
    identified_tool: str
    tool_result: List[Dict[str, Any]]
    clinically_significant: bool
    response: str
    trace: List[Dict[str, Any]]


def _log(state: ToolUseState, step: str, detail: Any) -> None:
    if isinstance(detail, str):
        detail = {"value": detail}
    state.setdefault("trace", []).append({"step": step, "detail": detail})


def intent_recognition(state: ToolUseState) -> ToolUseState:
    ehr_record = ehr_tool.query_ehr(state.get("patient_id", "")) if state.get("patient_id") else None
    context = {
        "symptoms": state["symptoms"],
        "notes": state.get("notes", ""),
        "known_medications": ehr_record["medications"] if ehr_record else [],
        "known_history": ehr_record["history"] if ehr_record else [],
    }
    state["parsed_context"] = context
    _log(state, "Intent Recognition and Context Parsing", context)
    return state


def identify_tool(state: ToolUseState) -> ToolUseState:
    tool_name = "medical_decision_support_db.lookup_symptoms"
    state["identified_tool"] = tool_name
    _log(state, "Identify Relevant Tool/API", tool_name)
    return state


def call_external_tool(state: ToolUseState) -> ToolUseState:
    matches = medical_db_tool.lookup_symptoms(state["symptoms"])
    state["tool_result"] = matches
    _log(state, "Call External Tool API / External API Call (medical database)", matches)
    return state


def check_significance(state: ToolUseState) -> ToolUseState:
    top_risk = state["tool_result"][0]["risk_level"] if state["tool_result"] else "unknown"
    significant = top_risk in _SIGNIFICANT_LEVELS
    state["clinically_significant"] = significant
    _log(state, "Is the outcome clinically significant?", {"top_risk": top_risk, "significant": significant})
    return state


def alert_and_escalate(state: ToolUseState) -> ToolUseState:
    top = state["tool_result"][0]
    state["response"] = (
        f"ALERT: findings match '{top['condition']}' (risk={top['risk_level']}). "
        f"Recommended tests: {', '.join(top['recommended_tests']) or 'none'}. "
        f"Action: {top['recommended_action']} Escalating to clinician now."
    )
    _log(state, "Alert User and Escalate to Clinician", {"response": state["response"]})
    return state


def log_and_continue(state: ToolUseState) -> ToolUseState:
    if state["tool_result"]:
        top = state["tool_result"][0]
        state["response"] = (
            f"Findings suggest '{top['condition']}' (risk={top['risk_level']}). "
            f"Routine advice: {top['recommended_action']}"
        )
    else:
        state["response"] = "No matching condition pattern found; general symptomatic care advice given."
    _log(state, "Log Result and Continue Routine Advice", {"response": state["response"]})
    return state


def route_significance(state: ToolUseState) -> str:
    return "alert_and_escalate" if state["clinically_significant"] else "log_and_continue"


def respond(state: ToolUseState) -> ToolUseState:
    _log(state, "Respond to User and Forward to Action Module", {"response": state["response"]})
    return state


def build_graph():
    graph = StateGraph(ToolUseState)
    graph.add_node("intent_recognition", intent_recognition)
    graph.add_node("identify_tool", identify_tool)
    graph.add_node("call_external_tool", call_external_tool)
    graph.add_node("check_significance", check_significance)
    graph.add_node("alert_and_escalate", alert_and_escalate)
    graph.add_node("log_and_continue", log_and_continue)
    graph.add_node("respond", respond)

    graph.set_entry_point("intent_recognition")
    graph.add_edge("intent_recognition", "identify_tool")
    graph.add_edge("identify_tool", "call_external_tool")
    graph.add_edge("call_external_tool", "check_significance")
    graph.add_conditional_edges(
        "check_significance",
        route_significance,
        {"alert_and_escalate": "alert_and_escalate", "log_and_continue": "log_and_continue"},
    )
    graph.add_edge("alert_and_escalate", "respond")
    graph.add_edge("log_and_continue", "respond")
    graph.add_edge("respond", END)
    return graph.compile()


_APP = build_graph()


def run(symptoms: List[str], patient_id: str = "", notes: str = "", **_ignored) -> Dict[str, Any]:
    state: ToolUseState = {"symptoms": symptoms, "patient_id": patient_id, "notes": notes, "trace": []}
    result = _APP.invoke(state)
    return {
        "pattern": "Tool Use",
        "trace": result["trace"],
        "final_output": {
            "clinically_significant": result["clinically_significant"],
            "response": result["response"],
        },
    }
