"""
Planning Pattern
================
Matches the flow on p.6 of the source document:

  Patient History and New Complaint
        -> Set Care Benchmark
        -> Generate Multi-Step Plan or Diagnosis
        -> Execute Plan Step-by-Step with Monitoring  <---+
        -> Any critical change or new data?               |
             -> [yes] Adjust Plan Dynamically  ------------+ (loop, bounded)
             -> [no]  Proceed with Current Plan  ----------+
        -> Complete Plan or Escalate to Clinician
        -> Report to Patient and Update EHR

Cognitive Module -> devises the multi-step strategy.
Action Module    -> executes/coordinates each step.
"""
from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from app.llm import get_completion
from app.tools import medical_db_tool

MAX_MONITORING_STEPS = 3


class PlanningState(TypedDict, total=False):
    symptoms: List[str]
    notes: str
    care_benchmark: str
    plan: List[str]
    step_index: int
    critical_change: bool
    escalated: bool
    final_status: str
    trace: List[Dict[str, Any]]


def _log(state: PlanningState, step: str, detail: Any) -> None:
    if isinstance(detail, str):
        detail = {"value": detail}
    state.setdefault("trace", []).append({"step": step, "detail": detail})


def set_benchmark(state: PlanningState) -> PlanningState:
    matches = medical_db_tool.lookup_symptoms(state["symptoms"])
    benchmark = (
        f"Target: assess and act within a timeframe appropriate for "
        f"risk level '{matches[0]['risk_level']}'." if matches
        else "Target: routine assessment within standard triage window."
    )
    state["care_benchmark"] = benchmark
    _log(state, "Set Care Benchmark", {"benchmark": benchmark})
    return state


def generate_plan(state: PlanningState) -> PlanningState:
    prompt = (
        f"Patient symptoms: {state['symptoms']}. Care benchmark: {state['care_benchmark']}.\n"
        "List a short multi-step care plan (3-5 steps) to reach that benchmark."
    )
    plan_text = get_completion(prompt)
    # Keep both the raw LLM text and a simple structured step list for execution.
    state["plan"] = [
        "Assess vitals and symptom severity",
        "Cross-check against medical decision-support tool",
        "Determine escalation need",
        "Coordinate next action (clinician / emergency / routine)",
    ]
    state["step_index"] = 0
    _log(state, "Generate Multi-Step Plan or Diagnosis", {"plan_text": plan_text, "structured_plan": state["plan"]})
    return state


def execute_step(state: PlanningState) -> PlanningState:
    idx = state.get("step_index", 0)
    current_step = state["plan"][idx] if idx < len(state["plan"]) else "Final review"
    _log(state, f"Execute Plan Step-by-Step with Monitoring (step {idx + 1})", current_step)
    state["step_index"] = idx + 1
    return state


def check_critical_change(state: PlanningState) -> PlanningState:
    matches = medical_db_tool.lookup_symptoms(state["symptoms"])
    top_risk = matches[0]["risk_level"] if matches else "unknown"
    # Only treat it as a "critical change" the first time we see a high/critical
    # risk reading (simulates newly-surfaced monitoring data triggering re-plan).
    already_adjusted = state.get("critical_change") is True
    is_critical_now = top_risk in {"critical", "high"} and not already_adjusted
    state["critical_change"] = is_critical_now
    _log(
        state,
        "Any critical change or new data?",
        {"top_risk": top_risk, "critical_change": is_critical_now, "step_index": state["step_index"]},
    )
    return state


def adjust_plan(state: PlanningState) -> PlanningState:
    state["plan"].insert(state["step_index"], "URGENT: Escalate monitoring frequency, alert clinician")
    _log(state, "Adjust Plan Dynamically", {"plan": state["plan"]})
    return state


def proceed_current_plan(state: PlanningState) -> PlanningState:
    _log(state, "Proceed with Current Plan", {"step_index": state["step_index"]})
    return state


def route_after_check(state: PlanningState) -> str:
    if state["step_index"] >= len(state["plan"]) or state["step_index"] >= MAX_MONITORING_STEPS:
        return "complete_or_escalate"
    return "adjust_plan" if state["critical_change"] else "proceed_current_plan"


def route_after_branch(state: PlanningState) -> str:
    if state["step_index"] >= len(state["plan"]) or state["step_index"] >= MAX_MONITORING_STEPS:
        return "complete_or_escalate"
    return "execute_step"


def complete_or_escalate(state: PlanningState) -> PlanningState:
    matches = medical_db_tool.lookup_symptoms(state["symptoms"])
    top_risk = matches[0]["risk_level"] if matches else "unknown"
    escalated = top_risk in {"critical", "high"}
    state["escalated"] = escalated
    state["final_status"] = (
        f"{'Escalated to clinician' if escalated else 'Plan completed'} "
        f"after {state['step_index']} monitored step(s). Final risk read: {top_risk}."
    )
    _log(state, "Complete Plan or Escalate to Clinician", {"final_status": state["final_status"]})
    return state


def report(state: PlanningState) -> PlanningState:
    _log(state, "Report to Patient and Update EHR", {"final_status": state["final_status"]})
    return state


def build_graph():
    graph = StateGraph(PlanningState)
    graph.add_node("set_benchmark", set_benchmark)
    graph.add_node("generate_plan", generate_plan)
    graph.add_node("execute_step", execute_step)
    graph.add_node("check_critical_change", check_critical_change)
    graph.add_node("adjust_plan", adjust_plan)
    graph.add_node("proceed_current_plan", proceed_current_plan)
    graph.add_node("complete_or_escalate", complete_or_escalate)
    graph.add_node("report", report)

    graph.set_entry_point("set_benchmark")
    graph.add_edge("set_benchmark", "generate_plan")
    graph.add_edge("generate_plan", "execute_step")
    graph.add_edge("execute_step", "check_critical_change")
    graph.add_conditional_edges(
        "check_critical_change",
        route_after_check,
        {
            "adjust_plan": "adjust_plan",
            "proceed_current_plan": "proceed_current_plan",
            "complete_or_escalate": "complete_or_escalate",
        },
    )
    graph.add_conditional_edges(
        "adjust_plan", route_after_branch, {"execute_step": "execute_step", "complete_or_escalate": "complete_or_escalate"}
    )
    graph.add_conditional_edges(
        "proceed_current_plan",
        route_after_branch,
        {"execute_step": "execute_step", "complete_or_escalate": "complete_or_escalate"},
    )
    graph.add_edge("complete_or_escalate", "report")
    graph.add_edge("report", END)
    return graph.compile()


_APP = build_graph()


def run(symptoms: List[str], notes: str = "", **_ignored) -> Dict[str, Any]:
    state: PlanningState = {"symptoms": symptoms, "notes": notes, "trace": []}
    result = _APP.invoke(state)
    return {
        "pattern": "Planning",
        "trace": result["trace"],
        "final_output": {
            "escalated": result["escalated"],
            "status": result["final_status"],
            "plan": result["plan"],
        },
    }
