"""
Reflection Pattern
==================
Matches the flow on p.3 of the source document:

  Patient Input/Query
        -> Initial Reasoning and Recommendation
        -> Checking for specific conditions (reflection triggered?)
             -> [yes] Review Past Similar Cases / Model Decisions
                    -> Compare with Medical Guidelines and Recent Outcomes
                    -> Revise Recommendation or Flag for Review
             -> [no]  (skip straight through)
        -> Return Enhanced or Audited Output to User/Clinician

Cognitive Module  -> reviews past encounters to see if similar symptoms
                     produced good outcomes.
Learning Module   -> retains facts from prior cases to improve future
                     triage protocol (here: a slow-response audit log).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from app.json_utils import load_json
from app.llm import get_completion

_PAST_CASES_PATH = Path(__file__).resolve().parent.parent / "mock_data" / "past_cases.json"
_PAST_CASES = load_json(_PAST_CASES_PATH)


class ReflectionState(TypedDict, total=False):
    symptoms: List[str]
    notes: str
    initial_recommendation: str
    reflection_triggered: bool
    trigger_reason: str
    past_case_review: Dict[str, Any]
    guideline_comparison: str
    final_recommendation: str
    trace: List[Dict[str, Any]]


def _log(state: ReflectionState, step: str, detail: Any) -> None:
    if isinstance(detail, str):
        detail = {"value": detail}
    state.setdefault("trace", []).append({"step": step, "detail": detail})


def initial_reasoning(state: ReflectionState) -> ReflectionState:
    prompt = (
        f"Patient reports symptoms: {state['symptoms']}. Notes: {state.get('notes', '')}\n"
        "Give an initial triage reasoning and recommendation."
    )
    state["initial_recommendation"] = get_completion(prompt)
    _log(state, "Initial Reasoning and Recommendation", {"initial_recommendation": state["initial_recommendation"]})
    return state


def check_condition(state: ReflectionState) -> ReflectionState:
    """Checking for specific conditions -> is reflection triggered?"""
    threshold = _PAST_CASES.get("slow_response_threshold", 0.25)
    triggered = False
    reason = "No matching audit entry for these symptoms; no reflection needed."
    log_lookup = _PAST_CASES.get("symptom_case_log", {})
    for symptom in state["symptoms"]:
        entry = log_lookup.get(symptom.strip().lower())
        if entry:
            ratio = entry["flagged_slow_response_cases"] / max(entry["total_cases"], 1)
            if ratio >= threshold:
                triggered = True
                reason = (
                    f"'{symptom}': {entry['flagged_slow_response_cases']}/{entry['total_cases']} "
                    f"past cases flagged for slow response ({ratio:.0%} >= {threshold:.0%} threshold). "
                    f"{entry['note']}"
                )
                break
    state["reflection_triggered"] = triggered
    state["trigger_reason"] = reason
    _log(state, "Checking for specific conditions", {"reflection_triggered": triggered, "reason": reason})
    return state


def review_past_cases(state: ReflectionState) -> ReflectionState:
    review = {
        symptom: _PAST_CASES.get("symptom_case_log", {}).get(symptom.strip().lower())
        for symptom in state["symptoms"]
        if symptom.strip().lower() in _PAST_CASES.get("symptom_case_log", {})
    }
    state["past_case_review"] = review
    _log(state, "Review Past Similar Cases / Model Decisions", review)
    return state


def compare_guidelines(state: ReflectionState) -> ReflectionState:
    prompt = (
        f"Prior-case audit shows: {state['trigger_reason']}\n"
        f"Initial recommendation was: {state['initial_recommendation']}\n"
        "Compare against best-practice triage guidance and note what should change "
        "to avoid a repeat of the slow-response pattern."
    )
    state["guideline_comparison"] = get_completion(prompt)
    _log(state, "Compare with Medical Guidelines and Recent Outcomes", {"guideline_comparison": state["guideline_comparison"]})
    return state


def revise_recommendation(state: ReflectionState) -> ReflectionState:
    state["final_recommendation"] = (
        f"[REVISED after reflection] Prioritize this case faster than the default queue. "
        f"{state['guideline_comparison']}"
    )
    _log(state, "Revise Recommendation or Flag for Review", {"final_recommendation": state["final_recommendation"]})
    return state


def passthrough(state: ReflectionState) -> ReflectionState:
    state["final_recommendation"] = state["initial_recommendation"]
    _log(state, "No reflection needed - passthrough", {"final_recommendation": state["final_recommendation"]})
    return state


def route_reflection(state: ReflectionState) -> str:
    return "review_past_cases" if state["reflection_triggered"] else "passthrough"


def build_graph():
    graph = StateGraph(ReflectionState)
    graph.add_node("initial_reasoning", initial_reasoning)
    graph.add_node("check_condition", check_condition)
    graph.add_node("review_past_cases", review_past_cases)
    graph.add_node("compare_guidelines", compare_guidelines)
    graph.add_node("revise_recommendation", revise_recommendation)
    graph.add_node("passthrough", passthrough)

    graph.set_entry_point("initial_reasoning")
    graph.add_edge("initial_reasoning", "check_condition")
    graph.add_conditional_edges(
        "check_condition",
        route_reflection,
        {"review_past_cases": "review_past_cases", "passthrough": "passthrough"},
    )
    graph.add_edge("review_past_cases", "compare_guidelines")
    graph.add_edge("compare_guidelines", "revise_recommendation")
    graph.add_edge("revise_recommendation", END)
    graph.add_edge("passthrough", END)
    return graph.compile()


_APP = build_graph()


def run(symptoms: List[str], notes: str = "", **_ignored) -> Dict[str, Any]:
    state: ReflectionState = {"symptoms": symptoms, "notes": notes, "trace": []}
    result = _APP.invoke(state)
    return {
        "pattern": "Reflection",
        "trace": result["trace"],
        "final_output": {
            "reflection_triggered": result["reflection_triggered"],
            "recommendation": result["final_recommendation"],
        },
    }
