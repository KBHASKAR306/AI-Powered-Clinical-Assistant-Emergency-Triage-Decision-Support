"""
ReWOO (Reasoning with Open Ontology) Pattern
=============================================
Matches the flow on p.8 of the source document:

  Patient Input/Query
        -> Extract Entities (symptoms, duration, severity)
        -> Map to Clinical Ontology (e.g. SNOMED CT / UMLS concepts)
        -> Reasoning Using Ontology (Symptoms -> Conditions -> Risk Scores -> Possible Tests)
        -> Use Ontology to Trigger Plans or Tools (decision support, triage workflows, drug checkers)
        -> Explainable Output (with linked concepts and traceable reasoning paths)

Learning Module   -> keeps the ontology's "recent_updates" current.
Cognitive Module  -> reasons over the (possibly just-updated) ontology.
"""
from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from app.tools import medical_db_tool, ontology_tool


class ReWOOState(TypedDict, total=False):
    symptoms: List[str]
    notes: str
    entities: Dict[str, Any]
    ontology_concepts: List[Dict[str, str]]
    candidate_conditions: List[str]
    risk_assessment: List[Dict[str, Any]]
    triggered_tools: List[str]
    explainable_output: Dict[str, Any]
    trace: List[Dict[str, Any]]


def _log(state: ReWOOState, step: str, detail: Any) -> None:
    if isinstance(detail, str):
        detail = {"value": detail}
    state.setdefault("trace", []).append({"step": step, "detail": detail})


def extract_entities(state: ReWOOState) -> ReWOOState:
    entities = {
        "symptoms": state["symptoms"],
        "duration": "unspecified" if "duration" not in state.get("notes", "").lower() else state["notes"],
        "severity": "unspecified",
    }
    state["entities"] = entities
    _log(state, "Extract Entities (symptoms, duration, severity)", entities)
    return state


def map_to_ontology(state: ReWOOState) -> ReWOOState:
    concepts = ontology_tool.map_to_ontology(state["symptoms"])
    state["ontology_concepts"] = concepts
    _log(state, "Map to Clinical Ontology (SNOMED CT / UMLS concepts)", concepts)
    return state


def reasoning_using_ontology(state: ReWOOState) -> ReWOOState:
    candidates = ontology_tool.candidate_conditions(state["ontology_concepts"])
    state["candidate_conditions"] = candidates
    guideline_matches = medical_db_tool.lookup_symptoms(state["symptoms"])
    state["risk_assessment"] = [
        {"condition": m["condition"], "risk_level": m["risk_level"], "possible_tests": m["recommended_tests"]}
        for m in guideline_matches
    ]
    recent = ontology_tool.recent_updates()
    _log(
        state,
        "Reasoning Using Ontology (Symptoms -> Conditions -> Risk Scores -> Possible Tests)",
        {
            "candidate_conditions": candidates,
            "risk_assessment": state["risk_assessment"],
            "ontology_recent_updates_considered": recent,
        },
    )
    return state


def trigger_plans_or_tools(state: ReWOOState) -> ReWOOState:
    triggered = []
    if state["risk_assessment"]:
        top = state["risk_assessment"][0]
        triggered.append("clinical_decision_support")
        if top["risk_level"] in {"critical", "high"}:
            triggered.append("triage_escalation_workflow")
        if top["possible_tests"]:
            triggered.append("drug_interaction_checker")
    else:
        triggered.append("routine_advice_workflow")
    state["triggered_tools"] = triggered
    _log(
        state,
        "Use Ontology to Trigger Plans or Tools (decision support, triage workflows, drug checkers)",
        triggered,
    )
    return state


def explainable_output(state: ReWOOState) -> ReWOOState:
    output = {
        "linked_concepts": state["ontology_concepts"],
        "reasoning_path": [
            f"{c['symptom']} -> {c['label']} ({c['concept_id']})" for c in state["ontology_concepts"]
        ],
        "candidate_conditions": state["candidate_conditions"],
        "risk_assessment": state["risk_assessment"],
        "triggered_tools": state["triggered_tools"],
    }
    state["explainable_output"] = output
    _log(state, "Explainable Output (linked concepts + traceable reasoning path)", output)
    return state


def build_graph():
    graph = StateGraph(ReWOOState)
    graph.add_node("extract_entities", extract_entities)
    graph.add_node("map_to_ontology", map_to_ontology)
    graph.add_node("reasoning_using_ontology", reasoning_using_ontology)
    graph.add_node("trigger_plans_or_tools", trigger_plans_or_tools)
    graph.add_node("explainable_output", explainable_output)

    graph.set_entry_point("extract_entities")
    graph.add_edge("extract_entities", "map_to_ontology")
    graph.add_edge("map_to_ontology", "reasoning_using_ontology")
    graph.add_edge("reasoning_using_ontology", "trigger_plans_or_tools")
    graph.add_edge("trigger_plans_or_tools", "explainable_output")
    graph.add_edge("explainable_output", END)
    return graph.compile()


_APP = build_graph()


def run(symptoms: List[str], notes: str = "", **_ignored) -> Dict[str, Any]:
    state: ReWOOState = {"symptoms": symptoms, "notes": notes, "trace": []}
    result = _APP.invoke(state)
    return {
        "pattern": "ReWOO",
        "trace": result["trace"],
        "final_output": result["explainable_output"],
    }
