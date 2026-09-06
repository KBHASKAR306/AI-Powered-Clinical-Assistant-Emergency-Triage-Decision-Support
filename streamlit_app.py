"""
Streamlit test UI for the Clinical Assistant agentic-pattern demo.

Run with:  streamlit run streamlit_app.py

Calls the pattern modules directly (no need to have the FastAPI server
running) so it's a fast way to click through Reflection / Tool Use /
Planning / ReAct / ReWOO / Multi-Agent and see each flowchart step fire.
"""
from __future__ import annotations

import streamlit as st

from app.patterns import multi_agent, planning, react, reflection, rewoo, tool_use
from app.tools import ehr_tool

st.set_page_config(page_title="Clinical Assistant — Agentic Patterns Demo", layout="wide")

st.title("AI-Powered Clinical Assistant — Emergency Triage & Decision Support")
st.caption(
    "Educational demo of Agentic AI design patterns (Reflection, Tool Use, Planning, "
    "ReAct, ReWOO, Multi-Agent). All patient data is synthetic. **Not a medical device.**"
)

PATTERNS = {
    "Reflection": reflection.run,
    "Tool Use": tool_use.run,
    "Planning": planning.run,
    "ReAct": react.run,
    "ReWOO": rewoo.run,
    "Multi-Agent": multi_agent.run,
}

with st.sidebar:
    st.header("Patient input")
    patient_id = st.selectbox("Synthetic patient (optional)", [""] + ehr_tool.list_patient_ids())
    symptoms_raw = st.text_area(
        "Symptoms (one per line)",
        value="chest discomfort\nshortness of breath",
        height=120,
    )
    notes = st.text_input("Additional notes (optional)", value="")
    symptoms = [s.strip() for s in symptoms_raw.splitlines() if s.strip()]

    if patient_id:
        with st.expander("Synthetic EHR record"):
            st.json(ehr_tool.query_ehr(patient_id))

tabs = st.tabs(list(PATTERNS.keys()) + ["Full Pipeline"])

for tab, (name, fn) in zip(tabs[:-1], PATTERNS.items()):
    with tab:
        st.subheader(f"{name} Pattern")
        if st.button(f"Run {name}", key=f"run_{name}"):
            if not symptoms:
                st.warning("Enter at least one symptom in the sidebar.")
            else:
                result = fn(symptoms=symptoms, patient_id=patient_id, notes=notes)
                st.markdown("**Flow trace**")
                for i, step in enumerate(result["trace"], start=1):
                    with st.expander(f"{i}. {step['step']}", expanded=(i == len(result["trace"]))):
                        st.json(step["detail"])
                st.markdown("**Final output**")
                st.json(result["final_output"])

with tabs[-1]:
    st.subheader("Full Pipeline (all six patterns chained)")
    st.caption(
        "ReWOO → ReAct → Tool Use → Planning → Multi-Agent → Reflection — "
        "see README 'How the patterns combine' for why this order."
    )
    if st.button("Run full pipeline"):
        if not symptoms:
            st.warning("Enter at least one symptom in the sidebar.")
        else:
            ordered = [
                ("1. ReWOO", rewoo.run),
                ("2. ReAct", react.run),
                ("3. Tool Use", tool_use.run),
                ("4. Planning", planning.run),
                ("5. Multi-Agent", multi_agent.run),
                ("6. Reflection", reflection.run),
            ]
            final_recommendation = None
            for label, fn in ordered:
                result = fn(symptoms=symptoms, patient_id=patient_id, notes=notes)
                with st.expander(label, expanded=False):
                    for i, step in enumerate(result["trace"], start=1):
                        st.markdown(f"**{i}. {step['step']}**")
                        st.json(step["detail"])
                    st.markdown("_Final output:_")
                    st.json(result["final_output"])
                if label.startswith("6"):
                    final_recommendation = result["final_output"]["recommendation"]
            st.success(f"Final recommendation after reflection: {final_recommendation}")
