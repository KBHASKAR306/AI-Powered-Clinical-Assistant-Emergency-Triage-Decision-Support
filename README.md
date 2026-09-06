# AI-Powered Clinical Assistant — Emergency Triage & Decision Support

Working implementation of **Implementing different agentic AI design
patterns** (Agentic AI Architectures and Design Patterns). Each of
the six patterns from the source document is built as its own state graph, matching
the flowcharts in the document step-for-step, and exposed through a FastAPI
service plus a Streamlit UI for interactive testing.

> ⚠️ **Educational / portfolio demo only.** All patient records, guidelines,
> and ontology mappings are synthetic (`app/mock_data/*.json`). This is **not
> a medical device**, has not been validated clinically, and must never be
> used for real triage, diagnosis, or treatment decisions.

## Patterns implemented

| Pattern | File | Matches document flow |
|---|---|---|
| Reflection | `app/patterns/reflection.py` | initial reasoning → check condition → review past cases → compare guidelines → revise |
| Tool Use | `app/patterns/tool_use.py` | intent parsing → identify tool → call external tool → significance check → alert/log |
| Planning | `app/patterns/planning.py` | set benchmark → generate plan → execute with monitoring → adjust/proceed loop → complete/escalate |
| ReAct | `app/patterns/react.py` | Think (causes) → Act (query EHR) → Think (differential) → Act (order tests/alert) → recommendation |
| ReWOO | `app/patterns/rewoo.py` | extract entities → map to ontology → reason over ontology → trigger tools → explainable output |
| Multi-Agent | `app/patterns/multi_agent.py` | broadcast → Triage Agent ∥ Admin Agent (real LangGraph fan-out/fan-in) → critical check → escalate/coordinate → aggregate |

Each pattern module exposes a single `run(symptoms, patient_id="", notes="")`
function returning:

```json
{
  "pattern": "...",
  "trace": [{"step": "...", "detail": "..."}, ...],
  "final_output": {...}
}
```

`trace` is the ordered list of flowchart boxes the run passed through — this
is what the API/UI display so you can see each pattern actually branching
the way the document describes (e.g. Tool Use alerting vs. logging routine
advice, Planning looping through monitored steps, Multi-Agent's two agents
genuinely running concurrently before the coordinator aggregates).

## How the patterns combine

The document's architecture section maps each pattern onto shared modules
(Cognitive, Action, Perception, Learning, Collaboration). `POST
/pipeline/full` chains all six in the order a real deployment would likely
use them for one incoming complaint:

1. **ReWOO** — structure the raw complaint into ontology-linked concepts
2. **ReAct** — interleaved reasoning + action to reach a working diagnosis
3. **Tool Use** — confirm findings against the decision-support tool
4. **Planning** — lay out the multi-step care plan
5. **Multi-Agent** — fan out to Triage/Admin agents, coordinate the response
6. **Reflection** — check the outcome against the prior-case audit log and
   revise if this symptom pattern has a history of slow response

## Project layout

```
clinical_assistant_demo/
├── app/
│   ├── main.py            # FastAPI app (all endpoints)
│   ├── llm.py              # Claude wrapper + offline heuristic fallback
│   ├── schemas.py          # Pydantic request/response models
│   ├── mock_data/          # synthetic EHR, guidelines, ontology, case log
│   ├── tools/               # mock EHR / medical-DB / ontology "Action Module" tools
│   └── patterns/            # the six LangGraph implementations
├── streamlit_app.py        # interactive test UI
├── tests/test_patterns.py  # 17 offline pytest cases (all branches)
├── requirements.txt
├── Dockerfile / docker-compose.yml
└── .env.example
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional — see below
```

`ANTHROPIC_API_KEY` is **optional**. If unset, every reasoning step in every
pattern falls back to a small deterministic keyword-based heuristic
(`app/llm.py::_heuristic`), so the entire system — API, UI, and test suite —
runs fully offline. Set the key in `.env` (or export it) to have the
Cognitive-Module reasoning steps call real Claude instead.

## Running it

**API server:**
```bash
uvicorn app.main:app --reload --port 8000
# then open http://127.0.0.1:8000/docs for interactive Swagger UI
```

**Streamlit test UI** (talks to the pattern modules directly, no server needed):
```bash
streamlit run streamlit_app.py
```

**Example requests:**
```bash
# List synthetic patients
curl http://127.0.0.1:8000/patients

# Run a single pattern
curl -X POST http://127.0.0.1:8000/patterns/react \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "P-1001", "symptoms": ["chest discomfort", "shortness of breath"]}'

# Run the full six-pattern pipeline
curl -X POST http://127.0.0.1:8000/pipeline/full \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "P-1003", "symptoms": ["palpitations", "dizziness"]}'
```

Valid `pattern_name` values for `POST /patterns/{pattern_name}`:
`reflection`, `tool-use`, `planning`, `react`, `rewoo`, `multi-agent`.

Synthetic patient IDs available out of the box: `P-1001`, `P-1002`, `P-1003`
(see `app/mock_data/ehr_records.json` — feel free to add more).

## Testing

```bash
pytest tests/ -v
```

17 tests cover both branches of every conditional in every pattern (e.g.
Reflection triggered vs. passthrough, Tool Use alert vs. routine log,
Planning escalated vs. completed, Multi-Agent escalation vs. standard
follow-up) plus all FastAPI endpoints, including 404s. No API key or network
access required — everything runs against the offline heuristic and the
mock data files.

## Deployment

**Docker:**
```bash
docker build -t clinical-assistant-demo .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-... clinical-assistant-demo
```

**Docker Compose:**
```bash
ANTHROPIC_API_KEY=sk-... docker compose up --build
```

Either way, the API is then live at `http://localhost:8000` (Swagger docs at
`/docs`). The image only bakes in the FastAPI service; run
`streamlit run streamlit_app.py` locally (or in a second container) for the
interactive UI.

## Extending this demo

- **Real data sources**: swap `app/tools/ehr_tool.py`,
  `medical_db_tool.py`, and `ontology_tool.py` for real EHR/FHIR, clinical
  decision-support, and SNOMED-CT/UMLS integrations — the pattern graphs
  don't need to change, only the tool implementations.
- **Real reasoning**: set `ANTHROPIC_API_KEY` to replace the offline
  heuristic with actual Claude calls in every Cognitive-Module step.
- **More patients/conditions**: add entries to the JSON files under
  `app/mock_data/` — no code changes needed.
- **Persistence / streaming**: LangGraph supports checkpointing and
  streaming out of the box; `graph.compile(checkpointer=...)` in any
  `patterns/*.py` file is the natural next step for a stateful, resumable
  multi-turn assistant.
