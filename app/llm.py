"""
Thin LLM wrapper used by every pattern's "reasoning" nodes.

- If ANTHROPIC_API_KEY is set in the environment, calls the real Claude API.
- Otherwise falls back to a small deterministic heuristic responder so the
  full pipeline can be exercised offline (e.g. in CI, or by a reviewer who
  hasn't wired up a key yet).

This is a teaching/demo project. Nothing here is a real medical device and
none of the "reasoning" should be treated as clinical advice.
"""
from __future__ import annotations

import json
import os
from typing import Optional

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

CRITICAL_KEYWORDS = {
    "chest pain", "chest discomfort", "shortness of breath", "difficulty in breathing",
    "cold sweat", "left arm pain", "fainting", "severe bleeding", "unresponsive",
}
MODERATE_KEYWORDS = {
    "wheezing", "palpitations", "dizziness", "chest tightness", "irregular heartbeat",
}


def get_completion(prompt: str, system: Optional[str] = None) -> str:
    """Return a JSON-serialized completion for `prompt`, with a plain-text body under `text`."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model=DEFAULT_MODEL,
                max_tokens=600,
                system=system or "",
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                block.text for block in resp.content if getattr(block, "type", "") == "text"
            ).strip()
            return json.dumps({"text": text}, ensure_ascii=False)
        except Exception as exc:  # pragma: no cover - network/credential errors
            fallback = _heuristic_payload(prompt)
            fallback["text"] = f"[LLM call failed ({exc}); using offline heuristic]\n{fallback['text']}"
            return json.dumps(fallback, ensure_ascii=False)
    return _heuristic(prompt)


def _heuristic_payload(prompt: str) -> dict:
    text = prompt.lower()
    hits_critical = [k for k in CRITICAL_KEYWORDS if k in text]
    hits_moderate = [k for k in MODERATE_KEYWORDS if k in text]

    if hits_critical:
        return {
            "text": (
                "Reasoning: symptom pattern overlaps strongly with a time-critical "
                f"presentation ({', '.join(sorted(hits_critical))}). "
                "Risk level: CRITICAL. Recommend immediate escalation."
            ),
            "risk_level": "CRITICAL",
            "keywords": sorted(hits_critical),
        }
    if hits_moderate:
        return {
            "text": (
                "Reasoning: symptom pattern suggests a condition needing prompt but "
                f"non-emergency follow-up ({', '.join(sorted(hits_moderate))}). "
                "Risk level: MODERATE. Recommend monitoring and clinician review."
            ),
            "risk_level": "MODERATE",
            "keywords": sorted(hits_moderate),
        }
    return {
        "text": (
            "Reasoning: no high-risk keywords detected in the reported symptoms. "
            "Risk level: LOW. Recommend routine, symptomatic care advice."
        ),
        "risk_level": "LOW",
        "keywords": [],
    }


def _heuristic(prompt: str) -> str:
    """
    Deterministic, keyword-based stand-in for an LLM call. Good enough to
    exercise every branch of every LangGraph pattern without any API key or
    network access — useful for local testing and CI.

    Returns a JSON string so downstream code and clients always receive valid
    JSON, never a bare text blob that can trigger JSON parse errors.
    """
    payload = _heuristic_payload(prompt)
    return json.dumps(payload, ensure_ascii=False)
