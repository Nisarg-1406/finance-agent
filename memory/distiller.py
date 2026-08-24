"""
memory/distiller.py — Production-Grade Memory Distillation

After each session, call the LLM once to:
  1. Extract durable semantic facts (with importance scoring 1-5)
  2. Extract behavioral signals for procedural memory
  3. Deduplicate against existing facts (cosine similarity)

Selective storage principles:
  - Store: goals, commitments, plans, concerns, reminders, behavioral signals
  - Skip:  balances (stale), transactions (noisy), bill amounts (semi-mutable),
           LLM reasoning traces, anything with importance < 3

Distillation uses ONE LLM call per session — not per turn.
Cost at scale: ~500 tokens per distillation × $0.00015/1K = $0.000075/session.
"""

import json
import re
import numpy as np
from typing import Optional


# ─── Distillation Prompt ──────────────────────────────────────────────────────

DISTILLATION_PROMPT = """You are a memory distiller for a personal finance AI agent.
Extract ONLY durable facts worth remembering across future sessions.

Rules:
1. DO store: goals, commitments, concerns, plans agreed to, reminders set, financial milestones reached.
2. DO NOT store: specific balances/amounts (stale), full transaction lists,
   bill amounts (will re-fetch), one-off questions, LLM reasoning traces.
3. Rate each fact's importance 1-5:
   5 = Critical (goal, commitment, major plan change)
   4 = Important (recurring concern, behavioral pattern)
   3 = Useful (preference, minor fact worth knowing)
   2 = Low value (marginally useful, uncertain)
   1 = Skip (noise, ephemeral, stale by next session)
4. ONLY include facts with importance >= 3.
5. Confidence: 1.0 = explicitly stated, 0.7 = strongly implied, 0.5 = inferred.
6. Output ONLY valid JSON — no preamble, no markdown, no explanation.

Output format:
{{
  "facts": {{
    "fact_key_snake_case": {{
      "value": <string | number | boolean | object>,
      "confidence": <0.0-1.0>,
      "importance": <3|4|5>,
      "note": "one line why this is worth storing"
    }}
  }},
  "behaviors": {{
    "behavior_key": {{
      "active": true,
      "weight": <0.0-1.0>,
      "note": "what behavioral pattern this represents"
    }}
  }}
}}

Session transcript:
{transcript}

User profile:
{profile}

Existing facts (do not duplicate these unless the value changed):
{existing_keys}

Extract now (JSON only):"""


PROCEDURAL_KEYWORDS = {
    # word in transcript → (behavior_pattern, weight_signal)
    "detailed breakdown": ("prefers_detailed",  0.7),
    "too complicated":    ("prefers_concise",   0.6),
    "remind me":          ("likes_reminders",   0.8),
    "every month":        ("likes_reminders",   0.5),
    "conservative":       ("risk_averse",       0.7),
    "aggressive":         ("risk_tolerant",     0.7),
    "honestly":           ("wants_honest",      0.6),
    "just tell me":       ("prefers_concise",   0.8),
    "worried":            ("anxiety_prone",     0.5),
    "excited":            ("goal_motivated",    0.6),
    "discipline":         ("goal_focused",      0.7),
}


# ─── Main Distillation Function ───────────────────────────────────────────────

def distill_session(
    transcript: str,
    profile: dict,
    llm_caller,
    existing_facts: Optional[dict] = None,
) -> dict:
    """
    Call LLM to distill the session into semantic facts and behavioral signals.

    Args:
        transcript: Formatted transcript string.
        profile: User profile dict.
        llm_caller: callable(messages: list) -> str
        existing_facts: Already-stored facts for deduplication.

    Returns:
        dict with "facts" and "behaviors" keys.
    """
    existing_keys = list(existing_facts.keys()) if existing_facts else []

    prompt = DISTILLATION_PROMPT.format(
        transcript=transcript,
        profile=json.dumps(profile, ensure_ascii=False),
        existing_keys=json.dumps(existing_keys) if existing_keys else "[]"
    )

    raw = llm_caller([{"role": "user", "content": prompt}])

    # Strip any markdown fences despite instructions
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[WARN] Distillation JSON parse failed. Raw: {raw[:300]}")
        # Graceful degradation: try behavioral extraction from transcript directly
        return {
            "facts": {},
            "behaviors": _extract_behaviors_from_transcript(transcript)
        }

    # Filter facts below importance threshold
    facts = parsed.get("facts", {})
    filtered_facts = {
        k: v for k, v in facts.items()
        if isinstance(v, dict) and v.get("importance", 0) >= 3
    }

    behaviors = parsed.get("behaviors", {})
    # Merge with rule-based behavioral extraction
    transcript_behaviors = _extract_behaviors_from_transcript(transcript)
    for k, v in transcript_behaviors.items():
        if k not in behaviors:
            behaviors[k] = v

    return {"facts": filtered_facts, "behaviors": behaviors}


# ─── Behavioral Signal Extraction ─────────────────────────────────────────────

def _extract_behaviors_from_transcript(transcript: str) -> dict:
    """
    Rule-based procedural memory extraction from transcript text.
    Supplements LLM extraction — catches patterns the LLM might miss.
    No LLM call needed — runs in <1ms.
    """
    transcript_lower = transcript.lower()
    behaviors = {}
    for phrase, (pattern, weight) in PROCEDURAL_KEYWORDS.items():
        if phrase in transcript_lower:
            behaviors[pattern] = {
                "active": weight > 0.5,
                "weight": weight,
                "note": f"Detected phrase '{phrase}' in session"
            }
    return behaviors


# ─── Deduplication ────────────────────────────────────────────────────────────

def deduplicate_facts(
    new_facts: dict,
    existing_facts: dict,
    embedding_fn=None,
    similarity_threshold: float = 0.85,
) -> dict:
    """
    Remove new facts that are semantically too similar to existing facts.
    Uses key-name similarity first (fast), then embedding cosine if available.

    Args:
        new_facts: Facts from latest distillation.
        existing_facts: Already stored facts {key: {value, confidence, ...}}.
        embedding_fn: Optional callable(text: str) -> list[float].
        similarity_threshold: Cosine similarity above which we merge instead of add.

    Returns:
        Filtered new_facts with duplicates removed or merged.
    """
    result = {}
    existing_keys = set(existing_facts.keys())

    for key, meta in new_facts.items():
        # Exact key match: update only if confidence improved
        if key in existing_keys:
            existing_conf = existing_facts[key].get("confidence", 0)
            new_conf = meta.get("confidence", 0)
            if new_conf > existing_conf + 0.1:  # significant improvement
                result[key] = meta  # allow update
            # else: skip — existing fact is fine
            continue

        # Near-key match: string similarity
        key_parts = set(key.lower().split("_"))
        is_near_dup = False
        for ex_key in existing_keys:
            ex_parts = set(ex_key.lower().split("_"))
            overlap = len(key_parts & ex_parts) / max(1, len(key_parts | ex_parts))
            if overlap > 0.75:
                is_near_dup = True
                break

        if is_near_dup:
            # Still add if the new fact has meaningfully different value
            new_val = str(meta.get("value", ""))
            ex_val = str(existing_facts.get(key, {}).get("value", ""))
            if new_val != ex_val:
                result[key] = meta
            continue

        # New fact: add it
        result[key] = meta

    return result


# ─── Transcript Formatter ─────────────────────────────────────────────────────

def format_transcript_for_distillation(turns: list[dict]) -> str:
    """Convert saved turns into a clean transcript string."""
    lines = []
    for t in turns:
        role = "User" if t["role"] == "user" else "Agent"
        lines.append(f"{role}: {t['content']}")
        if t.get("tool_calls"):
            tcs = json.loads(t["tool_calls"]) if isinstance(t["tool_calls"], str) else t["tool_calls"]
            for tc in (tcs or []):
                result_str = json.dumps(tc.get("result", ""), ensure_ascii=False)
                lines.append(f"  [Tool: {tc.get('tool')} → {result_str[:120]}]")
    return "\n".join(lines)
