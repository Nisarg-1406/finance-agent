"""
memory/context_manager.py — Sliding Window + Conversation Summarization

Handles long-session context management for production-scale agents.

Problem: After session 4+, conversation history grows beyond the token budget.
Naive approach: dump all turns → context distraction + token overflow.
This module implements:

  1. Sliding Window: Keep last N turns verbatim (recent, precise context)
  2. LLM Summarization: Compress older turns into a compact summary block
  3. Token Budget Enforcement: Estimate tokens, truncate if over budget
  4. Context Assembly: Combine memory facts + recent history + summary

Production note (from guide §1):
  "Optimal context = recent precise turns + compressed older history
   + relevant retrieved facts."

At scale: Store turn summaries in Redis (TTL=session duration), so
  summarization is only called once per summary window, not on every turn.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

# ─── Config ───────────────────────────────────────────────────────────────────

SLIDING_WINDOW_TURNS = 8        # Keep last N full turns in context
TOKEN_BUDGET = 6000             # Max tokens for context injection
CHARS_PER_TOKEN = 4             # Rough estimate (no tiktoken dependency)
SUMMARY_MAX_TOKENS = 300        # Target length for compressed older turns


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class ContextWindow:
    """Assembled context for a single agent turn."""
    recent_history: list[dict]     # Last N turns (OpenAI message format)
    summary_block: str             # Compressed summary of older turns ("" if none)
    memory_facts: str              # Distilled facts from prior sessions
    behaviors: str                 # Learned behavioral preferences
    total_estimated_tokens: int
    turns_summarized: int          # How many turns were compressed
    turns_in_window: int           # How many turns are verbatim


# ─── Token Estimation ─────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """
    Rough token estimate without tiktoken. Good enough for budget enforcement.
    Production: replace with tiktoken.encoding_for_model('gpt-4o-mini').encode()
    """
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_message_tokens(messages: list[dict]) -> int:
    """Estimate total tokens in a messages list."""
    total = 0
    for m in messages:
        content = m.get("content", "")
        total += estimate_tokens(content) + 4  # role + formatting overhead
    return total


# ─── Summarization ────────────────────────────────────────────────────────────

SUMMARIZATION_PROMPT = """Summarize the following conversation turns into a compact memory block (under 200 words).
Focus ONLY on: decisions made, commitments, concerns raised, and key numbers mentioned.
Do NOT include tool call details or pleasantries.
Write in third-person past tense. Be dense and factual.

Conversation turns to summarize:
{turns_text}

Compact summary (under 200 words):"""


def format_turns_for_summary(turns: list[dict]) -> str:
    """Format conversation turns as readable text for summarization."""
    lines = []
    for msg in turns:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            # Truncate very long assistant responses for the summary prompt
            truncated = content[:500] + "..." if len(content) > 500 else content
            lines.append(f"Agent: {truncated}")
        # Skip tool observation messages (role=user but contains "[tool result]")
    return "\n".join(lines)


def summarize_older_turns(turns: list[dict], llm_caller) -> str:
    """
    Compress older turns into a short summary.

    Args:
        turns: List of older messages (OpenAI format) to compress.
        llm_caller: Synchronous callable (messages: list) -> str

    Returns:
        Compact summary string, or empty string if turns is empty.
    """
    if not turns:
        return ""

    turns_text = format_turns_for_summary(turns)
    if not turns_text.strip():
        return ""

    prompt = SUMMARIZATION_PROMPT.format(turns_text=turns_text)
    try:
        summary = llm_caller([{"role": "user", "content": prompt}])
        # Strip any accidental markdown
        summary = re.sub(r"```.*?```", "", summary, flags=re.DOTALL).strip()
        return summary
    except Exception as e:
        # Graceful degradation: fall back to first 200 chars of turns
        return turns_text[:800] + "..." if len(turns_text) > 800 else turns_text


# ─── Main Context Builder ─────────────────────────────────────────────────────

def build_context_window(
    history: list[dict],
    memory_facts: str,
    behaviors: str,
    llm_caller=None,
    window_size: int = SLIDING_WINDOW_TURNS,
    token_budget: int = TOKEN_BUDGET,
) -> ContextWindow:
    """
    Build the context window for the next agent turn.

    Strategy:
      1. If history <= window_size turns: use all turns verbatim
      2. If history > window_size: summarize older turns, keep last N verbatim
      3. Enforce token budget — trim summary if over budget

    Args:
        history: Full conversation history (OpenAI message format).
        memory_facts: Pre-built string from build_memory_context().
        behaviors: Pre-built string of behavioral preferences.
        llm_caller: Callable for summarization (None = skip summarization).
        window_size: Number of recent turns to keep verbatim.
        token_budget: Max estimated tokens for the combined context.

    Returns:
        ContextWindow with all assembled pieces.
    """
    # Count actual turns (user + assistant pairs, not tool observations)
    convo_turns = [
        m for m in history
        if m.get("role") in ("user", "assistant")
        and not m.get("content", "").startswith("[")  # exclude tool observation messages
    ]

    summary_block = ""
    turns_summarized = 0

    if len(convo_turns) > window_size * 2:  # window_size pairs
        # Split: older turns get summarized, recent turns stay verbatim
        split_point = max(0, len(convo_turns) - window_size * 2)
        older_turns = convo_turns[:split_point]
        recent_turns = convo_turns[split_point:]

        turns_summarized = len(older_turns)

        # Only summarize if we have an LLM caller
        if llm_caller and older_turns:
            summary_block = summarize_older_turns(older_turns, llm_caller)

        # The recent_turns in window are our verbatim context
        # Map back to full history format to preserve tool observations
        recent_history = _get_recent_history_slice(history, recent_turns)
    else:
        recent_history = history
        recent_turns = convo_turns

    # Token budget enforcement
    fact_tokens = estimate_tokens(memory_facts)
    behav_tokens = estimate_tokens(behaviors)
    summary_tokens = estimate_tokens(summary_block)
    history_tokens = estimate_message_tokens(recent_history)

    total_tokens = fact_tokens + behav_tokens + summary_tokens + history_tokens

    # If over budget, trim summary first, then trim history window further
    if total_tokens > token_budget and summary_block:
        # Truncate summary
        target_summary_chars = max(100, (token_budget - fact_tokens - behav_tokens - history_tokens) * CHARS_PER_TOKEN)
        if len(summary_block) > target_summary_chars:
            summary_block = summary_block[:target_summary_chars] + "..."
        summary_tokens = estimate_tokens(summary_block)

    total_tokens = fact_tokens + behav_tokens + summary_tokens + history_tokens

    return ContextWindow(
        recent_history=recent_history,
        summary_block=summary_block,
        memory_facts=memory_facts,
        behaviors=behaviors,
        total_estimated_tokens=total_tokens,
        turns_summarized=turns_summarized,
        turns_in_window=len(recent_turns),
    )


def _get_recent_history_slice(full_history: list[dict], recent_convo: list[dict]) -> list[dict]:
    """
    Given the recent conversation turns, find and return the corresponding
    slice of the full history (which includes tool observation messages).
    """
    if not recent_convo:
        return []
    # Find the first recent message in the full history
    first_recent_content = recent_convo[0].get("content", "")
    for i, msg in enumerate(full_history):
        if msg.get("content") == first_recent_content:
            return full_history[i:]
    # Fallback: return last portion
    return full_history[-len(recent_convo) * 2:]


# ─── Context Injection String Builder ────────────────────────────────────────

def build_context_injection(window: ContextWindow, profile_name: str) -> str:
    """
    Build the injectable context string block for the system prompt.
    This is the DYNAMIC portion (not KV-cached).

    Structure:
      [Memory facts from prior sessions]
      [Behavioral preferences]
      [Summary of earlier turns in this session (if any)]
    """
    parts = []

    if window.memory_facts:
        parts.append(window.memory_facts)

    if window.behaviors:
        parts.append(f"\n## {profile_name}'s preferences (learned):\n{window.behaviors}")

    if window.summary_block:
        parts.append(
            f"\n## Earlier in this session (summary of {window.turns_summarized} turns):\n"
            f"{window.summary_block}"
        )

    if not parts:
        return ""

    return "\n".join(parts) + "\n---\n"


# ─── History Pruning for Token Budget ────────────────────────────────────────

def prune_tool_observations(history: list[dict], keep_last_n: int = 3) -> list[dict]:
    """
    Remove all but the last N tool observation messages from history.
    Tool observations are assistant-injected messages with "[tool_name result]:" prefix.
    This reduces token usage without losing recent tool context.

    Production use: Call before building context window on each turn.
    """
    tool_obs_indices = [
        i for i, m in enumerate(history)
        if m.get("role") == "user" and m.get("content", "").startswith("[")
    ]

    # Keep only the last keep_last_n tool observation messages
    to_remove = set(tool_obs_indices[:-keep_last_n]) if len(tool_obs_indices) > keep_last_n else set()
    return [m for i, m in enumerate(history) if i not in to_remove]
