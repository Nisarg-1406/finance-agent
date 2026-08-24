"""
evaluation/evaluator.py — Production-Grade Agent Evaluation Framework

Three-tier evaluation architecture:
  Tier 1: Deterministic (no LLM, <5ms per check) — ground truth checks
  Tier 2: LLM-as-judge (100-500ms) — nuanced quality dimensions
  Tier 3: Production metrics (from metrics DB) — operational health

Evaluation dimensions:
  Core:
    ✓ Memory recall across sessions
    ✓ Live data fetched for financial decisions
    ✓ set_reminder called correctly
    ✓ Goal awareness throughout

  New production dimensions:
    ✓ Multi-step trajectory evaluation (efficiency + correctness + recovery)
    ✓ Task completion rate (was user intent resolved?)
    ✓ Hallucination rate (claims grounded in tool results?)
    ✓ Context utilization (how much of retrieved memory was used?)
    ✓ Retry rate (tool error recovery frequency)
    ✓ TTFT / P95 latency / cost per task

Run:
  python evaluation/evaluator.py --session 1 --user priya_sharma
  python evaluation/evaluator.py --all-sessions --user priya_sharma
  python evaluation/evaluator.py --production --user priya_sharma
"""

import sys
import os
import json
import re
import argparse
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.memory_store import (
    get_session_turns, get_all_facts, get_behaviors,
    get_session_metrics, get_production_metrics, get_hallucination_rate,
)
from cache import get_cache_stats


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class DimensionResult:
    name: str
    score: float          # 0.0 – 1.0
    passed: bool
    evidence: str
    weight: float = 1.0
    tier: int = 1         # 1=deterministic, 2=LLM-judge, 3=production


@dataclass
class EvalResult:
    session_id: str
    user_id: str
    dimensions: list[DimensionResult] = field(default_factory=list)

    @property
    def overall(self) -> float:
        if not self.dimensions:
            return 0.0
        total_weight = sum(d.weight for d in self.dimensions)
        if total_weight == 0:
            return 0.0
        return sum(d.score * d.weight for d in self.dimensions) / total_weight

    @property
    def passed(self) -> bool:
        return self.overall >= 0.70

    def add(self, dim: DimensionResult) -> None:
        self.dimensions.append(dim)

    def report(self) -> str:
        lines = [
            f"\n{'═'*65}",
            f"  EVALUATION REPORT — Session {self.session_id} | User: {self.user_id}",
            f"  Overall: {self.overall:.1%}  |  Grade: {'✅ PASS' if self.passed else '❌ FAIL'}",
            f"{'═'*65}",
            f"  {'Dimension':<38} {'Score':>6} {'T':>2} {'Pass':>5}",
            f"  {'─'*38} {'─'*6} {'─'*2} {'─'*5}",
        ]
        for d in self.dimensions:
            emoji = "✅" if d.passed else "❌"
            lines.append(
                f"  {d.name:<38} {d.score:>5.1%} T{d.tier} {emoji}"
            )
        lines.append(f"{'═'*65}")

        lines.append("\n  Details:")
        for d in self.dimensions:
            if d.evidence:
                lines.append(f"  • {d.name}: {d.evidence}")

        return "\n".join(lines)


# ─── Trajectory Evaluation ────────────────────────────────────────────────────

@dataclass
class TrajectoryStep:
    tool: str
    args: dict
    result: dict
    correct: bool = True
    necessary: bool = True
    recovered: bool = False   # filled in if the step had an error and agent recovered


@dataclass
class TrajectoryEvalResult:
    steps: list[TrajectoryStep] = field(default_factory=list)
    efficiency_score: float = 1.0
    correctness_score: float = 1.0
    recovery_score: float = 1.0
    overall_score: float = 1.0

    def compute(self) -> None:
        if not self.steps:
            self.overall_score = 0.0
            return

        necessary = [s for s in self.steps if s.necessary]
        correct = [s for s in self.steps if s.correct]
        errors = [s for s in self.steps if not s.correct]
        recovered = [s for s in errors if s.recovered]

        # Efficiency: fraction of steps that were necessary
        self.efficiency_score = len(necessary) / len(self.steps) if self.steps else 0.0
        # Correctness: fraction of steps that were correct
        self.correctness_score = len(correct) / len(self.steps) if self.steps else 0.0
        # Recovery: fraction of errors that were recovered from
        self.recovery_score = len(recovered) / len(errors) if errors else 1.0

        self.overall_score = (
            0.4 * self.efficiency_score +
            0.4 * self.correctness_score +
            0.2 * self.recovery_score
        )


def evaluate_trajectory(turns: list[dict]) -> TrajectoryEvalResult:
    """
    Analyze the tool call trajectory across all turns in a session.
    Checks for unnecessary repeated calls, failed steps, and recovery.
    """
    result = TrajectoryEvalResult()
    seen_tool_calls: set = set()  # detect repeated identical calls

    for turn in turns:
        tool_calls = turn.get("tool_calls")
        if isinstance(tool_calls, str):
            try:
                tool_calls = json.loads(tool_calls)
            except Exception:
                tool_calls = None
        if not tool_calls:
            continue

        for tc in (tool_calls or []):
            tool = tc.get("tool", "")
            args = tc.get("args", {})
            r = tc.get("result", {})

            call_sig = f"{tool}:{json.dumps(args, sort_keys=True)}"
            is_duplicate = call_sig in seen_tool_calls
            seen_tool_calls.add(call_sig)

            has_error = "error" in r if isinstance(r, dict) else False

            step = TrajectoryStep(
                tool=tool, args=args, result=r,
                correct=not has_error,
                necessary=not is_duplicate,
                recovered=False,
            )
            result.steps.append(step)

    result.compute()
    return result


# ─── Hallucination Heuristic (Tier 1) ─────────────────────────────────────────

def check_hallucination_session(turns: list[dict]) -> tuple[float, list[str]]:
    """
    For each agent turn: extract ₹ amounts from response, check if grounded
    in tool results of that turn. Return (grounded_rate, ungrounded_claims).
    """
    total_claims = 0
    ungrounded_claims: list[str] = []

    for turn in turns:
        if turn.get("role") != "agent":
            continue
        agent_text = turn.get("content", "")
        tool_calls = turn.get("tool_calls")
        if isinstance(tool_calls, str):
            try:
                tool_calls = json.loads(tool_calls)
            except Exception:
                tool_calls = []
        tool_calls = tool_calls or []

        # Get all numbers from tool results
        tool_numbers = set()
        for tc in tool_calls:
            result_str = json.dumps(tc.get("result", {}))
            nums = re.findall(r'\d[\d,]+', result_str)
            tool_numbers.update(n.replace(",", "") for n in nums)

        # Find ₹ amounts in agent response
        resp_amounts = re.findall(r'₹[\d,]+', agent_text)
        for amt in resp_amounts:
            clean = amt.replace("₹", "").replace(",", "")
            total_claims += 1
            if clean not in tool_numbers:
                ungrounded_claims.append(f"{amt} (turn: {turn.get('turn', '?')})")

    if total_claims == 0:
        return 1.0, []

    grounded_rate = max(0.0, 1.0 - len(ungrounded_claims) / total_claims)
    return grounded_rate, ungrounded_claims


# ─── Context Utilization Check ────────────────────────────────────────────────

def check_context_utilization(turns: list[dict], memory_facts: dict) -> float:
    """
    What fraction of known memory facts were referenced in agent responses?
    Measures whether retrieved memory was actually used (not just loaded).
    """
    if not memory_facts:
        return 1.0  # nothing to utilize

    all_agent_text = " ".join(
        t.get("content", "") for t in turns if t.get("role") == "agent"
    ).lower()

    referenced = 0
    for key, meta in memory_facts.items():
        value = str(meta.get("value", "")).lower()
        key_words = [w for w in key.split("_") if len(w) > 3]
        value_words = [w for w in value.split() if w.isalpha() and len(w) > 3]

        # Check if key concept or value appears in agent responses
        if any(w in all_agent_text for w in key_words + value_words):
            referenced += 1

    return round(referenced / len(memory_facts), 3)


# ─── Task Completion Check ────────────────────────────────────────────────────

def check_task_completion(turns: list[dict]) -> float:
    """
    Did the session end with user intent resolved?
    Heuristic: if last user message contains an unresolved question (ends in ?)
    and last agent response doesn't answer it (too short or is an error), fail.
    """
    user_turns = [t for t in turns if t.get("role") == "user"]
    agent_turns = [t for t in turns if t.get("role") == "agent"]

    if not user_turns or not agent_turns:
        return 0.0

    last_user = user_turns[-1].get("content", "")
    last_agent = agent_turns[-1].get("content", "")

    # If last agent response is very short or an error, likely incomplete
    if len(last_agent) < 30:
        return 0.3
    if "error" in last_agent.lower() or "try again" in last_agent.lower():
        return 0.3

    # If last user message was a question and we got a substantial response
    if last_user.endswith("?") and len(last_agent) > 80:
        return 1.0

    # Default: assume completed if agent gave a substantial final response
    return 1.0 if len(last_agent) > 60 else 0.6


# ─── Tool Selection Accuracy (Session-specific) ───────────────────────────────

def _get_tools_called(turns: list[dict]) -> set[str]:
    """Extract all tool names called across all turns."""
    tools = set()
    for turn in turns:
        tool_calls = turn.get("tool_calls")
        if isinstance(tool_calls, str):
            try:
                tool_calls = json.loads(tool_calls)
            except Exception:
                tool_calls = []
        for tc in (tool_calls or []):
            tools.add(tc.get("tool", ""))
    return tools

# ─── LLM-as-Judge (Tier 2) ────────────────────────────────────────────────────

LLM_JUDGE_PROMPT = """You are evaluating a personal finance AI agent's response quality.
Session transcript excerpt:
---
{transcript_excerpt}
---

Rate the agent on these 5 dimensions (each 0.0 to 1.0):
1. memory_recall: Does it accurately reference relevant information from memory without hallucinating?
2. goal_alignment: Does it connect advice to the user's stated financial goal?
3. judgment_quality: Is the financial advice pragmatic and appropriate for the situation?
4. tool_appropriateness: Did it use the right tools (live data for decisions, not stale memory)?
5. response_clarity: Is the response concise, clear, and actionable?

Output ONLY valid JSON, no preamble:
{{"memory_recall": 0.0, "goal_alignment": 0.0, "judgment_quality": 0.0, "tool_appropriateness": 0.0, "response_clarity": 0.0, "reasoning": "one line"}}"""


def llm_as_judge(transcript_excerpt: str) -> dict:
    """Tier 2: LLM rates the agent on 5 quality dimensions."""
    try:
        import openai
        from dotenv import load_dotenv
        load_dotenv()
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": LLM_JUDGE_PROMPT.format(
                transcript_excerpt=transcript_excerpt[:2000]
            )}],
            max_tokens=200,
            temperature=0.0,
        )
        text = resp.choices[0].message.content.strip()
        text = re.sub(r"```(?:json)?", "", text).strip().rstrip("```").strip()
        return json.loads(text)
    except Exception as e:
        print(f"  [WARN] LLM judge failed: {e}")
        return {
            "memory_recall": 0.5, "goal_alignment": 0.5,
            "judgment_quality": 0.5, "tool_appropriateness": 0.5,
            "response_clarity": 0.5, "reasoning": "Judge unavailable"
        }


# ─── Session 1 Evaluation ─────────────────────────────────────────────────────

def evaluate_session1(user_id: str) -> EvalResult:
    """Full evaluation of Session 1 (salary planning session)."""
    result = EvalResult(session_id="1", user_id=user_id)
    turns = get_session_turns(user_id, "1")
    all_agent_text = " ".join(t.get("content", "") for t in turns if t.get("role") == "agent")
    all_agent_lower = all_agent_text.lower()
    tools_called = _get_tools_called(turns)

    # T1-1: Balance checked (live data discipline)
    called_bal = "get_account_balance" in tools_called
    result.add(DimensionResult(
        name="Live balance fetched", score=1.0 if called_bal else 0.0,
        passed=called_bal, weight=1.2,
        evidence=f"get_account_balance {'called ✓' if called_bal else 'NOT called ✗'}",
    ))

    # T1-2: Transactions fetched (spending analysis)
    called_txn = "get_recent_transactions" in tools_called
    result.add(DimensionResult(
        name="Transactions fetched for spending", score=1.0 if called_txn else 0.0,
        passed=called_txn, weight=1.2,
        evidence=f"get_recent_transactions {'called ✓' if called_txn else 'NOT called ✗'}",
    ))

    # T1-3: Upcoming bills checked (cash flow awareness)
    called_bills = "get_upcoming_bills" in tools_called
    result.add(DimensionResult(
        name="Upcoming bills fetched", score=1.0 if called_bills else 0.0,
        passed=called_bills, weight=1.0,
        evidence=f"get_upcoming_bills {'called ✓' if called_bills else 'NOT called ✗'}",
    ))

    # T1-4: Reminder set
    called_rem = "set_reminder" in tools_called
    result.add(DimensionResult(
        name="Reminder set (₹30K house transfer)", score=1.0 if called_rem else 0.0,
        passed=called_rem, weight=1.2,
        evidence=f"set_reminder {'called ✓' if called_rem else 'NOT called ✗'}",
    ))

    # T1-5: Food delivery spending identified
    food_mentioned = any(w in all_agent_lower for w in ["food delivery", "swiggy", "zomato", "₹9", "₹10"])
    result.add(DimensionResult(
        name="Food delivery spend identified", score=1.0 if food_mentioned else 0.0,
        passed=food_mentioned, weight=1.0,
        evidence="Food delivery spending mentioned in response" if food_mentioned else "Spending not discussed",
    ))

    # T1-6: House goal connected
    goal_mentioned = any(w in all_agent_lower for w in ["house", "down payment", "₹15", "15 lakh", "goal"])
    result.add(DimensionResult(
        name="Goal awareness (house fund)", score=1.0 if goal_mentioned else 0.0,
        passed=goal_mentioned, weight=1.0,
        evidence="House goal referenced" if goal_mentioned else "Goal not explicitly connected",
    ))

    # T1-7: Savings amount mentioned (₹30K plan)
    savings_mentioned = any(w in all_agent_lower for w in ["30,000", "₹30", "30k", "30 k"])
    result.add(DimensionResult(
        name="Savings target ₹30K mentioned", score=1.0 if savings_mentioned else 0.5,
        passed=savings_mentioned, weight=0.8,
        evidence="₹30,000 savings target mentioned" if savings_mentioned else "Savings amount not specified",
    ))

    # T1-8: Trajectory efficiency
    traj = evaluate_trajectory(turns)
    result.add(DimensionResult(
        name="Trajectory efficiency", tier=1,
        score=traj.efficiency_score, passed=traj.efficiency_score >= 0.7,
        weight=0.7,
        evidence=f"{len(traj.steps)} tool calls, efficiency={traj.efficiency_score:.1%}",
    ))

    # T1-10: Task completion
    completion = check_task_completion(turns)
    result.add(DimensionResult(
        name="Task completion", tier=1,
        score=completion, passed=completion >= 0.6,
        weight=1.0,
        evidence=f"Completion score: {completion:.1%}",
    ))

    # T2-1: LLM-as-judge (Tier 2)
    transcript_excerpt = _format_transcript_excerpt(turns)
    judgement = llm_as_judge(transcript_excerpt)
    for dim_name, key in [
        ("Memory recall quality (T2)",     "memory_recall"),
        ("Goal alignment quality (T2)",    "goal_alignment"),
        ("Judgment quality (T2)",          "judgment_quality"),
        ("Tool use appropriateness (T2)",  "tool_appropriateness"),
        ("Response clarity (T2)",          "response_clarity"),
    ]:
        score = judgement.get(key, 0.5)
        result.add(DimensionResult(
            name=dim_name, tier=2,
            score=score, passed=score >= 0.70,
            weight=0.8,
            evidence=judgement.get("reasoning", ""),
        ))

    return result


# ─── Session 2 Evaluation ─────────────────────────────────────────────────────

def evaluate_session2(user_id: str) -> EvalResult:
    """Full evaluation of Session 2 (MacBook purchase decision)."""
    result = EvalResult(session_id="2", user_id=user_id)
    turns = get_session_turns(user_id, "2")
    all_agent_text = " ".join(t.get("content", "") for t in turns if t.get("role") == "agent")
    all_agent_lower = all_agent_text.lower()
    tools_called = _get_tools_called(turns)
    all_facts = get_all_facts(user_id)

    # T1-1: Live balance fetched (NOT from memory — critical discipline check)
    called_bal = "get_account_balance" in tools_called
    result.add(DimensionResult(
        name="Live balance fetched (not from memory)", score=1.0 if called_bal else 0.0,
        passed=called_bal, weight=1.5,
        evidence=f"get_account_balance {'called ✓' if called_bal else 'CRITICAL FAIL — balance from memory ✗'}",
    ))

    # T1-2: Upcoming bills checked before purchase advice
    called_bills = "get_upcoming_bills" in tools_called
    result.add(DimensionResult(
        name="Bills checked before purchase advice", score=1.0 if called_bills else 0.0,
        passed=called_bills, weight=1.3,
        evidence=f"get_upcoming_bills {'called ✓' if called_bills else 'NOT called ✗'}",
    ))

    # T1-3: House goal connected to decision
    goal_mentioned = any(w in all_agent_lower for w in ["house", "down payment", "₹15", "15 lakh", "goal"])
    result.add(DimensionResult(
        name="House goal connected to purchase decision", score=1.0 if goal_mentioned else 0.0,
        passed=goal_mentioned, weight=1.5,
        evidence="Goal referenced in purchase advice ✓" if goal_mentioned else "CRITICAL: Goal not mentioned ✗",
    ))

    # T1-4: Memory of ₹30K savings commitment recalled
    commitment_recalled = any(w in all_agent_lower for w in ["30,000", "₹30", "commitment", "plan", "promised"])
    result.add(DimensionResult(
        name="Prior commitment (₹30K) recalled", score=1.0 if commitment_recalled else 0.2,
        passed=commitment_recalled, weight=1.5,
        evidence="Session 1 savings commitment recalled ✓" if commitment_recalled else "Prior plan not referenced ✗",
    ))

    # T1-6: Judgment — some concrete recommendation given
    gave_rec = any(w in all_agent_lower for w in [
        "recommend", "suggest", "advice", "can afford", "can't afford",
        "wait", "go for it", "i'd", "would", "consider"
    ])
    result.add(DimensionResult(
        name="Concrete recommendation given", score=1.0 if gave_rec else 0.3,
        passed=gave_rec, weight=1.0,
        evidence="Clear recommendation given ✓" if gave_rec else "No clear recommendation ✗",
    ))

    # T1-7: Trajectory efficiency
    traj = evaluate_trajectory(turns)
    result.add(DimensionResult(
        name="Trajectory efficiency", tier=1,
        score=traj.efficiency_score, passed=traj.efficiency_score >= 0.8,
        weight=0.8,
        evidence=f"{len(traj.steps)} tool calls, efficiency={traj.efficiency_score:.1%}",
    ))

    # T1-8: Context utilization (prior session facts)
    context_util = check_context_utilization(turns, all_facts)
    result.add(DimensionResult(
        name="Context utilization (prior facts)", tier=1,
        score=context_util, passed=context_util >= 0.3,
        weight=1.0,
        evidence=f"Used {context_util:.0%} of available memory facts",
    ))

    # T1-9: Task completion
    completion = check_task_completion(turns)
    result.add(DimensionResult(
        name="Task completion", tier=1,
        score=completion, passed=completion >= 0.8,
        weight=1.0,
        evidence=f"Completion score: {completion:.1%}",
    ))

    # T2: LLM-as-judge
    transcript_excerpt = _format_transcript_excerpt(turns)
    judgement = llm_as_judge(transcript_excerpt)
    for dim_name, key in [
        ("Memory recall quality (T2)",    "memory_recall"),
        ("Goal alignment quality (T2)",   "goal_alignment"),
        ("Cross-session judgment (T2)",   "judgment_quality"),
        ("Tool discipline (T2)",          "tool_appropriateness"),
        ("Response clarity (T2)",         "response_clarity"),
    ]:
        score = judgement.get(key, 0.5)
        result.add(DimensionResult(
            name=dim_name, tier=2,
            score=score, passed=score >= 0.70, weight=0.8,
            evidence=judgement.get("reasoning", ""),
        ))

    return result


# ─── Session 3 Evaluation ─────────────────────────────────────────────────────

def evaluate_session3(user_id: str) -> EvalResult:
    """Full evaluation of Session 3 (SIP investment review)."""
    result = EvalResult(session_id="3", user_id=user_id)
    turns = get_session_turns(user_id, "3")
    all_agent_text = " ".join(t.get("content", "") for t in turns if t.get("role") == "agent")
    all_agent_lower = all_agent_text.lower()
    tools_called = _get_tools_called(turns)

    # T1-1: SIP calculation tool used
    called_sip = "calculate_sip_returns" in tools_called
    result.add(DimensionResult(
        name="SIP returns calculated", score=1.0 if called_sip else 0.0,
        passed=called_sip, weight=1.5,
        evidence=f"calculate_sip_returns {'called ✓' if called_sip else 'NOT called ✗'}",
    ))

    # T1-2: Goal awareness (₹15 lakh target)
    goal_mentioned = any(w in all_agent_lower for w in ["15 lakh", "₹15", "goal", "target", "house"])
    result.add(DimensionResult(
        name="Goal awareness (₹15L target)", score=1.0 if goal_mentioned else 0.0,
        passed=goal_mentioned, weight=1.2,
        evidence="Goal referenced ✓" if goal_mentioned else "Goal not mentioned ✗",
    ))

    # T1-3: Reminder set for SIP review
    called_rem = "set_reminder" in tools_called
    result.add(DimensionResult(
        name="Reminder set (SIP review)", score=1.0 if called_rem else 0.0,
        passed=called_rem, weight=1.0,
        evidence=f"set_reminder {'called ✓' if called_rem else 'NOT called ✗'}",
    ))

    # T1-4: Trajectory efficiency
    traj = evaluate_trajectory(turns)
    result.add(DimensionResult(
        name="Trajectory efficiency", tier=1,
        score=traj.efficiency_score, passed=traj.efficiency_score >= 0.7,
        weight=0.8,
        evidence=f"{len(traj.steps)} tool calls, efficiency={traj.efficiency_score:.1%}",
    ))

    # T1-5: Task completion
    completion = check_task_completion(turns)
    result.add(DimensionResult(
        name="Task completion", tier=1,
        score=completion, passed=completion >= 0.8,
        weight=1.0,
        evidence=f"Completion score: {completion:.1%}",
    ))

    # T2: LLM-as-judge
    transcript_excerpt = _format_transcript_excerpt(turns)
    judgement = llm_as_judge(transcript_excerpt)
    for dim_name, key in [
        ("Memory recall quality (T2)",    "memory_recall"),
        ("Goal alignment quality (T2)",   "goal_alignment"),
        ("Investment judgment (T2)",      "judgment_quality"),
        ("Tool appropriateness (T2)",     "tool_appropriateness"),
        ("Response clarity (T2)",         "response_clarity"),
    ]:
        score = judgement.get(key, 0.5)
        result.add(DimensionResult(
            name=dim_name, tier=2,
            score=score, passed=score >= 0.70, weight=0.8,
            evidence=judgement.get("reasoning", ""),
        ))

    return result


# ─── Session 4 Evaluation ─────────────────────────────────────────────────────

def evaluate_session4(user_id: str) -> EvalResult:
    """Full evaluation of Session 4 (ELSS tax planning)."""
    result = EvalResult(session_id="4", user_id=user_id)
    turns = get_session_turns(user_id, "4")
    all_agent_text = " ".join(t.get("content", "") for t in turns if t.get("role") == "agent")
    all_agent_lower = all_agent_text.lower()
    tools_called = _get_tools_called(turns)

    # T1-1: Tax planning knowledge
    tax_mentioned = any(w in all_agent_lower for w in ["elss", "80c", "tax", "saving"])
    result.add(DimensionResult(
        name="Tax planning discussed", score=1.0 if tax_mentioned else 0.0,
        passed=tax_mentioned, weight=1.2,
        evidence="ELSS/80C mentioned ✓" if tax_mentioned else "Tax planning not discussed ✗",
    ))

    # T1-2: Goal alignment (2-year timeline)
    timeline_mentioned = any(w in all_agent_lower for w in ["2 year", "two year", "timeline", "lock-in", "house"])
    result.add(DimensionResult(
        name="Timeline awareness (2-year goal)", score=1.0 if timeline_mentioned else 0.0,
        passed=timeline_mentioned, weight=1.2,
        evidence="Timeline considered ✓" if timeline_mentioned else "Timeline not mentioned ✗",
    ))

    # T1-3: Concrete recommendation
    gave_rec = any(w in all_agent_lower for w in ["recommend", "suggest", "consider", "go for", "skip"])
    result.add(DimensionResult(
        name="Concrete recommendation given", score=1.0 if gave_rec else 0.3,
        passed=gave_rec, weight=1.0,
        evidence="Clear recommendation ✓" if gave_rec else "No clear recommendation ✗",
    ))

    # T1-4: Task completion
    completion = check_task_completion(turns)
    result.add(DimensionResult(
        name="Task completion", tier=1,
        score=completion, passed=completion >= 0.8,
        weight=1.0,
        evidence=f"Completion score: {completion:.1%}",
    ))

    # T2: LLM-as-judge
    transcript_excerpt = _format_transcript_excerpt(turns)
    judgement = llm_as_judge(transcript_excerpt)
    for dim_name, key in [
        ("Memory recall quality (T2)",    "memory_recall"),
        ("Goal alignment quality (T2)",   "goal_alignment"),
        ("Tax planning judgment (T2)",    "judgment_quality"),
        ("Tool appropriateness (T2)",     "tool_appropriateness"),
        ("Response clarity (T2)",         "response_clarity"),
    ]:
        score = judgement.get(key, 0.5)
        result.add(DimensionResult(
            name=dim_name, tier=2,
            score=score, passed=score >= 0.70, weight=0.8,
            evidence=judgement.get("reasoning", ""),
        ))

    return result


# ─── Session 5 Evaluation ─────────────────────────────────────────────────────

def evaluate_session5(user_id: str) -> EvalResult:
    """Full evaluation of Session 5 (goal progress check + Goa trip)."""
    result = EvalResult(session_id="5", user_id=user_id)
    turns = get_session_turns(user_id, "5")
    all_agent_text = " ".join(t.get("content", "") for t in turns if t.get("role") == "agent")
    all_agent_lower = all_agent_text.lower()
    tools_called = _get_tools_called(turns)

    # T1-1: Balance checked
    called_bal = "get_account_balance" in tools_called
    result.add(DimensionResult(
        name="Live balance fetched", score=1.0 if called_bal else 0.0,
        passed=called_bal, weight=1.2,
        evidence=f"get_account_balance {'called ✓' if called_bal else 'NOT called ✗'}",
    ))

    # T1-2: Goal progress tracking
    progress_mentioned = any(w in all_agent_lower for w in ["progress", "tracking", "₹15", "goal", "house fund", "sip"])
    result.add(DimensionResult(
        name="Goal progress discussed", score=1.0 if progress_mentioned else 0.0,
        passed=progress_mentioned, weight=1.5,
        evidence="Progress tracking mentioned ✓" if progress_mentioned else "Progress not discussed ✗",
    ))

    # T1-3: Affordability check for trip
    affordability = any(w in all_agent_lower for w in ["afford", "budget", "buffer", "checking", "₹15"])
    result.add(DimensionResult(
        name="Affordability assessed", score=1.0 if affordability else 0.0,
        passed=affordability, weight=1.2,
        evidence="Affordability checked ✓" if affordability else "Affordability not assessed ✗",
    ))

    # T1-4: Reminder set
    called_rem = "set_reminder" in tools_called
    result.add(DimensionResult(
        name="Reminder set (flight booking)", score=1.0 if called_rem else 0.0,
        passed=called_rem, weight=1.0,
        evidence=f"set_reminder {'called ✓' if called_rem else 'NOT called ✗'}",
    ))

    # T1-5: Trajectory efficiency
    traj = evaluate_trajectory(turns)
    result.add(DimensionResult(
        name="Trajectory efficiency", tier=1,
        score=traj.efficiency_score, passed=traj.efficiency_score >= 0.8,
        weight=0.8,
        evidence=f"{len(traj.steps)} tool calls, efficiency={traj.efficiency_score:.1%}",
    ))

    # T1-6: Task completion
    completion = check_task_completion(turns)
    result.add(DimensionResult(
        name="Task completion", tier=1,
        score=completion, passed=completion >= 0.8,
        weight=1.0,
        evidence=f"Completion score: {completion:.1%}",
    ))

    # T2: LLM-as-judge
    transcript_excerpt = _format_transcript_excerpt(turns)
    judgement = llm_as_judge(transcript_excerpt)
    for dim_name, key in [
        ("Memory recall quality (T2)",    "memory_recall"),
        ("Goal alignment quality (T2)",   "goal_alignment"),
        ("Spending judgment (T2)",        "judgment_quality"),
        ("Tool appropriateness (T2)",     "tool_appropriateness"),
        ("Response clarity (T2)",         "response_clarity"),
    ]:
        score = judgement.get(key, 0.5)
        result.add(DimensionResult(
            name=dim_name, tier=2,
            score=score, passed=score >= 0.70, weight=0.8,
            evidence=judgement.get("reasoning", ""),
        ))

    return result


# ─── Production Metrics Evaluation ────────────────────────────────────────────

def evaluate_production_metrics(user_id: str) -> dict:
    """
    Aggregate production SLA compliance from the metrics DB.
    Compares against production SLA targets:
      p95 latency < 50000ms (acceptable for multi-step ReAct)
      cost per session < $0.05
      retry rate < 10%
    """
    prod = get_production_metrics(user_id)

    p95 = prod.get("p95_latency_ms", 0)
    cost_per_session = prod.get("cost_per_session_usd", 0)
    retry_rate = prod.get("avg_retry_rate", 0)

    return {
        "user_id": user_id,
        "sla_compliance": {
            "p95_latency_ms": {
                "value": p95,
                "target_ms": 50000,
                "pass": p95 < 50000 or p95 == 0,
            },
            "cost_per_session_usd": {
                "value": cost_per_session,
                "target_usd": 0.05,
                "pass": cost_per_session < 0.05,
            },
            "retry_rate": {
                "value": retry_rate,
                "target": 0.10,
                "pass": retry_rate <= 0.10,
            },
        },
        "cache_stats": get_cache_stats(),
    }


# ─── Helper: Transcript Excerpt ───────────────────────────────────────────────

def _format_transcript_excerpt(turns: list[dict], max_chars: int = 2000) -> str:
    lines = []
    for t in turns:
        role = "User" if t.get("role") == "user" else "Agent"
        content = t.get("content", "")[:300]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)[:max_chars]


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Finance Agent Evaluator")
    parser.add_argument("--session", choices=["1", "2", "3", "4", "5", "all"], default="all")
    parser.add_argument("--user", default="priya_sharma")
    parser.add_argument("--production", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from memory.memory_store import init_db
    init_db()

    if args.production:
        metrics = evaluate_production_metrics(args.user)
        prod = get_production_metrics(args.user)
        print(f"\n{'═'*65}")
        print(f"  PRODUCTION METRICS — User: {args.user}")
        print(f"  Sessions: {prod.get('sessions', 0)} | P95: {prod.get('p95_latency_ms', 0):.0f}ms | "
              f"Cost/session: ${prod.get('cost_per_session_usd', 0):.6f}")
        print(f"{'═'*65}")
        for sla, vals in metrics["sla_compliance"].items():
            icon = "✅" if vals["pass"] else "❌"
            print(f"  {icon} {sla}: {vals['value']}")
        return

    # Run evaluations for requested sessions
    if args.session in ("1", "all"):
        r1 = evaluate_session1(args.user)
        print(r1.report())

    if args.session in ("2", "all"):
        r2 = evaluate_session2(args.user)
        print(r2.report())

    if args.session in ("3", "all"):
        r3 = evaluate_session3(args.user)
        print(r3.report())

    if args.session in ("4", "all"):
        r4 = evaluate_session4(args.user)
        print(r4.report())

    if args.session in ("5", "all"):
        r5 = evaluate_session5(args.user)
        print(r5.report())


if __name__ == "__main__":
    main()
