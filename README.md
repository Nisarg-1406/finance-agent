# Production-Grade AI Finance Agent

A cross-session personal finance companion demonstrating production-scale agent architecture: multi-layer memory, async parallel tool execution, three-tier caching, and a multi-dimensional evaluation framework.

Main part for the agent - **what you remember, what you fetch live, and how you connect them.**

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        Finance Agent                                     │
│                                                                          │
│  ┌─────────────┐    ┌──────────────────────────────────────────────┐    │
│  │  UI / CLI   │    │         FastAPI (api.py) — Async              │    │
│  │  ui.html    │───▶│  /turn  /session/run  /evaluate  /metrics     │    │
│  └─────────────┘    └────────────────────┬─────────────────────────┘    │
│                                          │                               │
│                    ┌─────────────────────▼──────────────────────────┐   │
│                    │           ReAct Agent Loop (agent.py)           │   │
│                    │                                                  │   │
│                    │  1. Metacognitive pre-check (<5ms, rule-based)   │   │
│                    │  2. Build context window (sliding window + sum)  │   │
│                    │  3. LLM call (async streaming, TTFT measured)    │   │
│                    │  4. Parse tool calls                             │   │
│                    │  5. Execute tools in PARALLEL (asyncio.gather)   │   │
│                    │  6. Inject observations → LLM final response     │   │
│                    │  7. Persist to episodic memory + metrics DB     │   │
│                    └─────────┬──────────────┬──────────────────────┘   │
│                               │              │                           │
│          ┌────────────────────┘  ┌───────────┘                          │
│          ▼                       ▼                                       │
│  ┌──────────────────┐   ┌─────────────────────────────────────────┐    │
│  │   Tool Layer     │   │         Three-Tier Cache (cache.py)      │    │
│  │  (tools.py)      │   │  L1: Exact match   — 5min TTL  (Redis)  │    │
│  │                  │   │  L2: Semantic — 10min TTL (Qdrant)      │    │
│  │  12 tools:       │   │  L3: Tool result   — 2min TTL  (Redis)  │    │
│  │  · get_balance   │   │  → Eliminates 30-50% LLM calls at scale │    │
│  │  · get_txns      │   └─────────────────────────────────────────┘    │
│  │  · get_bills     │                                                    │
│  │  · set_reminder  │   ┌─────────────────────────────────────────┐    │
│  │  · filter_month  │   │       Memory Layer (memory/)             │    │
│  │  · analyze_spend │   │  Episodic  (SQLite → episodic table)     │    │
│  │  · calc_sip      │   │   Raw turns + tool calls per session     │    │
│  │  · calc_savings  │   │                                          │    │
│  │  · calc_goal     │   │  Semantic  (SQLite → semantic table)     │    │
│  │  · calc_networth │   │   Distilled facts: goals, commitments,   │    │
│  │  · calc_health   │   │   concerns. With importance scoring 1-5  │    │
│  │  · calc_balance  │   │   + cosine reranking                     │    │
│  │                  │   │                                          │    │
│  └──────────────────┘   │  Procedural (SQLite → procedural table)  │    │
│                          │   Behavioral patterns from feedback      │    │
│                          │                                          │    │
│                          │  Metrics  (SQLite → metrics table)       │    │
│                          │   TTFT/P95/cost per turn                 │    │
│                          │                                          │    │
│                          │  Distiller (distiller.py)                │    │
│                          │   1 LLM call/session → importance-scored │    │
│                          │   facts + deduplication                  │    │
│                          │                                          │    │
│                          │  Context Manager (context_manager.py)    │    │
│                          │   Sliding window (last 8 turns) +        │    │
│                          │   LLM summarization of older turns       │    │
│                          └─────────────────────────────────────────┘    │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Evaluation Framework (evaluation/evaluator.py)                   │   │
│  │  Tier 1 (<5ms): deterministic tool/memory/reminder checks        │   │
│  │  Tier 2 (~300ms): LLM-as-judge — quality, judgment, clarity      │   │
│  │  Tier 3: production metrics — P95 latency, cost, retry rate      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Why These Architecture Decisions

### Single-Agent ReAct (not Multi-Agent)

This is a well-scoped domain: one user, one goal, 12 tools. Multi-agent adds orchestration overhead with no benefit. Single ReAct is adaptive (each turn can change tool plan based on the user's response), simpler to debug (every thought-action-observation cycle is logged), and faster (no inter-agent message passing).

I did not considered a Planner-Executor split (planner decides tool sequence up front, executor runs them), as ReAct is strictly better here: plans change based on what tools return. The user in Session 2 might have ₹10,000 in checking or ₹1,00,000 — the plan for "should I buy the MacBook?" depends entirely on the live balance. Also, I did not used the Multi-agent system specifically a Planner-Replan pattern with a supervisor agent dispatching to specialist sub-agents due to latency and cost issue. Every inter-agent handoff is an additional LLM call: 500–1,500ms per hop, 2–3× the token cost per turn. For a real-time conversational app, that overhead has no justification when a single ReAct agent handles all the reasoning in one pass.

### Async + Parallel Tool Execution

```python
# All tool calls from one LLM response run in parallel
results = await asyncio.gather(*[execute_tool(c) for c in calls])
```

When the LLM asks for `get_account_balance` + `get_upcoming_bills` simultaneously, they run concurrently. For 2 tools at ~200ms each: sequential = 400ms, parallel = ~210ms. At scale, this is the single largest latency win (60–70% reduction per tool-heavy turn).

### Memory: Three Types, Selective Storage

| Type | Storage | What's stored | What's deliberately excluded |
|------|---------|---------------|------------------------------|
| **Episodic** | SQLite `episodic` | Raw turns + tool calls | — (all turns saved for distillation) |
| **Semantic** | SQLite `semantic` | Goals, commitments, concerns — importance ≥ 3 only | Balances, transaction amounts, bill amounts |
| **Procedural** | SQLite `procedural` | Behavioral patterns from user ratings | — |
| **Metrics** | SQLite `metrics` | TTFT, cost, tool counts per turn | — |

Balances are excluded from memory because they change constantly — re-fetching from `get_account_balance()` is always safer than trusting a stored number. A stale balance in advice is an error.

### Context Engineering

The system prompt is split for cache efficiency:

```
[STATIC_SYSTEM_PREFIX]     ← tokens are KV-cached as it is identical for all users
[User identity]            ← tokens per-user
[Memory + Behaviors]       ← per-user per-session, distilled facts only
[Session summary]          ← compressed older turns (sliding window)
[Tool definitions]         ← intent-filtered, max 5 tools per turn
```

At scale, the static prefix is cached in the LLM serving layer (prompt KV caching). This eliminates re-encoding ~800 tokens on every call → ~40% TTFT reduction.

Tool definitions are intent-filtered: a balance query gets 2 tools, not all 12. Fewer tools = less context distraction = more accurate tool selection.

### Memory Distillation: One LLM Call Per Session

After each session, one LLM call distills the full transcript into importance-scored facts. The distiller scores each fact 1–5 (5 = critical goal/commitment, 1 = noise). Only facts with importance ≥ 3 are stored. Deduplication uses cosine similarity (numpy) to prevent the same fact accreting over sessions with slightly different phrasing.

### Three-Tier Cache

- **L1 Exact**: SHA256 hash of (user_message + session_id). Hit = return cached response immediately. 5-min TTL. Handles repeated queries ("what's my balance?" in rapid succession).
- **L2 Semantic**: Cosine similarity of embeddings. 10-min TTL. Handles paraphrased repeats ("how much money do I have?" ≈ "what's my checking balance?").
- **L3 Tool Result**: Per-session tool output cache. 2-min TTL. Prevents calling `get_account_balance()` twice in the same session turn.

Production swap: L1/L3 → Redis `SETEX`. L2 → Qdrant or pgvector nearest-neighbour search.

---

### The Two Critical Disciplines

Two properties separate a good finance agent from a mediocre one:

**1. Tool vs memory discipline.** The agent must re-fetch every balance, transaction, and bill — even if it remembers them from a prior session. A stale balance in advice is not just inaccurate; it's potentially harmful. The system prompt has a hard rule: "NEVER quote a specific balance from memory — always re-fetch." The evaluation framework checks this automatically.

**2. Goal-linkage.** Every financial decision — a MacBook purchase, a Goa trip, an ELSS investment — must be connected back to the user's stated goal. In Session 2, the right response to "should I buy the MacBook?" is not "yes, you have enough money." It's "you have enough, but this costs 2.7 months of your savings gap toward the ₹15L goal — here's the tradeoff." The agent does this naturally because the goal is baked into the system prompt and the semantic memory.

---

### Evaluation Highlights

Session 2 is the projects's key test: can the agent recall Session 1 memory and connect a new question to it without being prompted?

The actual Session 2 response:

> "Dropping ₹80,000 on a MacBook today would eat into your liquid buffer. [...] remember you're already saving ₹40,000/month toward that ₹15 lakh down-payment goal. Pulling ₹80,000 now means shaving off two full months of planned savings — and slowing your timeline."

This demonstrates all four project requirements simultaneously:
1. **Memory**: recalls the ₹30K + ₹10K SIP = ₹40K/month commitment from Session 1
2. **Judgment**: connects MacBook cost to savings gap without being asked
3. **Tool discipline**: fetches fresh balance (₹99,820) rather than quoting Session 1's ₹1,28,000
4. **Tool action**: would set a reminder if requested

---

## Scalability Path

| Component | Current (Demo) | Production (1M users) |
|-----------|---------------|----------------------|
| Database | SQLite (WAL mode) | PostgreSQL + pgvector |
| Cache | In-process dict | Redis Cluster |
| Tool execution | asyncio.gather | Same — already async |
| Context storage | Local dict | Redis per-session (TTL=30min) |
| LLM serving | OpenAI API | vLLM pool or Azure OpenAI |
| Agent pods | Single process | K8s stateless pods (all state external) |
| Distillation | Sync, end of session | Async background worker (Celery/SQS) |
| Embeddings | numpy cosine | pgvector ANN (HNSW index) |

---

## Production-Grade System Features

This system demonstrates several production-ready capabilities beyond the basic project requirements:

### 1. **Parallel Tool Execution (60-70% Latency Reduction)**
- Multiple tool calls execute concurrently via `asyncio.gather`
- Example: `get_account_balance` + `get_upcoming_bills` run in parallel (~210ms) vs sequential (~400ms)
- Critical for scale: 5+ tools per turn benefit from concurrent execution
- All tool functions use thread pool executor to avoid blocking the async event loop

### 2. **Three-Layer Memory System**
- **Episodic**: Raw conversation logs for post-session distillation
- **Semantic**: Importance-scored facts (1-5 scale, only ≥3 stored) with cosine similarity deduplication
- **Procedural**: Behavioral patterns from user feedback (preferences, communication style)
- Strict boundary: memory stores intent, tools fetch live state

### 3. **Sliding Window Context Management**
- Keeps last 5 turns verbatim, compresses older turns via LLM summarization
- Prevents unbounded context growth in long sessions
- Maintains conversation coherence while managing token limits

### 4. **Three-Tier Caching System (30-50% LLM Call Reduction)**
- **L1 Exact Match**: SHA256 hash, 5min TTL (handles rapid repeats)
- **L2 Semantic**: Cosine similarity ≥0.92, 10min TTL (paraphrased queries)
- **L3 Tool Result**: Per-session cache, 2min TTL (highest ROI - same tool, same session)
- Production-ready: designed for Redis (L1/L3) and Qdrant/pgvector (L2) swap

### 5. **Multi-Dimensional Evaluation Framework**
- **Tier 1 (<5ms)**: Deterministic checks - tool calls, memory retrieval, reminder setting
- **Tier 2 (~300ms)**: LLM-as-judge - quality, judgment, goal-linkage, clarity
- **Tier 3**: Production metrics - P95 latency, cost/session, retry rate, cache hit rate
- Scalable approach: Tier 1 runs continuously, Tier 2 on samples, Tier 3 on ops dashboard

### 6. **Prompt Engineering for Performance**
- **Static prefix (~800 tokens)**: KV-cached across all users → ~40% TTFT reduction
- **Intent-filtered tools**: 2-5 relevant tools per turn (not all 12) → better tool selection accuracy
- **Explicit discipline rules**: "Never quote balance from memory, always re-fetch" prevents stale data errors
- **Failure mode fixes**: Reminder confirmation loop eliminated via explicit prompt rules

### 7. **Memory Distillation (One LLM Call Per Session)**
- Post-session fact extraction with importance scoring (1-5)
- Deduplication via cosine similarity prevents fact accumulation
- Only stores durable intent (goals, commitments), excludes ephemeral state (balances, transactions)
- Structured JSON output with `existing_keys` parameter prevents duplicates

### 8. **Metacognitive Pre-Check (<5ms)**
- Rule-based finance query detection before LLM call
- Intent classification maps to relevant tool subset
- Reduces unnecessary LLM invocations and improves tool selection

### 9. **Comprehensive Testing Across 5 Sessions**
- Session 1: Savings planning (99.2%)
- Session 2: Purchase decision with memory recall (97.6%)
- Session 3: Compound investment math (93.7%)
- Session 4: Tax optimization linked to 2-year goal (90.5%)
- Session 5: Goal progress tracking + lifestyle affordability (98.8%)
- 17/18 correct tool executions across all sessions

### 10. **Production Scalability Path**
- SQLite → PostgreSQL + pgvector (ANSI SQL compatibility maintained)
- In-process cache → Redis Cluster with TTL
- Sync distillation → Async background workers (Celery/SQS)
- Single process → K8s stateless pods (all state externalized)
- Designed for horizontal scaling from day one

### 11. **Observability & Metrics**
- TTFT (Time To First Token) tracking
- P95 latency monitoring
- Cost per session/turn tracking
- Tool retry rate measurement
- Cache hit rate analytics
- All metrics stored in SQLite `metrics` table for analysis

### 12. **Tool vs Memory Discipline**
- Hard rule: Always re-fetch live state (balances, transactions, bills)
- Memory stores only durable intent (goals, commitments, concerns)
- Prevents stale data errors that could harm financial advice
- Evaluation framework automatically validates this discipline

## Tools

| Tool | Description |
|------|-------------|
| `get_recent_transactions` | Returns all transactions from the last N days, with negative amounts as debits and positive as credits (INR). |
| `get_account_balance` | Returns current balances across checking, savings, house fund, and mutual fund accounts for the active session. |
| `get_upcoming_bills` | Lists scheduled bills and auto-debits due within the next N days from the current session date. |
| `set_reminder` | Logs a dated reminder and returns a confirmation with a generated reminder ID. |
| `calculate_current_balance_from_transactions` | Derives updated account balances by replaying transactions and pending transfers on top of an initial balance snapshot. |
| `filter_transactions_by_month` | Filters a transaction list to only those from a specified year and month, useful for "last month" spending queries. |
| `calculate_sip_returns` | Computes SIP future value using the standard annuity formula with monthly compounding at a given annual return rate. |
| `calculate_savings_projection` | Projects the compounded future value of a fixed monthly savings contribution over a given number of months. |
| `analyze_spending_by_category` | Aggregates total spend, transaction count, and average per transaction for a given category from a transaction list. |
| `calculate_goal_progress` | Estimates percent completion and months remaining toward a savings goal given current savings and monthly contribution. |
| `calculate_net_worth` | Sums all account balances into total net worth, broken down by liquid assets, investments, and goal savings. |
| `calculate_financial_health_score` | Produces a 0–100 composite health score with grade and recommendations based on income, expenses, savings rate, and debt. |

---

## File Structure

```
finance-agent/
├── agent.py                  # ReAct loop, async tool execution, session runner
├── cache.py                  # Three-tier in-process cache (L1/L2/L3)
├── tools.py                  # Original 4 stubs + 8 pure-Python extensions
├── simulate_users.py         # 5-user × 10-session stress test runner
├── users.py                  # User profiles + session scripts for all 5 users
├── requirements.txt
│
├── memory/
│   ├── memory_store.py       # SQLite: episodic + semantic + procedural + metrics
│   ├── distiller.py          # LLM-based fact extraction with importance scoring
│   └── context_manager.py   # Sliding window + LLM summarization for long sessions
│
├── prompts/
│   └── templates.py          # Versioned prompts, intent detection, tool selection
│
├── evaluation/
│   └── evaluator.py          # Tier 1/2/3 evaluation framework
│
├── results/
│   ├── evaluation_results.txt # Full eval report across 5 sessions
│   ├── TRANSCRIPTS.md
│   └── transcripts/
│       └── priya_sharma/     # Session 1–5 transcript files
└──
```

---

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=o4-mini   # or gpt-4o-mini
```

### Run the sessions

```bash
# Session 1 (CURRENT_SESSION=1 in tools.py — default)
python agent.py --session 1 --user priya_sharma

# Session 2 (change CURRENT_SESSION=2 in tools.py first)
python agent.py --session 2 --user priya_sharma
```

### Run extended sessions (3–5)

```bash
python agent.py --session 3 --user priya_sharma
python agent.py --session 4 --user priya_sharma
python agent.py --session 5 --user priya_sharma
```

### Run all sessions in one go
```bash
python simulate_users.py --users priya_sharma --sessions 1 2 3 4 5
```

### Interactive mode

```bash
python agent.py --session 1 --interactive
# Commands: 'memory', 'cache', 'metrics', 'quit'
```

### Evaluation

```bash
python evaluation/evaluator.py --all-sessions --user priya_sharma
python evaluation/evaluator.py --production --user priya_sharma
```

---

## Evaluation Results (Priya Sharma — 5 Sessions)

| Session | Score | Grade | Key Result |
|---------|-------|-------|------------|
| 1 | **99.2%** | ✅ PASS | All 4 tools called, reminder set, food delivery identified |
| 2 | **97.6%** | ✅ PASS | Session 1 memory recalled, live balance fetched, goal linked |
| 3 | **93.7%** | ✅ PASS | SIP compound math correct, reminder set |
| 4 | **90.5%** | ✅ PASS | ELSS/PPF advice correctly linked to 2yr house goal |
| 5 | **98.8%** | ✅ PASS | Progress tracked, Goa trip affordability assessed |

**Production metrics:** P95 latency 41,986ms | Cost/session $0.042 | Retry rate 0%

---

## Model Notes

The `MODEL` env var selects the LLM. The agent handles both families correctly:

```bash
# Chat model: streaming, temperature=0, max_tokens
OPENAI_MODEL=gpt-4o-mini

# Reasoning model: non-streaming, no temperature, max_completion_tokens (30× multiplier)
OPENAI_MODEL=o4-mini
```