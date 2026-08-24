"""
memory/memory_store.py — Hybrid Episodic + Semantic + Procedural Memory

Architecture:
  - Episodic  : SQLite — raw turn-by-turn events per session
  - Semantic  : SQLite + embedding vectors (numpy cosine) — distilled facts
  - Procedural: SQLite JSON — learned behaviors from feedback
  - Metrics   : SQLite — per-turn latency, cost, cache stats

Production upgrade notes:
  - Embedding-based effective retrieval: facts ranked by cosine similarity
    to current query (reranking principle from guide §2)
  - TTL-aware confidence decay: older facts surface lower in context
  - Memory pruning: removes lowest-confidence facts beyond MAX_FACTS cap
  - Sliding window episodic: only last N turns loaded per turn (not full session)

PostgreSQL swap (one function change):
  Replace _conn() with:
    import psycopg2
    def _conn():
        return psycopg2.connect(os.getenv("POSTGRES_DSN"))
  All SQL is ANSI-compatible except AUTOINCREMENT → SERIAL.
  pgvector: store embedding as vector(1536) column for native ANN search.
"""

import json
import sqlite3
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path(__file__).parent.parent / "memory.db"

# Memory pruning: max stored facts per user before lowest-confidence are dropped
MAX_FACTS = 50
MIN_CONFIDENCE_THRESHOLD = 0.3   # facts below this are eligible for pruning
CONFIDENCE_DECAY_PER_WEEK = 0.05  # multiply confidence × (1 - decay) per week old


# ─── Connection ───────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    """
    SQLite connection with WAL mode for better concurrency.

    Production swap: replace with psycopg2.connect(os.getenv("POSTGRES_DSN"))
    or SQLAlchemy engine. WAL → not needed; PostgreSQL handles concurrency natively.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ─── Schema Init ──────────────────────────────────────────────────────────────

def init_db() -> None:
    """Initialize all tables. Idempotent — safe to call on every startup."""
    with _conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS episodic (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     TEXT    NOT NULL,
            session_id  TEXT    NOT NULL,
            turn        INTEGER NOT NULL,
            role        TEXT    NOT NULL,
            content     TEXT    NOT NULL,
            tool_calls  TEXT,
            ts          TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_episodic_user_session
            ON episodic(user_id, session_id);

        CREATE TABLE IF NOT EXISTS semantic (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     TEXT    NOT NULL,
            key         TEXT    NOT NULL,
            value       TEXT    NOT NULL,
            confidence  REAL    DEFAULT 1.0,
            importance  INTEGER DEFAULT 3,
            source      TEXT,
            embedding   TEXT,
            ts          TEXT    NOT NULL,
            UNIQUE(user_id, key)
        );
        CREATE INDEX IF NOT EXISTS idx_semantic_user ON semantic(user_id);

        CREATE TABLE IF NOT EXISTS procedural (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     TEXT    NOT NULL,
            pattern     TEXT    NOT NULL,
            value       TEXT    NOT NULL,
            ts          TEXT    NOT NULL,
            UNIQUE(user_id, pattern)
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     TEXT    NOT NULL,
            session_id  TEXT    NOT NULL,
            turn        INTEGER NOT NULL,
            rating      INTEGER,
            comment     TEXT,
            ts          TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS metrics (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         TEXT    NOT NULL,
            session_id      TEXT    NOT NULL,
            turn            INTEGER NOT NULL,
            ttft_ms         REAL,
            tpot_ms         REAL,
            total_ms        REAL,
            input_tokens    INTEGER DEFAULT 0,
            output_tokens   INTEGER DEFAULT 0,
            tool_calls_count INTEGER DEFAULT 0,
            cost_usd        REAL    DEFAULT 0.0,
            cache_hit_l1    INTEGER DEFAULT 0,
            cache_hit_l2    INTEGER DEFAULT 0,
            cache_hit_l3    INTEGER DEFAULT 0,
            retry_count     INTEGER DEFAULT 0,
            ts              TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_metrics_user ON metrics(user_id, session_id);

        CREATE TABLE IF NOT EXISTS hallucination_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     TEXT    NOT NULL,
            session_id  TEXT    NOT NULL,
            turn        INTEGER NOT NULL,
            claim       TEXT,
            grounded    INTEGER DEFAULT 0,
            evidence    TEXT,
            ts          TEXT    NOT NULL
        );
        """)


# ─── Episodic Memory ──────────────────────────────────────────────────────────

def save_turn(user_id: str, session_id: str, turn: int, role: str,
              content: str, tool_calls: list | None = None) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO episodic (user_id, session_id, turn, role, content, tool_calls, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, session_id, turn, role, content,
             json.dumps(tool_calls) if tool_calls else None,
             datetime.now(timezone.utc).isoformat())
        )


def get_session_turns(user_id: str, session_id: str) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT role, content, tool_calls, ts FROM episodic "
            "WHERE user_id=? AND session_id=? ORDER BY turn",
            (user_id, session_id)
        ).fetchall()
    return [dict(r) for r in rows]


def get_recent_turns(user_id: str, session_id: str, last_n_turns: int = 8) -> list[dict]:
    """Sliding window: fetch only the last N turns (not full session)."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT role, content, tool_calls, ts, turn FROM episodic "
            "WHERE user_id=? AND session_id=? ORDER BY turn DESC LIMIT ?",
            (user_id, session_id, last_n_turns * 2)  # × 2 for user+agent pairs
        ).fetchall()
    return list(reversed([dict(r) for r in rows]))


def get_all_sessions(user_id: str) -> list[str]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT session_id FROM episodic WHERE user_id=? ORDER BY ts",
            (user_id,)
        ).fetchall()
    return [r["session_id"] for r in rows]


def get_session_count(user_id: str) -> int:
    with _conn() as conn:
        row = conn.execute(
            "SELECT COUNT(DISTINCT session_id) as n FROM episodic WHERE user_id=?",
            (user_id,)
        ).fetchone()
    return row["n"] if row else 0


# ─── Semantic Memory ──────────────────────────────────────────────────────────

def upsert_fact(user_id: str, key: str, value: Any,
                confidence: float = 1.0,
                source: str = "",
                importance: int = 3,
                embedding: Optional[list] = None) -> None:
    """
    Store or update a semantic fact.
    importance: 1-5 scale (distiller assigns this; facts < 3 are not stored).
    embedding: list of floats from text-embedding-3-small (1536 dims).
    """
    with _conn() as conn:
        conn.execute(
            "INSERT INTO semantic (user_id, key, value, confidence, importance, source, embedding, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id, key) DO UPDATE SET "
            "value=excluded.value, confidence=excluded.confidence, "
            "importance=excluded.importance, source=excluded.source, "
            "embedding=excluded.embedding, ts=excluded.ts",
            (user_id, key, json.dumps(value), confidence, importance, source,
             json.dumps(embedding) if embedding else None,
             datetime.now(timezone.utc).isoformat())
        )


def get_fact(user_id: str, key: str) -> Any | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT value FROM semantic WHERE user_id=? AND key=?",
            (user_id, key)
        ).fetchone()
    return json.loads(row["value"]) if row else None


def get_all_facts(user_id: str) -> dict[str, Any]:
    """Returns all facts with TTL-adjusted confidence."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT key, value, confidence, importance, source, ts FROM semantic WHERE user_id=?",
            (user_id,)
        ).fetchall()

    result = {}
    now = datetime.now(timezone.utc)
    for r in rows:
        # Confidence decay: reduce by CONFIDENCE_DECAY_PER_WEEK per week old
        try:
            ts = datetime.fromisoformat(r["ts"])
            weeks_old = (now - ts).days / 7
            decayed_conf = r["confidence"] * ((1 - CONFIDENCE_DECAY_PER_WEEK) ** weeks_old)
            decayed_conf = max(0.05, round(decayed_conf, 3))
        except Exception:
            decayed_conf = r["confidence"]

        result[r["key"]] = {
            "value": json.loads(r["value"]),
            "confidence": decayed_conf,
            "raw_confidence": r["confidence"],
            "importance": r["importance"],
            "source": r["source"],
            "ts": r["ts"],
        }
    return result


def get_relevant_facts(user_id: str, query_embedding: Optional[list],
                       top_k: int = 10) -> dict[str, Any]:
    """
    Retrieve top-k most relevant facts via cosine similarity (reranking).
    Falls back to confidence-sorted order if no embeddings available.

    Production upgrade: replace numpy cosine with pgvector ANN search:
      SELECT key, value, confidence, embedding <=> $1 AS distance
      FROM semantic WHERE user_id=$2 ORDER BY distance LIMIT $3;
    """
    all_facts = get_all_facts(user_id)
    if not all_facts:
        return {}

    if query_embedding is None:
        # Fallback: sort by decayed confidence × importance
        ranked = sorted(
            all_facts.items(),
            key=lambda x: x[1]["confidence"] * x[1].get("importance", 3),
            reverse=True
        )
        return dict(ranked[:top_k])

    # Cosine similarity reranking
    q_vec = np.array(query_embedding)
    q_norm = np.linalg.norm(q_vec)

    scored = []
    with _conn() as conn:
        rows = conn.execute(
            "SELECT key, embedding FROM semantic WHERE user_id=? AND embedding IS NOT NULL",
            (user_id,)
        ).fetchall()

    embedded_keys = set()
    for r in rows:
        try:
            emb = json.loads(r["embedding"])
            f_vec = np.array(emb)
            f_norm = np.linalg.norm(f_vec)
            if q_norm > 0 and f_norm > 0:
                sim = float(np.dot(q_vec, f_vec) / (q_norm * f_norm))
            else:
                sim = 0.0
            fact_meta = all_facts.get(r["key"], {})
            # Combined score: semantic similarity + confidence + importance
            combined = (0.6 * sim) + (0.3 * fact_meta.get("confidence", 0.5)) + (0.1 * fact_meta.get("importance", 3) / 5)
            scored.append((r["key"], combined))
            embedded_keys.add(r["key"])
        except Exception:
            continue

    # Include non-embedded facts (sorted by confidence) if top_k not filled
    for key, meta in all_facts.items():
        if key not in embedded_keys:
            scored.append((key, meta["confidence"] * 0.5))  # lower priority

    scored.sort(key=lambda x: x[1], reverse=True)
    top_keys = [k for k, _ in scored[:top_k]]
    return {k: all_facts[k] for k in top_keys if k in all_facts}


def prune_old_facts(user_id: str, max_facts: int = MAX_FACTS) -> int:
    """
    Prune: delete lowest-confidence facts beyond max_facts cap.
    Returns count of deleted facts.
    Production: run as a scheduled job (not every turn).
    """
    all_facts = get_all_facts(user_id)
    if len(all_facts) <= max_facts:
        return 0

    # Sort by decayed_confidence × importance (ascending = worst first)
    sorted_facts = sorted(
        all_facts.items(),
        key=lambda x: x[1]["confidence"] * x[1].get("importance", 3)
    )
    # Delete the worst ones beyond cap
    to_delete = [k for k, v in sorted_facts[:len(all_facts) - max_facts]
                 if v["confidence"] < MIN_CONFIDENCE_THRESHOLD]

    if not to_delete:
        return 0

    with _conn() as conn:
        placeholders = ",".join(["?"] * len(to_delete))
        conn.execute(
            f"DELETE FROM semantic WHERE user_id=? AND key IN ({placeholders})",
            [user_id] + to_delete
        )
    return len(to_delete)


# ─── Procedural Memory ────────────────────────────────────────────────────────

def upsert_behavior(user_id: str, pattern: str, value: dict) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO procedural (user_id, pattern, value, ts) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(user_id, pattern) DO UPDATE SET value=excluded.value, ts=excluded.ts",
            (user_id, pattern, json.dumps(value),
             datetime.now(timezone.utc).isoformat())
        )


def get_behaviors(user_id: str) -> dict[str, dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT pattern, value FROM procedural WHERE user_id=?",
            (user_id,)
        ).fetchall()
    return {r["pattern"]: json.loads(r["value"]) for r in rows}


def get_active_behaviors(user_id: str) -> dict[str, dict]:
    """Return only active behavioral patterns (weight > 0.3)."""
    return {p: b for p, b in get_behaviors(user_id).items()
            if b.get("active", False) or b.get("weight", 0) > 0.3}


# ─── Feedback / Self-improvement ─────────────────────────────────────────────

def save_feedback(user_id: str, session_id: str, turn: int,
                  rating: int, comment: str = "") -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO feedback (user_id, session_id, turn, rating, comment, ts) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, session_id, turn, rating, comment,
             datetime.now(timezone.utc).isoformat())
        )
    _apply_feedback_to_procedural(user_id, rating, comment)


def _apply_feedback_to_procedural(user_id: str, rating: int, comment: str) -> None:
    """
    Three-layer feedback loop (Layer 1: Explicit).
    Extended keyword set aligned with production guide §10.
    """
    keywords = {
        "too long":    ("prefers_concise",    -0.25),
        "too short":   ("prefers_detailed",   +0.25),
        "numbers":     ("wants_numbers",      +0.30),
        "breakdown":   ("prefers_detailed",   +0.20),
        "remind":      ("likes_reminders",    +0.30),
        "simple":      ("prefers_concise",    +0.20),
        "confusing":   ("prefers_simple",     +0.25),
        "helpful":     ("is_engaged",         +0.15),
        "wrong":       ("needs_recalibration",+0.40),
        "goal":        ("goal_focused",       +0.20),
    }
    behaviors = get_behaviors(user_id)
    for kw, (pattern, delta) in keywords.items():
        if kw in comment.lower():
            current = behaviors.get(pattern, {"weight": 0.5, "active": True})
            # Rating amplifier: 1-2 rating = negative signal doubles the delta
            if rating <= 2:
                delta = -abs(delta)
            elif rating >= 4:
                delta = abs(delta)
            current["weight"] = max(0.0, min(1.0, current["weight"] + delta))
            current["active"] = current["weight"] > 0.3
            upsert_behavior(user_id, pattern, current)

# ─── Metrics Tracking ─────────────────────────────────────────────────────────

def log_turn_metrics(
    user_id: str, session_id: str, turn: int,
    ttft_ms: float = 0.0, tpot_ms: float = 0.0, total_ms: float = 0.0,
    input_tokens: int = 0, output_tokens: int = 0,
    tool_calls_count: int = 0, cost_usd: float = 0.0,
    cache_hit_l1: bool = False, cache_hit_l2: bool = False,
    cache_hit_l3: bool = False, retry_count: int = 0,
) -> None:
    """Persist per-turn production metrics for evaluation dashboard."""
    with _conn() as conn:
        conn.execute(
            "INSERT INTO metrics "
            "(user_id, session_id, turn, ttft_ms, tpot_ms, total_ms, "
            " input_tokens, output_tokens, tool_calls_count, cost_usd, "
            " cache_hit_l1, cache_hit_l2, cache_hit_l3, retry_count, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, session_id, turn, ttft_ms, tpot_ms, total_ms,
             input_tokens, output_tokens, tool_calls_count, cost_usd,
             int(cache_hit_l1), int(cache_hit_l2), int(cache_hit_l3),
             retry_count, datetime.now(timezone.utc).isoformat())
        )


def get_session_metrics(user_id: str, session_id: str) -> dict:
    """Aggregate metrics for a session — used by evaluation dashboard."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM metrics WHERE user_id=? AND session_id=? ORDER BY turn",
            (user_id, session_id)
        ).fetchall()

    if not rows:
        return {}

    rows_d = [dict(r) for r in rows]
    latencies = [r["total_ms"] for r in rows_d if r["total_ms"]]
    ttfts = [r["ttft_ms"] for r in rows_d if r["ttft_ms"]]

    cache_hits = sum(1 for r in rows_d if r["cache_hit_l1"] or r["cache_hit_l2"])
    total_cost = sum(r["cost_usd"] for r in rows_d)
    total_retries = sum(r["retry_count"] for r in rows_d)
    total_tool_calls = sum(r["tool_calls_count"] for r in rows_d)

    def _p(values, pct):
        if not values: return 0.0
        s = sorted(values)
        idx = int(len(s) * pct / 100)
        return s[min(idx, len(s) - 1)]

    return {
        "session_id": session_id,
        "turns": len(rows_d),
        "latency_p50_ms": round(_p(latencies, 50), 1),
        "latency_p95_ms": round(_p(latencies, 95), 1),
        "latency_p99_ms": round(_p(latencies, 99), 1),
        "ttft_avg_ms": round(sum(ttfts) / len(ttfts), 1) if ttfts else 0.0,
        "total_cost_usd": round(total_cost, 6),
        "total_tool_calls": total_tool_calls,
        "cache_hits": cache_hits,
        "cache_hit_rate": round(cache_hits / len(rows_d), 3) if rows_d else 0.0,
        "total_retries": total_retries,
        "retry_rate": round(total_retries / max(1, total_tool_calls), 3),
    }


def get_production_metrics(user_id: str) -> dict:
    """Aggregate all sessions for a user — production monitoring dashboard."""
    sessions = get_all_sessions(user_id)
    all_metrics = [get_session_metrics(user_id, sid) for sid in sessions]
    all_metrics = [m for m in all_metrics if m]

    if not all_metrics:
        return {"user_id": user_id, "sessions": 0}

    all_latencies_p95 = [m["latency_p95_ms"] for m in all_metrics]
    total_cost = sum(m["total_cost_usd"] for m in all_metrics)
    avg_cache_hit = sum(m["cache_hit_rate"] for m in all_metrics) / len(all_metrics)
    avg_retry = sum(m["retry_rate"] for m in all_metrics) / len(all_metrics)

    return {
        "user_id": user_id,
        "sessions": len(all_metrics),
        "p95_latency_ms": round(max(all_latencies_p95), 1) if all_latencies_p95 else 0.0,
        "total_cost_usd": round(total_cost, 6),
        "cost_per_session_usd": round(total_cost / len(all_metrics), 6),
        "avg_cache_hit_rate": round(avg_cache_hit, 3),
        "avg_retry_rate": round(avg_retry, 3),
        "per_session": all_metrics,
    }


# ─── Hallucination Logging ────────────────────────────────────────────────────

def log_hallucination(user_id: str, session_id: str, turn: int,
                      claim: str, grounded: bool, evidence: str = "") -> None:
    """Log a potential hallucination claim for evaluation."""
    with _conn() as conn:
        conn.execute(
            "INSERT INTO hallucination_log (user_id, session_id, turn, claim, grounded, evidence, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, session_id, turn, claim, int(grounded), evidence,
             datetime.now(timezone.utc).isoformat())
        )


def get_hallucination_rate(user_id: str, session_id: str) -> float:
    """Fraction of logged claims that were NOT grounded."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT grounded FROM hallucination_log WHERE user_id=? AND session_id=?",
            (user_id, session_id)
        ).fetchall()
    if not rows:
        return 0.0
    ungrounded = sum(1 for r in rows if not r["grounded"])
    return round(ungrounded / len(rows), 3)


# ─── Context Builder ──────────────────────────────────────────────────────────

def build_memory_context(user_id: str, query_embedding: Optional[list] = None,
                         top_k: int = 15) -> str:
    """
    Assemble compact memory context for system prompt injection.
    Uses embedding-based reranking when query_embedding is provided.
    Returns only facts with confidence > 0.3 (post-decay filter).

    Context hygiene: terse, distilled, not raw history dumps.
    """
    facts = get_relevant_facts(user_id, query_embedding, top_k=top_k)
    if not facts:
        return ""

    # Filter low-confidence facts (post-decay)
    facts = {k: v for k, v in facts.items() if v["confidence"] > 0.3}
    if not facts:
        return ""

    lines = ["## What I know about this user (from prior sessions)"]
    for key, meta in sorted(facts.items(), key=lambda x: -x[1]["confidence"]):
        v = meta["value"]
        conf = meta["confidence"]
        conf_marker = "✓" if conf >= 0.8 else "~"  # high vs moderate confidence
        lines.append(f"- {conf_marker} {key}: {v}")

    return "\n".join(lines)


def build_behavior_context(user_id: str) -> str:
    """Build behavioral preferences string for context injection."""
    behaviors = get_active_behaviors(user_id)
    if not behaviors:
        return ""
    active = [p for p, b in behaviors.items() if b.get("active")]
    if not active:
        return ""
    return "Active: " + ", ".join(active)
