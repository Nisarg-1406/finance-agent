"""
simulate_users.py — Multi-User Concurrent Simulation Runner

Usage:
  python simulate_users.py                          # all users, all sessions
  python simulate_users.py --users priya arjun      # specific users
  python simulate_users.py --user priya --sessions 1 2 3
  python simulate_users.py --parallel               # run users concurrently (async)
  python simulate_users.py --evaluate               # run eval suite after simulation
  python simulate_users.py --report                 # print metrics report only (no run)

Tool overrides: per-user financial snapshots injected into tool execution
so each user has realistic unique data without modifying tools.py stubs.
"""

import asyncio
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.memory_store import init_db, get_production_metrics, get_all_facts
from agent import run_session_async
from tools import set_session_context
from users import ALL_USERS, USER_TOOL_OVERRIDES, SESSION_DATES

# ─── Output directories ───────────────────────────────────────────────────────

TRANSCRIPTS_DIR = Path("transcripts")
REPORTS_DIR = Path("reports")
TRANSCRIPTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


# ─── Tool Override Injection ──────────────────────────────────────────────────

def build_tool_overrides(user_id: str, session_id: str) -> Optional[dict]:
    """
    Build per-user tool override dict.
    Makes get_account_balance() and get_upcoming_bills() return user-specific data.
    Priya Sharma uses original tools.py data (no override needed).
    """
    if user_id == "priya_sharma":
        return None  # use original stub data

    overrides = USER_TOOL_OVERRIDES.get(user_id, {})
    if not overrides:
        return None

    # Build callable overrides that match tools.py signatures
    result = {}

    if "get_account_balance" in overrides:
        balance_data = overrides["get_account_balance"]
        result["get_account_balance"] = lambda: balance_data

    if "get_upcoming_bills" in overrides:
        bills_data = overrides["get_upcoming_bills"]
        result["get_upcoming_bills"] = lambda days=30: bills_data

    return result or None


# ─── Transcript Writing ───────────────────────────────────────────────────────

def save_transcript(user_id: str, session_id: str, transcript_turns: list[dict]) -> None:
    """Save session transcript to transcripts/{user_id}/session_{n}.txt"""
    user_dir = TRANSCRIPTS_DIR / user_id
    user_dir.mkdir(exist_ok=True)

    path = user_dir / f"session_{session_id}.txt"
    lines = [
        f"Session {session_id} — {SESSION_DATES.get(session_id, '')}",
        f"User: {user_id}",
        "=" * 60,
    ]
    for turn in transcript_turns:
        lines.append(f"\nTurn {turn['turn']}:")
        lines.append(f"  User: {turn['user']}")
        lines.append(f"  Agent: {turn['agent']}")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  📄 Transcript saved: {path}")


# ─── Single User Runner ───────────────────────────────────────────────────────

async def run_user_sessions(
    user_id: str,
    sessions_to_run: list[str],
    verbose: bool = False,
) -> dict:
    """Run all specified sessions for a single user, sequentially."""
    user_data = ALL_USERS[user_id]
    profile = user_data["profile"]
    all_sessions = user_data["sessions"]

    print(f"\n{'█'*60}")
    print(f"  USER: {profile['name']} ({profile['city']}, ₹{profile['monthly_income_inr']:,}/mo)")
    print(f"  Goal: {profile['stated_goal']}")
    print(f"  Running sessions: {sessions_to_run}")
    print(f"{'█'*60}")

    user_results = []

    for session_id in sessions_to_run:
        if session_id not in all_sessions:
            print(f"  [SKIP] Session {session_id} not defined for {user_id}")
            continue

        messages = all_sessions[session_id]
        tool_overrides = build_tool_overrides(user_id, session_id)

        # For Priya's sessions 1-2, use the same CURRENT_SESSION flag mechanism
        session_int = int(session_id)
        ctx_session = session_int if session_id in ("1", "2") else 1

        try:
            with set_session_context(ctx_session):
                result = await run_session_async(
                    session_id=session_id,
                    messages=messages,
                    user_id=user_id,
                    profile=profile,
                    verbose=verbose,
                    tool_overrides=tool_overrides,
                )

            # Save transcript
            save_transcript(user_id, session_id, result.get("turns", []))
            user_results.append(result)

        except Exception as e:
            print(f"  [ERROR] Session {session_id} failed: {e}")
            import traceback
            traceback.print_exc()

    # Per-user production metrics
    prod_metrics = get_production_metrics(user_id)
    return {
        "user_id": user_id,
        "profile": profile,
        "sessions_run": len(user_results),
        "production_metrics": prod_metrics,
        "session_results": user_results,
    }


# ─── Parallel Multi-User Runner ───────────────────────────────────────────────

async def run_all_users_parallel(
    user_ids: list[str],
    sessions_to_run: list[str],
    verbose: bool = False,
) -> list[dict]:
    """
    Run multiple users concurrently using asyncio.gather.
    Each user's sessions run sequentially within the user's coroutine,
    but different users run in parallel.

    Production note: This simulates concurrent users on the same pod.
    At scale: distribute users across pods via a queue (Celery, SQS, Kafka).
    """
    print(f"\n🚀 Running {len(user_ids)} users in parallel...")
    tasks = [
        run_user_sessions(uid, sessions_to_run, verbose=verbose)
        for uid in user_ids
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Surface any exceptions
    clean_results = []
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            print(f"  [ERROR] User {user_ids[i]}: {res}")
        else:
            clean_results.append(res)

    return clean_results


# ─── Aggregate Report ─────────────────────────────────────────────────────────

def print_aggregate_report(all_results: list[dict]) -> None:
    """Print a consolidated multi-user performance report."""
    print(f"\n\n{'═'*65}")
    print(f"  MULTI-USER SIMULATION REPORT")
    print(f"  Users: {len(all_results)} | Sessions: {sum(r.get('sessions_run', 0) for r in all_results)}")
    print(f"{'═'*65}")
    print(f"  {'User':<20} {'Sessions':>8} {'P95ms':>7} {'Cost/s':>8} {'Facts':>6}")
    print(f"  {'─'*20} {'─'*8} {'─'*7} {'─'*8} {'─'*6}")

    total_sessions = 0
    total_cost = 0.0

    for res in all_results:
        if not isinstance(res, dict):
            continue
        user_id = res["user_id"]
        sessions_run = res.get("sessions_run", 0)
        pm = res.get("production_metrics", {})
        p95 = pm.get("p95_latency_ms", 0)
        cost = pm.get("cost_per_session_usd", 0)
        facts = len(get_all_facts(user_id))

        total_sessions += sessions_run
        total_cost += pm.get("total_cost_usd", 0)

        print(f"  {user_id:<20} {sessions_run:>8} {p95:>7.0f} {cost:>8.6f} {facts:>6}")

    print(f"{'─'*65}")
    print(f"  {'TOTAL':<20} {total_sessions:>8} {'---':>7} {total_cost:>8.6f}")
    print(f"{'═'*65}\n")


def save_aggregate_report(all_results: list[dict]) -> None:
    """Save JSON report to reports/ directory."""
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_users": len(all_results),
        "total_sessions": sum(r.get("sessions_run", 0) for r in all_results if isinstance(r, dict)),
        "results": [r for r in all_results if isinstance(r, dict)],
    }
    path = REPORTS_DIR / f"simulation_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, default=str, indent=2), encoding="utf-8")
    print(f"  📊 Report saved: {path}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

async def _main_async():
    parser = argparse.ArgumentParser(description="Multi-User Finance Agent Simulation")
    parser.add_argument(
        "--users", nargs="+",
        choices=["all"] + list(ALL_USERS.keys()) + ["priya", "arjun", "sneha"],
        default=["all"],
        help="Users to simulate (default: all)",
    )
    parser.add_argument(
        "--sessions", nargs="+",
        choices=[str(i) for i in range(1, 11)],
        default=["1", "2"],
        help="Session numbers to run (default: 1 2)",
    )
    parser.add_argument("--parallel", action="store_true",
                        help="Run users concurrently (async)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation suite after simulation")
    parser.add_argument("--report", action="store_true",
                        help="Print metrics report only (no simulation)")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose tool call output")
    args = parser.parse_args()

    init_db()

    # Resolve user IDs
    user_short_map = {
        "priya": "priya_sharma", "arjun": "arjun_mehta", "sneha": "sneha_patel"
    }
    if "all" in args.users:
        user_ids = list(ALL_USERS.keys())
    else:
        user_ids = [user_short_map.get(u, u) for u in args.users]
        user_ids = [uid for uid in user_ids if uid in ALL_USERS]

    sessions_to_run = sorted(set(args.sessions), key=int)

    # Report-only mode
    if args.report:
        print(f"\n{'═'*65}")
        print("  CURRENT PRODUCTION METRICS")
        print(f"{'═'*65}")
        for uid in user_ids:
            pm = get_production_metrics(uid)
            profile = ALL_USERS[uid]["profile"]
            facts = len(get_all_facts(uid))
            print(f"  {profile['name']:<20} sessions={pm.get('sessions', 0)} | "
                  f"p95={pm.get('p95_latency_ms', 0):.0f}ms | "
                  f"cost=${pm.get('total_cost_usd', 0):.6f} | "
                  f"facts={facts}")
        return

    print(f"\n{'━'*65}")
    print(f"  MULTI-USER SIMULATION")
    print(f"  Users: {user_ids}")
    print(f"  Sessions: {sessions_to_run}")
    print(f"  Mode: {'parallel' if args.parallel else 'sequential'}")
    print(f"{'━'*65}")

    start = time.time()

    if args.parallel:
        all_results = await run_all_users_parallel(user_ids, sessions_to_run, verbose=args.verbose)
    else:
        all_results = []
        for uid in user_ids:
            res = await run_user_sessions(uid, sessions_to_run, verbose=args.verbose)
            all_results.append(res)

    elapsed = time.time() - start

    print(f"\n✅ Simulation complete in {elapsed:.1f}s")
    print_aggregate_report(all_results)
    save_aggregate_report(all_results)

    # Optional evaluation
    if args.evaluate:
        print("\n🔬 Running evaluation suite...")
        from evaluation.evaluator import evaluate_session1, evaluate_session2, evaluate_production_metrics

        for uid in user_ids:
            if "1" in sessions_to_run:
                r1 = evaluate_session1(uid)
                print(r1.report())
            if "2" in sessions_to_run:
                r2 = evaluate_session2(uid)
                print(r2.report())

            prod_eval = evaluate_production_metrics(uid)
            report_path = REPORTS_DIR / f"{uid}_eval.json"
            report_path.write_text(json.dumps(prod_eval, indent=2), encoding="utf-8")
            print(f"  📊 Eval report: {report_path}")


if __name__ == "__main__":
    asyncio.run(_main_async())
