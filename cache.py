"""
cache.py — Three-Tier In-Process Cache

Production design:
  L1: Exact match cache  — Redis hash(query), TTL 5min  → eliminates identical repeated calls
  L2: Semantic cache     — Vector similarity > 0.92, TTL 10min → handles paraphrased repeats
  L3: Tool result cache  — Redis per-session, TTL 2min → reuses tool results within session

For production swap: replace _store dicts with Redis client calls.
  L1/L3 → redis.setex(key, ttl, value) / redis.get(key)
  L2 → Vector DB cosine search (Qdrant / pgvector)

This in-process implementation is functionally identical for single-pod testing.

Design principle:
  "Three-tier caching eliminates 30-50% of LLM calls at scale."
"""

import hashlib
import json
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional


# ─── Cache Config ─────────────────────────────────────────────────────────────

L1_TTL_SECONDS = 300     # 5 min  — exact response cache
L2_TTL_SECONDS = 600     # 10 min — semantic response cache
L3_TTL_SECONDS = 120     # 2 min  — tool result cache (session-scoped)
L2_SIMILARITY_THRESHOLD = 0.92   # cosine similarity threshold for L2 hit


# ─── Cache Entry ──────────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    value: Any
    created_at: float = field(default_factory=time.time)
    embedding: Optional[list] = None  # for L2 semantic cache
    hits: int = 0

    def is_expired(self, ttl: float) -> bool:
        return (time.time() - self.created_at) > ttl


# ─── Cache Statistics ─────────────────────────────────────────────────────────

@dataclass
class CacheStats:
    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    l3_hits: int = 0
    l3_misses: int = 0

    @property
    def total_hits(self) -> int:
        return self.l1_hits + self.l2_hits + self.l3_hits

    @property
    def total_requests(self) -> int:
        return self.total_hits + self.l1_misses + self.l2_misses + self.l3_misses

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_hits / self.total_requests

    def to_dict(self) -> dict:
        return {
            "l1_hits": self.l1_hits, "l1_misses": self.l1_misses,
            "l2_hits": self.l2_hits, "l2_misses": self.l2_misses,
            "l3_hits": self.l3_hits, "l3_misses": self.l3_misses,
            "total_hit_rate": round(self.hit_rate, 3),
        }


# ─── Global Cache Stores ──────────────────────────────────────────────────────
# In prod: replace with Redis client. Keys and TTLs remain identical.

_l1_store: dict[str, CacheEntry] = {}   # exact response cache
_l2_store: dict[str, CacheEntry] = {}   # semantic response cache (with embeddings)
_l3_store: dict[str, CacheEntry] = {}   # tool result cache (session-scoped key)

_stats = CacheStats()


# ─── Key Helpers ──────────────────────────────────────────────────────────────

def _hash_key(data: str) -> str:
    """Stable SHA-256 hash for cache key. Identical to Redis hash key pattern."""
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def _tool_cache_key(session_id: str, tool_name: str, args: dict) -> str:
    payload = f"{session_id}:{tool_name}:{json.dumps(args, sort_keys=True)}"
    return _hash_key(payload)


def _response_cache_key(user_id: str, message: str) -> str:
    payload = f"{user_id}:{message.lower().strip()}"
    return _hash_key(payload)


# ─── L1: Exact Response Cache ─────────────────────────────────────────────────

def l1_get(user_id: str, message: str) -> Optional[str]:
    """Check L1 exact cache for agent response."""
    _purge_expired(_l1_store, L1_TTL_SECONDS)
    key = _response_cache_key(user_id, message)
    entry = _l1_store.get(key)
    if entry and not entry.is_expired(L1_TTL_SECONDS):
        entry.hits += 1
        _stats.l1_hits += 1
        return entry.value
    _stats.l1_misses += 1
    return None


def l1_set(user_id: str, message: str, response: str) -> None:
    """Store agent response in L1 cache."""
    key = _response_cache_key(user_id, message)
    _l1_store[key] = CacheEntry(value=response)


# ─── L2: Semantic Response Cache ─────────────────────────────────────────────

def _cosine_similarity(a: list, b: list) -> float:
    """numpy cosine similarity between two embedding vectors."""
    va, vb = np.array(a), np.array(b)
    norm_a, norm_b = np.linalg.norm(va), np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def l2_get(query_embedding: list) -> Optional[str]:
    """Check L2 semantic cache by cosine similarity (threshold=0.92)."""
    _purge_expired(_l2_store, L2_TTL_SECONDS)
    best_score = 0.0
    best_value = None
    for key, entry in _l2_store.items():
        if entry.is_expired(L2_TTL_SECONDS) or entry.embedding is None:
            continue
        score = _cosine_similarity(query_embedding, entry.embedding)
        if score > best_score:
            best_score = score
            best_value = entry
    if best_score >= L2_SIMILARITY_THRESHOLD and best_value:
        best_value.hits += 1
        _stats.l2_hits += 1
        return best_value.value
    _stats.l2_misses += 1
    return None


def l2_set(query_embedding: list, response: str) -> None:
    """Store agent response in L2 semantic cache with embedding."""
    key = _hash_key(str(query_embedding[:5]))  # partial embedding as key
    _l2_store[key] = CacheEntry(value=response, embedding=query_embedding)


# ─── L3: Tool Result Cache (per session) ─────────────────────────────────────

def l3_get(session_id: str, tool_name: str, args: dict) -> Optional[dict]:
    """Check L3 tool result cache. Prevents duplicate tool calls within session."""
    _purge_expired(_l3_store, L3_TTL_SECONDS)
    key = _tool_cache_key(session_id, tool_name, args)
    entry = _l3_store.get(key)
    if entry and not entry.is_expired(L3_TTL_SECONDS):
        entry.hits += 1
        _stats.l3_hits += 1
        return entry.value
    _stats.l3_misses += 1
    return None


def l3_set(session_id: str, tool_name: str, args: dict, result: dict) -> None:
    """Cache tool result for the duration of the session."""
    key = _tool_cache_key(session_id, tool_name, args)
    _l3_store[key] = CacheEntry(value=result)


def l3_evict_session(session_id: str) -> None:
    """Clear all L3 cache entries for a completed session."""
    prefix = session_id + ":"
    keys_to_delete = [k for k in _l3_store if prefix in k]
    for k in keys_to_delete:
        del _l3_store[k]


# ─── Per-Turn Tool Cache (ephemeral, cleared each agent turn) ─────────────────

class TurnCache:
    """
    Ephemeral cache for a single ReAct loop turn.
    Deduplicates identical tool calls within the same agent loop iteration.
    Cleared after each turn — not persisted.
    """
    def __init__(self):
        self._store: dict[str, dict] = {}

    def get(self, tool_name: str, args: dict) -> Optional[dict]:
        key = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
        return self._store.get(key)

    def set(self, tool_name: str, args: dict, result: dict) -> None:
        key = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
        self._store[key] = result

    def clear(self) -> None:
        self._store.clear()


# ─── Utilities ────────────────────────────────────────────────────────────────

def _purge_expired(store: dict, ttl: float) -> None:
    """Remove expired entries. In Redis: this is handled automatically by TTL."""
    expired = [k for k, v in store.items() if v.is_expired(ttl)]
    for k in expired:
        del store[k]


def get_cache_stats() -> dict:
    """Return cache performance statistics for monitoring."""
    return {
        **_stats.to_dict(),
        "l1_size": len(_l1_store),
        "l2_size": len(_l2_store),
        "l3_size": len(_l3_store),
    }


def clear_all_caches() -> None:
    """Full cache flush — use for testing or emergency cache invalidation."""
    _l1_store.clear()
    _l2_store.clear()
    _l3_store.clear()
