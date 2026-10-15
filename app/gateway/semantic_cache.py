"""
semantic_cache.py — C++17 AVX2 similarity lookup + response store.
Uses SIMD cache_lookup for sub-microsecond vector matching.
Supports Redis response store with in-memory fallback.
"""
import numpy as np
import time
import logging
import hashlib
import json
import threading
from typing import Optional, Tuple
from dataclasses import dataclass

from ..config import (
    CACHE_SIMILARITY_THRESHOLD,
    CACHE_MAX_ENTRIES,
    CACHE_TTL_SECONDS,
    REDIS_ENABLED,
    REDIS_URL,
)

logger = logging.getLogger(__name__)

# Try importing SIMD engine
try:
    import simd_engine
    SIMD_AVAILABLE = True
except ImportError:
    SIMD_AVAILABLE = False


def is_negative_or_refusal_response(text: str) -> bool:
    """
    Check if LLM response is a refusal, apology, or negative response that must NEVER be cached.
    Prevents cache poisoning from LLM apologies or empty context responses.
    """
    if not text or not text.strip():
        return True
    lower = text.lower()
    refusal_phrases = [
        "i'm sorry", "i’m sorry", "i am sorry",
        "i don't have", "i’t have", "i do not have",
        "don't have access", "do not have access",
        "i cannot answer", "i can't answer",
        "not mentioned in the provided context",
        "not provided in the context",
        "the provided text does not contain",
        "could you clarify", "can you clarify",
        "please clarify",
        "i apologize", "apologies",
        "no information provided",
    ]
    return any(p in lower for p in refusal_phrases)


@dataclass
class CacheEntry:
    """A cached query-response pair."""
    query_text: str
    response_text: str
    embedding: np.ndarray
    timestamp: float
    hit_count: int = 0
    model_used: str = ""
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CacheLookupResult:
    """Result of a cache lookup."""
    hit: bool
    response: Optional[str] = None
    similarity_score: float = 0.0
    cache_key: str = ""
    lookup_latency_ms: float = 0.0


class SemanticCache:
    """
    High-performance semantic cache using C++17 AVX2 SIMD for vector matching.
    Stores response text in-memory (or Redis) keyed by embedding similarity.
    """

    def __init__(
        self,
        similarity_threshold: float = CACHE_SIMILARITY_THRESHOLD,
        max_entries: int = CACHE_MAX_ENTRIES,
        ttl_seconds: int = CACHE_TTL_SECONDS,
    ):
        self._threshold = similarity_threshold
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

        # In-memory stores
        self._entries: list = []  # List[CacheEntry]
        self._embedding_matrix: Optional[np.ndarray] = None  # shape: (n, dim)

        # Redis store (optional)
        self._redis = None
        if REDIS_ENABLED:
            try:
                import redis as redis_lib
                self._redis = redis_lib.from_url(REDIS_URL)
                self._redis.ping()
                logger.info("Redis connected for cache response store")
            except Exception as e:
                logger.warning(f"Redis not available: {e}, using in-memory store")
                self._redis = None

        # Stats
        self._total_lookups = 0
        self._total_hits = 0
        self._total_misses = 0
        self._total_stores = 0
        self._total_lookup_latency_ms = 0.0

    def lookup(self, query_embedding: np.ndarray) -> CacheLookupResult:
        """
        Look up a query embedding in the cache using AVX2 SIMD.

        Args:
            query_embedding: L2-normalized query vector, shape (dim,).

        Returns:
            CacheLookupResult with hit status and cached response if found.
        """
        start = time.perf_counter()
        self._total_lookups += 1

        with self._lock:
            if self._embedding_matrix is None or len(self._entries) == 0:
                elapsed = (time.perf_counter() - start) * 1000.0
                self._total_misses += 1
                self._total_lookup_latency_ms += elapsed
                return CacheLookupResult(hit=False, lookup_latency_ms=elapsed)

            query_vec = np.ascontiguousarray(query_embedding.flatten(), dtype=np.float32)

            if SIMD_AVAILABLE:
                result = simd_engine.cache_lookup(
                    query_vec,
                    self._embedding_matrix,
                    self._threshold,
                )
                best_idx = result.best_index
                best_score = result.best_score
                is_hit = result.hit
            else:
                # NumPy fallback
                similarities = self._embedding_matrix @ query_vec
                best_idx = int(np.argmax(similarities))
                best_score = float(similarities[best_idx])
                is_hit = best_score >= self._threshold

            elapsed = (time.perf_counter() - start) * 1000.0
            self._total_lookup_latency_ms += elapsed

            if is_hit and 0 <= best_idx < len(self._entries):
                entry = self._entries[best_idx]

                # Check TTL
                if time.time() - entry.timestamp > self._ttl:
                    self._total_misses += 1
                    return CacheLookupResult(hit=False, similarity_score=best_score, lookup_latency_ms=elapsed)

                entry.hit_count += 1
                self._total_hits += 1

                return CacheLookupResult(
                    hit=True,
                    response=entry.response_text,
                    similarity_score=best_score,
                    cache_key=f"cache_{best_idx}",
                    lookup_latency_ms=elapsed,
                )
            else:
                self._total_misses += 1
                return CacheLookupResult(
                    hit=False,
                    similarity_score=best_score if best_idx >= 0 else 0.0,
                    lookup_latency_ms=elapsed,
                )

    def store(
        self,
        query_text: str,
        response_text: str,
        query_embedding: np.ndarray,
        model_used: str = "",
        metadata: dict = None,
    ):
        """
        Store a query-response pair in the cache.

        Args:
            query_text: Original query text.
            response_text: LLM response text.
            query_embedding: L2-normalized query vector.
            model_used: Which LLM model generated the response.
            metadata: Optional metadata dict.
        """
        if is_negative_or_refusal_response(response_text):
            logger.warning(
                f"Negative/refusal response rejected from semantic cache for query: '{query_text[:50]}...'"
            )
            return

        with self._lock:
            embedding = np.ascontiguousarray(query_embedding.flatten(), dtype=np.float32)

            entry = CacheEntry(
                query_text=query_text,
                response_text=response_text,
                embedding=embedding,
                timestamp=time.time(),
                model_used=model_used,
                metadata=metadata or {},
            )

            self._entries.append(entry)
            self._total_stores += 1

            # Rebuild embedding matrix
            all_embeddings = [e.embedding for e in self._entries]
            self._embedding_matrix = np.vstack(all_embeddings).astype(np.float32)
            self._embedding_matrix = np.ascontiguousarray(self._embedding_matrix)

            # Evict oldest if over capacity
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries:]
                all_embeddings = [e.embedding for e in self._entries]
                self._embedding_matrix = np.vstack(all_embeddings).astype(np.float32)
                self._embedding_matrix = np.ascontiguousarray(self._embedding_matrix)

            # Also store in Redis if available
            if self._redis:
                try:
                    cache_key = hashlib.sha256(embedding.tobytes()).hexdigest()[:16]
                    self._redis.setex(
                        f"sentinelai:cache:{cache_key}",
                        self._ttl,
                        json.dumps({
                            "query": query_text,
                            "response": response_text,
                            "model": model_used,
                        }),
                    )
                except Exception as e:
                    logger.warning(f"Redis store failed: {e}")

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._entries.clear()
            self._embedding_matrix = None

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def stats(self) -> dict:
        total = max(self._total_lookups, 1)
        avg_lookup = self._total_lookup_latency_ms / total
        return {
            "total_lookups": self._total_lookups,
            "total_hits": self._total_hits,
            "total_misses": self._total_misses,
            "hit_rate": round(self._total_hits / total, 4),
            "total_stores": self._total_stores,
            "current_size": len(self._entries),
            "avg_lookup_latency_ms": round(avg_lookup, 4),
            "simd_available": SIMD_AVAILABLE,
        }
