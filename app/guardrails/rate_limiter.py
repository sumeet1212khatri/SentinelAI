"""
rate_limiter.py — Token-bucket rate limiter with Redis atomic Lua + in-memory fallback.
Provides per-client and global rate limiting with burst support.
"""
import time
import threading
from dataclasses import dataclass
from typing import Optional

# Redis Lua script for atomic token bucket
_LUA_TOKEN_BUCKET = """
local key = KEYS[1]
local max_tokens = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])

local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(bucket[1])
local last_refill = tonumber(bucket[2])

if tokens == nil then
    tokens = max_tokens
    last_refill = now
end

local elapsed = now - last_refill
local new_tokens = math.min(max_tokens, tokens + elapsed * refill_rate)

if new_tokens >= requested then
    new_tokens = new_tokens - requested
    redis.call('HMSET', key, 'tokens', new_tokens, 'last_refill', now)
    redis.call('EXPIRE', key, 120)
    return 1
else
    redis.call('HMSET', key, 'tokens', new_tokens, 'last_refill', now)
    redis.call('EXPIRE', key, 120)
    return 0
end
"""


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    remaining_tokens: float
    retry_after_seconds: float = 0.0
    client_id: str = ""


class InMemoryTokenBucket:
    """Thread-safe in-memory token bucket for single-instance deployments."""

    def __init__(self, max_tokens: float, refill_rate: float):
        """
        Args:
            max_tokens: Maximum tokens in the bucket (burst capacity).
            refill_rate: Tokens added per second.
        """
        self._max_tokens = max_tokens
        self._refill_rate = refill_rate
        self._tokens = max_tokens
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def try_consume(self, tokens: float = 1.0) -> RateLimitResult:
        """Attempt to consume tokens. Returns RateLimitResult."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(
                self._max_tokens,
                self._tokens + elapsed * self._refill_rate
            )
            self._last_refill = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                return RateLimitResult(
                    allowed=True,
                    remaining_tokens=self._tokens,
                )
            else:
                deficit = tokens - self._tokens
                retry_after = deficit / self._refill_rate
                return RateLimitResult(
                    allowed=False,
                    remaining_tokens=self._tokens,
                    retry_after_seconds=retry_after,
                )


class RateLimiter:
    """
    Rate limiter with Redis atomic Lua token-bucket and in-memory fallback.
    Supports per-client and global limiting.
    """

    def __init__(
        self,
        max_requests_per_minute: int = 60,
        burst: int = 10,
        redis_client=None,
    ):
        self._rpm = max_requests_per_minute
        self._burst = burst
        self._refill_rate = max_requests_per_minute / 60.0  # tokens per second
        self._max_tokens = float(burst)
        self._redis = redis_client
        self._lua_sha: Optional[str] = None

        # In-memory fallback buckets
        self._global_bucket = InMemoryTokenBucket(self._max_tokens, self._refill_rate)
        self._client_buckets: dict = {}
        self._bucket_lock = threading.Lock()

        self._total_calls = 0
        self._total_rejected = 0

        # Try to register Lua script in Redis
        if self._redis:
            try:
                self._lua_sha = self._redis.script_load(_LUA_TOKEN_BUCKET)
            except Exception:
                self._redis = None  # Fall back to in-memory

    def _get_client_bucket(self, client_id: str) -> InMemoryTokenBucket:
        """Get or create a per-client token bucket."""
        with self._bucket_lock:
            if client_id not in self._client_buckets:
                self._client_buckets[client_id] = InMemoryTokenBucket(
                    self._max_tokens, self._refill_rate
                )
            return self._client_buckets[client_id]

    def check(self, client_id: str = "global", tokens: float = 1.0) -> RateLimitResult:
        """
        Check if a request should be allowed.
        Uses Redis if available, falls back to in-memory.
        """
        self._total_calls += 1

        if self._redis and self._lua_sha:
            try:
                result = self._redis.evalsha(
                    self._lua_sha,
                    1,
                    f"ratelimit:{client_id}",
                    self._max_tokens,
                    self._refill_rate,
                    time.time(),
                    tokens,
                )
                allowed = bool(result)
                if not allowed:
                    self._total_rejected += 1
                return RateLimitResult(
                    allowed=allowed,
                    remaining_tokens=-1,  # Redis doesn't return this easily
                    client_id=client_id,
                )
            except Exception:
                pass  # Fall through to in-memory

        # In-memory fallback
        bucket = self._get_client_bucket(client_id)
        result = bucket.try_consume(tokens)
        result.client_id = client_id
        if not result.allowed:
            self._total_rejected += 1
        return result

    @property
    def stats(self) -> dict:
        return {
            "total_calls": self._total_calls,
            "total_rejected": self._total_rejected,
            "rejection_rate": round(self._total_rejected / max(self._total_calls, 1), 4),
            "using_redis": self._redis is not None,
        }
