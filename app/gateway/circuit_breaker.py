"""
circuit_breaker.py — Sliding-window state machine circuit breaker.
States: CLOSED -> OPEN -> HALF_OPEN
Prevents cascading failures when LLM providers are down.
"""
import time
import threading
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from collections import deque

from ..config import (
    CB_FAILURE_THRESHOLD,
    CB_RECOVERY_TIMEOUT,
    CB_HALF_OPEN_MAX_CALLS,
    CB_WINDOW_SIZE,
)

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass
class CircuitBreakerStats:
    """Statistics for a circuit breaker."""
    state: str
    total_calls: int = 0
    total_successes: int = 0
    total_failures: int = 0
    total_rejected: int = 0
    consecutive_failures: int = 0
    last_failure_time: Optional[float] = None
    last_state_change: Optional[float] = None


class CircuitBreaker:
    """
    Sliding-window circuit breaker state machine.

    CLOSED: Normal operation. Track failures in sliding window.
            If failures exceed threshold -> transition to OPEN.
    OPEN:   All calls rejected immediately.
            After recovery_timeout -> transition to HALF_OPEN.
    HALF_OPEN: Allow limited calls to test recovery.
               If success -> transition to CLOSED.
               If failure -> transition back to OPEN.
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = CB_FAILURE_THRESHOLD,
        recovery_timeout: float = CB_RECOVERY_TIMEOUT,
        half_open_max_calls: int = CB_HALF_OPEN_MAX_CALLS,
        window_size: float = CB_WINDOW_SIZE,
    ):
        self.name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_calls = half_open_max_calls
        self._window_size = window_size

        self._state = CircuitState.CLOSED
        self._lock = threading.Lock()

        # Sliding window of failure timestamps
        self._failure_window: deque = deque()
        self._consecutive_failures = 0
        self._half_open_calls = 0
        self._half_open_successes = 0

        # Stats
        self._total_calls = 0
        self._total_successes = 0
        self._total_failures = 0
        self._total_rejected = 0
        self._last_failure_time: Optional[float] = None
        self._last_state_change = time.time()
        self._opened_at: Optional[float] = None

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._check_state_transition()
            return self._state

    def _check_state_transition(self):
        """Check if a state transition should occur."""
        now = time.time()

        if self._state == CircuitState.OPEN:
            if self._opened_at and (now - self._opened_at) >= self._recovery_timeout:
                self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._last_state_change = time.time()

        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._half_open_successes = 0
        elif new_state == CircuitState.CLOSED:
            self._consecutive_failures = 0
            self._failure_window.clear()
        elif new_state == CircuitState.OPEN:
            self._opened_at = time.time()

        logger.info(f"Circuit breaker '{self.name}': {old_state} -> {new_state}")

    def can_execute(self) -> bool:
        """Check if a call is allowed through the circuit breaker."""
        with self._lock:
            self._check_state_transition()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                self._total_rejected += 1
                return False

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self._half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                self._total_rejected += 1
                return False

            return False

    def record_success(self):
        """Record a successful call."""
        with self._lock:
            self._total_calls += 1
            self._total_successes += 1
            self._consecutive_failures = 0

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self._half_open_max_calls:
                    self._transition_to(CircuitState.CLOSED)

    def record_failure(self):
        """Record a failed call."""
        with self._lock:
            now = time.time()
            self._total_calls += 1
            self._total_failures += 1
            self._consecutive_failures += 1
            self._last_failure_time = now

            # Add to sliding window
            self._failure_window.append(now)

            # Remove old entries outside the window
            cutoff = now - self._window_size
            while self._failure_window and self._failure_window[0] < cutoff:
                self._failure_window.popleft()

            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if len(self._failure_window) >= self._failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def reset(self):
        """Manually reset the circuit breaker to CLOSED."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._failure_window.clear()
            self._consecutive_failures = 0

    @property
    def stats(self) -> dict:
        with self._lock:
            self._check_state_transition()
            return {
                "name": self.name,
                "state": self._state.value,
                "total_calls": self._total_calls,
                "total_successes": self._total_successes,
                "total_failures": self._total_failures,
                "total_rejected": self._total_rejected,
                "consecutive_failures": self._consecutive_failures,
                "failures_in_window": len(self._failure_window),
                "last_failure_time": self._last_failure_time,
            }
