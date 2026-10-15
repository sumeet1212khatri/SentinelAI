"""
output_guard.py — Response validation guardrail.
Checks LLM outputs for PII leaks, empty/hallucinated responses, and content policy violations.
"""
import re
import time
from dataclasses import dataclass, field
from typing import List
from .pii_redactor import PIIRedactor


@dataclass
class OutputGuardResult:
    """Result of output validation."""
    is_safe: bool
    sanitized_text: str
    issues: List[str] = field(default_factory=list)
    pii_found: int = 0
    latency_ms: float = 0.0


# Hallucination / low-quality indicators
_HALLUCINATION_PHRASES = [
    "as an ai language model",
    "i don't have access to",
    "i cannot provide",
    "i'm not able to",
    "my training data",
    "my knowledge cutoff",
    "i was trained",
    "i apologize, but i cannot",
    "please note that i",
]

# Repetition detection regex
_REPETITION_PATTERN = re.compile(r'(.{20,}?)\1{2,}', re.DOTALL)


class OutputGuard:
    """
    Validates LLM responses for safety, quality, and PII leaks.
    """

    def __init__(self, min_response_length: int = 10, max_hallucination_score: float = 0.3):
        self._pii_redactor = PIIRedactor()
        self._min_response_length = min_response_length
        self._max_hallucination_score = max_hallucination_score
        self._total_calls = 0
        self._total_blocked = 0
        self._total_pii_found = 0

    def validate(self, response_text: str) -> OutputGuardResult:
        """
        Validate and sanitize LLM response.
        Returns OutputGuardResult with safety assessment and sanitized text.
        """
        start = time.perf_counter()
        issues = []
        is_safe = True
        sanitized = response_text

        # 1. Empty / too-short response check
        stripped = response_text.strip()
        if len(stripped) < self._min_response_length:
            issues.append("EMPTY_OR_SHORT_RESPONSE")
            is_safe = False

        # 2. PII leak check — redact any PII found in response
        pii_result = self._pii_redactor.redact(sanitized)
        pii_count = pii_result.redaction_count
        if pii_count > 0:
            sanitized = pii_result.redacted_text
            issues.append(f"PII_LEAK_DETECTED: {pii_count} items redacted")
            self._total_pii_found += pii_count

        # 3. Hallucination / disclaimer detection
        lower_text = stripped.lower()
        hallucination_hits = sum(
            1 for phrase in _HALLUCINATION_PHRASES if phrase in lower_text
        )
        hallucination_score = hallucination_hits / max(len(_HALLUCINATION_PHRASES), 1)
        if hallucination_score >= self._max_hallucination_score:
            issues.append(f"POTENTIAL_HALLUCINATION: score={hallucination_score:.2f}")

        # 4. Repetition / stuck-in-loop detection
        if _REPETITION_PATTERN.search(stripped):
            issues.append("REPETITIVE_OUTPUT_DETECTED")
            is_safe = False

        # 5. Sanitize formatting: normalize multiple horizontal spaces while strictly preserving single and double \n line breaks
        sanitized = re.sub(r"[ \t]+", " ", sanitized)
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized).strip()

        # 6. Content still safe if only warnings (not hard blocks)
        hard_blocks = {"EMPTY_OR_SHORT_RESPONSE", "REPETITIVE_OUTPUT_DETECTED"}
        if not any(issue.split(":")[0] in hard_blocks for issue in issues):
            is_safe = True

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        self._total_calls += 1
        if not is_safe:
            self._total_blocked += 1

        return OutputGuardResult(
            is_safe=is_safe,
            sanitized_text=sanitized,
            issues=issues,
            pii_found=pii_count,
            latency_ms=elapsed_ms,
        )

    @property
    def stats(self) -> dict:
        return {
            "total_calls": self._total_calls,
            "total_blocked": self._total_blocked,
            "total_pii_found": self._total_pii_found,
        }
