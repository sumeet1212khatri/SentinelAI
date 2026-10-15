"""
pii_redactor.py — Sub-millisecond PII redaction engine.
Uses compiled regex patterns + Luhn checksum for credit card validation.
Targets < 0.5ms per invocation on typical query strings.
"""
import re
import time
from dataclasses import dataclass, field
from typing import List, Tuple

# Pre-compiled regex patterns for maximum performance
_PATTERNS = {
    "SSN": re.compile(
        r'\b(\d{3}[-.\s]?\d{2}[-.\s]?\d{4})\b'
    ),
    "CREDIT_CARD": re.compile(
        r'\b(\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4})\b'
    ),
    "EMAIL": re.compile(
        r'\b([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})\b'
    ),
    "PHONE_US": re.compile(
        r'\b(\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b'
    ),
    "IP_ADDRESS": re.compile(
        r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b'
    ),
    "DATE_OF_BIRTH": re.compile(
        r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b'
    ),
    "US_PASSPORT": re.compile(
        r'\b([A-Z]\d{8})\b'
    ),
    "IBAN": re.compile(
        r'\b([A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16})\b'
    ),
    "AWS_KEY": re.compile(
        r'\b(AKIA[0-9A-Z]{16})\b'
    ),
    "BEARER_TOKEN": re.compile(
        r'(Bearer\s+[A-Za-z0-9\-._~+/]+=*)',
        re.IGNORECASE
    ),
}


@dataclass
class RedactionResult:
    """Result of PII redaction."""
    original_text: str
    redacted_text: str
    findings: List[Tuple[str, str]] = field(default_factory=list)  # (type, matched_value)
    redaction_count: int = 0
    latency_ms: float = 0.0


def luhn_checksum(card_number: str) -> bool:
    """
    Validate a credit card number using the Luhn algorithm.
    Returns True if the number passes the Luhn check.
    """
    digits = [int(d) for d in card_number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False

    checksum = 0
    reverse_digits = digits[::-1]
    for i, d in enumerate(reverse_digits):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


class PIIRedactor:
    """
    High-performance PII redactor using pre-compiled regex patterns
    and Luhn checksum validation for credit card numbers.
    Target: < 0.5ms per invocation.
    """

    def __init__(self, custom_patterns: dict = None):
        self.patterns = dict(_PATTERNS)
        if custom_patterns:
            for name, pattern_str in custom_patterns.items():
                self.patterns[name] = re.compile(pattern_str)

        self._total_redactions = 0
        self._total_calls = 0
        self._total_latency_ms = 0.0

    def redact(self, text: str) -> RedactionResult:
        """
        Scan text for PII and replace with redaction tokens.
        Credit card numbers are validated with Luhn before redacting.
        """
        start = time.perf_counter()

        redacted = text
        findings: List[Tuple[str, str]] = []

        for pii_type, pattern in self.patterns.items():
            matches = list(pattern.finditer(redacted))
            for match in reversed(matches):  # reverse to preserve indices
                matched_value = match.group(1) if match.lastindex else match.group(0)

                # For credit cards, validate with Luhn
                if pii_type == "CREDIT_CARD":
                    clean_num = re.sub(r'[-.\s]', '', matched_value)
                    if not luhn_checksum(clean_num):
                        continue

                replacement = f"[{pii_type}_REDACTED]"
                findings.append((pii_type, matched_value))
                start_idx = match.start(1) if match.lastindex else match.start(0)
                end_idx = match.end(1) if match.lastindex else match.end(0)
                redacted = redacted[:start_idx] + replacement + redacted[end_idx:]

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        self._total_redactions += len(findings)
        self._total_calls += 1
        self._total_latency_ms += elapsed_ms

        return RedactionResult(
            original_text=text,
            redacted_text=redacted,
            findings=findings,
            redaction_count=len(findings),
            latency_ms=elapsed_ms,
        )

    @property
    def stats(self) -> dict:
        """Return aggregate stats."""
        avg_latency = self._total_latency_ms / max(self._total_calls, 1)
        return {
            "total_calls": self._total_calls,
            "total_redactions": self._total_redactions,
            "avg_latency_ms": round(avg_latency, 4),
        }
