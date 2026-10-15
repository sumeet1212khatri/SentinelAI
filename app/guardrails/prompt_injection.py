"""
prompt_injection.py — Prompt injection, jailbreak & system-prompt leak blocker.
Multi-layer defense with pattern matching and heuristic scoring.
"""
import re
import time
from dataclasses import dataclass, field
from typing import List, Tuple

# ─────────────────────────────────────────────────
# Attack pattern categories
# ─────────────────────────────────────────────────

# Jailbreak patterns
_JAILBREAK_PATTERNS = [
    re.compile(r'ignore\s+(all\s+)?(previous\s+|prior\s+|above\s+|earlier\s+)?(instructions?|prompts?|rules?|directives?|limits?|policies?|guidelines?)', re.IGNORECASE),
    re.compile(r'disregard\s+(all\s+)?(previous\s+|prior\s+|above\s+|earlier\s+)?(instructions?|prompts?|rules?|directives?|limits?|policies?|guidelines?)', re.IGNORECASE),
    re.compile(r'forget\s+(all\s+)?(previous\s+|prior\s+|above\s+|earlier\s+)?(instructions?|prompts?|rules?|directives?|limits?|policies?|guidelines?)', re.IGNORECASE),
    re.compile(r'you\s+are\s+now\s+(DAN|a\s+new|an?\s+unrestricted|an?\s+unfiltered)', re.IGNORECASE),
    re.compile(r'(DAN|do\s+anything\s+now)\s+mode', re.IGNORECASE),
    re.compile(r'jailbreak(ed)?', re.IGNORECASE),
    re.compile(r'pretend\s+(you\s+)?(are|have)\s+(an?\s+unrestricted|no\s+(restrictions?|rules?|limitations?|filters?))', re.IGNORECASE),
    re.compile(r'act\s+as\s+if\s+(you\s+)?(have\s+)?no\s+(restrictions?|guidelines?|safety)', re.IGNORECASE),
    re.compile(r'bypass\s+(your\s+)?(safety|content|ethical|billing)\s*(filters?|restrictions?|guidelines?|rules?)', re.IGNORECASE),
    re.compile(r'override\s+(your\s+)?(system|core|safety|OFAC)\s*(prompt|instructions?|programming|sanctions)', re.IGNORECASE),
]

# System prompt & secret extraction patterns
_SYSTEM_LEAK_PATTERNS = [
    re.compile(r'\b(show\s+me|reveal|display|print|output|tell\s+me|what\s+is|repeat|echo|copy|dump|leak|extract)\b.*?\b(system\s+prompt|initial\s+prompt|original\s+prompt|hidden\s+prompt|internal\s+prompt|instructions?|directives?|private\s+key|signing\s+key|secrets?|credentials?|confidential|discount\s+negotiation|api\s*key)', re.IGNORECASE),
    re.compile(r'(show\s+me|show|reveal|display|print|output|tell\s+me|what\s+is|repeat|echo|copy|dump|leak|extract)\s+(your\s+|the\s+)?(initial\s+|original\s+|system\s+|hidden\s+|internal\s+|confidential\s+)?(system\s+prompt|instructions?|prompt|directives?|rules?|discount|private\s+key|secrets?|keys?)', re.IGNORECASE),
    re.compile(r'(repeat|echo|copy)\s+(your\s+|the\s+)?(system|initial|original)\s+(prompt|message|instructions?)', re.IGNORECASE),
    re.compile(r'what\s+(were|are)\s+(your\s+)?(original|initial|system)\s+(instructions?|prompt|directives?)', re.IGNORECASE),
    re.compile(r'(dump|leak|expose|extract)\s+(the\s+)?(system|hidden|internal|confidential)\s*(prompt|instructions?|configuration|pricing)', re.IGNORECASE),
    re.compile(r'begin\s+your\s+(response|reply|answer)\s+with\s*["\']', re.IGNORECASE),
]

# Injection patterns (attempting to inject new instructions or escape context)
_INJECTION_PATTERNS = [
    re.compile(r'<\s*(system|assistant|user)\s*>', re.IGNORECASE),
    re.compile(r'\[\s*(SYSTEM|INST|SYS)\s*\]', re.IGNORECASE),
    re.compile(r'###\s*(System|Instruction|New\s+Role)', re.IGNORECASE),
    re.compile(r'(new\s+)?(system\s+)?(prompt|instruction)\s*[:=]\s*', re.IGNORECASE),
    re.compile(r'from\s+now\s+on,?\s+(you\s+)?(will|must|should|are)', re.IGNORECASE),
    re.compile(r'(above|previous)\s+(text|prompt|instructions?)\s+(is|are|was|were)\s+(just\s+)?(a\s+)?(test|joke|fake)', re.IGNORECASE),
    re.compile(r'(DROP\s+TABLE|SELECT\s+.*FROM|UNION\s+SELECT|--\s*verify)', re.IGNORECASE),
]

# Encoding/obfuscation attempts
_ENCODING_PATTERNS = [
    re.compile(r'(base64|rot13|hex)\s*(encode|decode|convert)', re.IGNORECASE),
    re.compile(r'\\x[0-9a-fA-F]{2}', re.IGNORECASE),
    re.compile(r'&#\d{2,4};', re.IGNORECASE),  # HTML entities
    re.compile(r'%[0-9a-fA-F]{2}', re.IGNORECASE),  # URL encoding
]


@dataclass
class InjectionResult:
    """Result of prompt injection analysis."""
    is_blocked: bool
    risk_score: float  # 0.0 to 1.0
    attack_types: List[str] = field(default_factory=list)
    matched_patterns: List[Tuple[str, str]] = field(default_factory=list)  # (category, matched text)
    latency_ms: float = 0.0


class PromptInjectionGuard:
    """
    Multi-layer prompt injection defense engine.
    Detects jailbreaks, system prompt leaks, role injections, and encoding attacks.
    """

    BLOCK_THRESHOLD = 0.4  # Risk score above this triggers blocking

    def __init__(self, block_threshold: float = None):
        if block_threshold is not None:
            self.BLOCK_THRESHOLD = block_threshold

        self._total_blocked = 0
        self._total_calls = 0
        self._attack_type_counts: dict = {}

    def analyze(self, text: str) -> InjectionResult:
        """
        Analyze input text for prompt injection attempts.
        Returns InjectionResult with risk assessment.
        """
        start = time.perf_counter()

        attack_types = []
        matched_patterns = []
        risk_score = 0.0

        # Check jailbreak patterns (highest severity)
        for pattern in _JAILBREAK_PATTERNS:
            match = pattern.search(text)
            if match:
                attack_types.append("JAILBREAK")
                matched_patterns.append(("JAILBREAK", match.group(0)))
                risk_score += 0.5
                break  # One match is enough per category

        # Check system prompt leak patterns
        for pattern in _SYSTEM_LEAK_PATTERNS:
            match = pattern.search(text)
            if match:
                attack_types.append("SYSTEM_PROMPT_LEAK")
                matched_patterns.append(("SYSTEM_PROMPT_LEAK", match.group(0)))
                risk_score += 0.45
                break

        # Check injection patterns
        for pattern in _INJECTION_PATTERNS:
            match = pattern.search(text)
            if match:
                attack_types.append("PROMPT_INJECTION")
                matched_patterns.append(("PROMPT_INJECTION", match.group(0)))
                risk_score += 0.45
                break

        # Check encoding/obfuscation
        encoding_count = 0
        for pattern in _ENCODING_PATTERNS:
            matches = pattern.findall(text)
            encoding_count += len(matches)
            if matches:
                attack_types.append("ENCODING_ATTACK")
                matched_patterns.append(("ENCODING_ATTACK", str(matches[0])))

        if encoding_count > 3:
            risk_score += 0.3

        # Heuristic checks
        # Excessive special characters (potential obfuscation)
        special_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
        if special_ratio > 0.4:
            risk_score += 0.15
            attack_types.append("HIGH_SPECIAL_CHAR_RATIO")

        # Extremely long input (potential stuffing attack)
        if len(text) > 5000:
            risk_score += 0.1
            attack_types.append("EXCESSIVE_LENGTH")

        risk_score = min(risk_score, 1.0)
        is_blocked = risk_score >= self.BLOCK_THRESHOLD

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        self._total_calls += 1
        if is_blocked:
            self._total_blocked += 1
        for atype in set(attack_types):
            self._attack_type_counts[atype] = self._attack_type_counts.get(atype, 0) + 1

        return InjectionResult(
            is_blocked=is_blocked,
            risk_score=risk_score,
            attack_types=list(set(attack_types)),
            matched_patterns=matched_patterns,
            latency_ms=elapsed_ms,
        )

    @property
    def stats(self) -> dict:
        """Return aggregate stats."""
        return {
            "total_calls": self._total_calls,
            "total_blocked": self._total_blocked,
            "block_rate": round(self._total_blocked / max(self._total_calls, 1), 4),
            "attack_type_counts": dict(self._attack_type_counts),
        }
