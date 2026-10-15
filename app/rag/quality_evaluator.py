"""
quality_evaluator.py — Production Pre-Generation Retrieval Quality & Confidence Evaluator.
Measures context relevance, factual grounding density, and lexical-semantic alignment
before passing contexts to the LLM to prevent hallucinations.
"""
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import logging

from .pdf_chunker import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievalQualityReport:
    """Detailed quality evaluation of the retrieved contexts."""
    quality_score: float  # 0.0 to 1.0 (0% to 100%)
    confidence_level: str  # 'high', 'medium', 'low'
    is_sufficient: bool
    context_relevance: float
    information_density: float
    matched_key_concepts: List[str]
    missing_concepts: List[str]
    recommended_action: str  # 'generate', 'clarify', 'fallback'


class RetrievalQualityEvaluator:
    """
    Evaluates whether the retrieved Top-K chunks contain sufficient evidence
    to accurately and truthfully answer the user's query before invoking the LLM.
    """

    POLICY_KEYWORDS = {
        "refund", "rate limit", "chargeback", "webhook", "tokenization", "settlement",
        "kyb", "uptime", "subscription", "pricing", "sla", "reversal", "quota",
        "dispute", "dlq", "rsa", "kms", "swift", "currencies", "mandate", "dunning"
    }

    def __init__(self, sufficiency_threshold: float = 0.35):
        self.sufficiency_threshold = sufficiency_threshold

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[DocumentChunk],
        scores: Optional[List[float]] = None,
    ) -> RetrievalQualityReport:
        """
        Evaluate retrieval quality before trusting the context for LLM synthesis.
        """
        if not retrieved_chunks:
            return RetrievalQualityReport(
                quality_score=0.0,
                confidence_level="low",
                is_sufficient=False,
                context_relevance=0.0,
                information_density=0.0,
                matched_key_concepts=[],
                missing_concepts=list(re.findall(r"\w{3,}", query.lower())),
                recommended_action="fallback",
            )

        lower_query = query.lower()
        query_terms = set(re.findall(r"\b[A-Za-z0-9\-\.]{3,}\b", lower_query))

        # Filter out common filler words
        filler = {"what", "when", "where", "which", "how", "does", "have", "with", "from", "tell", "explain", "about"}
        salient_query_terms = query_terms - filler

        # Combine text of all retrieved chunks
        combined_text = " ".join([c.text.lower() + " " + (c.section_title or "").lower() for c in retrieved_chunks])

        # 1. Calculate Concept Coverage
        matched_concepts = []
        missing_concepts = []

        for term in salient_query_terms:
            if term in combined_text:
                matched_concepts.append(term)
            else:
                missing_concepts.append(term)

        if salient_query_terms:
            relevance = len(matched_concepts) / len(salient_query_terms)
        else:
            relevance = 0.85  # Fallback for very short greetings/general queries

        # 2. Information Density (ratio of numbers, acronyms, and uppercase anchors)
        numbers_and_codes = re.findall(r"\b(?:[A-Z]{2,}-\d+|\d+(?:\.\d+)?%?|\$\d+)\b", combined_text)
        density = min(1.0, len(numbers_and_codes) / 8.0)

        # 3. Base similarity component
        top_score = scores[0] if scores and len(scores) > 0 else 0.5
        top_score_norm = max(0.0, min(1.0, top_score))

        # Composite Retrieval Quality Score
        composite_quality = (
            (relevance * 0.50) +
            (top_score_norm * 0.30) +
            (density * 0.20)
        )
        composite_quality = round(min(1.0, max(0.0, composite_quality)), 3)

        if composite_quality >= 0.70:
            confidence = "high"
            action = "generate"
        elif composite_quality >= self.sufficiency_threshold:
            confidence = "medium"
            action = "generate"
        else:
            confidence = "low"
            action = "clarify"

        is_sufficient = composite_quality >= self.sufficiency_threshold

        return RetrievalQualityReport(
            quality_score=composite_quality,
            confidence_level=confidence,
            is_sufficient=is_sufficient,
            context_relevance=round(relevance, 3),
            information_density=round(density, 3),
            matched_key_concepts=matched_concepts,
            missing_concepts=missing_concepts,
            recommended_action=action,
        )
