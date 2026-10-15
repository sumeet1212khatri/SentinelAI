"""
query_rewriter.py — Production Pre-Retrieval Query Rewriting & Expansion Engine.
Normalizes colloquial questions, expands enterprise acronyms, resolves synonyms,
and constructs dense domain queries to maximize both BM25 and dense vector recall.
"""
import re
import logging
from typing import List, Set, Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RewrittenQuery:
    """Structured representation of a rewritten and expanded query."""
    original_query: str
    expanded_query: str
    search_keywords: List[str]
    detected_policies: List[str] = field(default_factory=list)
    intent_category: str = "general"


class QueryRewriter:
    """
    Intelligent Pre-Retrieval Query Rewriter for Enterprise RAG.
    Resolves domain terminology, maps informal user phrasing to canonical
    enterprise documentation sections, and injects policy anchors.
    """

    POLICY_EXPANSIONS: Dict[str, Dict[str, str]] = {
        "FIN-101": {
            "title": "Payment Processing & Refunds",
            "terms": "standard refunds 5-7 business days instant UPI refunds 15 minutes fee $0.25 automated reversals 120 seconds timeout payment processing",
            "triggers": ["refund", "refunds", "cash back", "money back", "reversal", "reversals", "upi instant", "payment failed", "settlement delay", "fin-101", "fin 101"]
        },
        "GW-204": {
            "title": "Gateway Rate Limiting",
            "terms": "token-bucket quotas 60 RPM Starter 600 RPM Growth 3000 RPM Enterprise burst rate limit quota 429 throttling",
            "triggers": ["rate limit", "rate limits", "quota", "quotas", "throttle", "throttling", "429", "rpm", "gw-204", "gw 204", "burst"]
        },
        "RISK-309": {
            "title": "Chargeback Lifecycle",
            "terms": "14-day representment window $15 dispute fee 0.90% VDMP card-brand threshold chargeback dispute fraud risk",
            "triggers": ["chargeback", "chargebacks", "dispute", "disputes", "representment", "vdmp", "visa dispute", "mastercard fraud", "risk-309", "risk 309"]
        },
        "HOOK-401": {
            "title": "Webhook Delivery & DLQ",
            "terms": "5000ms timeout 8 exponential retries HMAC-SHA256 signature X-NovaPay-Signature 14-day DLQ retention dead letter queue",
            "triggers": ["webhook", "webhooks", "dlq", "dead letter", "retry", "retries", "callback", "hook-401", "hook 401", "hmac"]
        },
        "SEC-402": {
            "title": "Security & Tokenization",
            "terms": "RSA-OAEP-2048 tokenization vault Luhn algorithm PII redaction 90-day AWS KMS key rotation encryption security token",
            "triggers": ["security", "token", "tokenization", "tokenize", "rsa", "kms", "pii", "luhn", "credit card store", "sec-402", "sec 402"]
        },
        "FX-603": {
            "title": "Cross-Border FX Settlement",
            "terms": "135 currencies supported 1.75% FX conversion markup T+3 SWIFT settlement foreign exchange cross border",
            "triggers": ["fx", "foreign exchange", "cross-border", "cross border", "currencies", "currency", "swift", "fx-603", "fx 603", "markup"]
        },
        "KYC-702": {
            "title": "Automated KYB Verification",
            "terms": "120s automated KYB $10,000 monthly transaction cap Tier-1 unverified merchants 180-day dormant merchant deactivation",
            "triggers": ["kyc", "kyb", "merchant verification", "unverified", "onboarding", "dormant", "kyc-702", "kyc 702", "tier-1"]
        },
        "INFRA-805": {
            "title": "High Availability & DR",
            "terms": "99.995% uptime SLA RPO=0s RTO<15s multi-region active-active clusters disaster recovery failover",
            "triggers": ["uptime", "disaster recovery", "dr", "rpo", "rto", "high availability", "failover", "infra-805", "infra 805", "ha"]
        },
        "SUB-904": {
            "title": "Recurring Subscriptions",
            "terms": "24h pre-debit notifications INR 15,000 mandate cap Day +1 +3 +5 +7 Smart Dunning recurring billing subscriptions",
            "triggers": ["subscription", "subscriptions", "recurring", "autopay", "mandate", "dunning", "sub-904", "sub 904", "pre-debit"]
        },
        "SLA-1001": {
            "title": "Pricing & Incident SLAs",
            "terms": "2.0% domestic 0.0% UPI 3.0% international Sev-1 incident response SLA < 15 minutes pricing merchant fee",
            "triggers": ["pricing", "fees", "fee", "incident", "sev-1", "sev 1", "response time", "sla-1001", "sla 1001", "service level"]
        },
    }

    POLICY_REGEX = re.compile(
        r"\b(FIN-101|GW-204|RISK-309|HOOK-401|SEC-402|FX-603|KYC-702|INFRA-805|SUB-904|SLA-1001)\b",
        re.IGNORECASE,
    )

    STOP_WORDS = {
        "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "is",
        "are", "was", "were", "what", "which", "who", "whom", "this", "that", "these",
        "those", "am", "been", "being", "have", "has", "had", "do", "does", "did",
        "can", "could", "should", "would", "i", "me", "my", "we", "our", "you", "your",
        "it", "its", "they", "them", "their", "please", "tell", "explain", "about",
    }

    def __init__(self):
        logger.info("Initialized QueryRewriter with 10 enterprise policy expansion rules")

    def rewrite(self, query: str) -> RewrittenQuery:
        """
        Rewrite, normalize, and enrich the query for hybrid search.
        
        Args:
            query: Raw user query string.
            
        Returns:
            RewrittenQuery containing expanded query and search keywords.
        """
        clean_q = query.strip()
        lower_q = clean_q.lower()

        detected_policies: List[str] = []
        expanded_fragments: List[str] = [clean_q]

        # 1. Direct Regex match for explicit policy codes
        for match in self.POLICY_REGEX.finditer(query):
            code = match.group(1).upper()
            if code not in detected_policies:
                detected_policies.append(code)

        # 2. Broad Query Detection (e.g. "what is the policy?", "show company policies")
        is_broad_policy_query = any(
            phrase in lower_q for phrase in [
                "company policy", "all policies", "what are the policies",
                "what is policy", "company policies", "list policies", "all slas",
                "core policies", "rules of company", "overview of policies"
            ]
        )

        if is_broad_policy_query:
            expanded_fragments.append(
                "NovaPay Enterprise 10 core policies FIN-101 refunds GW-204 rate limits "
                "RISK-309 chargebacks HOOK-401 webhooks SEC-402 tokenization FX-603 settlement "
                "KYC-702 verification INFRA-805 uptime SUB-904 subscriptions SLA-1001 pricing"
            )
            detected_policies = list(self.POLICY_EXPANSIONS.keys())
        else:
            # 3. Trigger-based domain intent matching
            for code, meta in self.POLICY_EXPANSIONS.items():
                if any(trigger in lower_q for trigger in meta["triggers"]):
                    if code not in detected_policies:
                        detected_policies.append(code)
                    expanded_fragments.append(f"{code} {meta['title']}: {meta['terms']}")

        # 4. Extract search keywords for BM25
        tokens = re.findall(r"[A-Za-z0-9_\-\.\%]+", lower_q)
        keywords = [t for t in tokens if t not in self.STOP_WORDS and len(t) > 1]

        # Add detected policy codes into keywords
        for p in detected_policies:
            if p.lower() not in keywords:
                keywords.insert(0, p.lower())

        expanded_query = " ".join(expanded_fragments)

        intent = "broad_policy" if is_broad_policy_query else (
            detected_policies[0] if detected_policies else "general"
        )

        return RewrittenQuery(
            original_query=clean_q,
            expanded_query=expanded_query,
            search_keywords=keywords,
            detected_policies=detected_policies,
            intent_category=intent,
        )
