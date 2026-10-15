"""
llm_router.py — Async multi-provider LLM router with failover.
Routes: OpenAI -> Groq -> Deterministic Local Simulator
Supports SSE streaming and circuit-breaker integration.
"""
import asyncio
import time
import json
import logging
import hashlib
from typing import AsyncGenerator, Optional, List
from dataclasses import dataclass, field

from ..config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_BASE_URL,
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_BASE_URL,
    USE_LOCAL_SIMULATOR,
    LLM_REQUEST_TIMEOUT,
    LLM_PRICING,
    AVG_INPUT_TOKENS,
    AVG_OUTPUT_TOKENS,
    LLM_MAX_TOKENS,
)
from .circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    text: str
    model: str
    provider: str
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    is_cached: bool = False
    is_simulated: bool = False


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    name: str
    api_key: str
    model: str
    base_url: str
    circuit_breaker: CircuitBreaker = field(default_factory=lambda: CircuitBreaker())
    priority: int = 0


NOVA_PAY_CORPUS_OVERVIEW = """
NOVA PAY ENTERPRISE MANUAL — CORPUS OVERVIEW (10 CORE POLICIES):
- DOC-OPS-001 (Policy FIN-101) Payment Processing & Refund SLAs: Standard card refunds take 5-7 business days; instant UPI refunds complete within 15 minutes at $0.25 fee. Automated reversals trigger within 120 seconds for API-initiated timeout events.
- DOC-GW-002 (Policy GW-204) API Gateway Rate Limits & Quotas: Token-bucket quotas at 60 RPM (Starter), 600 RPM (Growth), and 3,000 RPM (Enterprise tier). Returns HTTP 429 on exhaustion.
- DOC-RISK-003 (Policy RISK-309) Chargebacks & Dispute Lifecycle: 14-day dispute representment window, $15 non-refundable dispute fee, 0.90% VDMP card-brand threshold.
- DOC-HOOK-004 (Policy HOOK-401) Asynchronous Webhook Delivery & Dead-Letter Queue (DLQ): 5,000ms HTTP timeout, 8 exponential backoff retries, HMAC-SHA256 signature verification in X-NovaPay-Signature header, Dead-Letter Queue (DLQ) retention 14 days.
- DOC-SEC-005 (Policy SEC-402) Security & Tokenization Vault: PCI-DSS v4.0 Level 1 compliant, RSA-OAEP-2048 card vaulting, sub-1ms regex + Luhn checksum PII redaction, 90-day AWS KMS envelope key rotation.
- DOC-FX-006 (Policy FX-603) Cross-Border FX Settlement: Supports 135 currencies, 1.75% FX conversion markup, T+3 SWIFT settlement cycle, automated electronic Foreign Inward Remittance Certificate (e-FIRC).
- DOC-KYC-007 (Policy KYC-702) KYB Onboarding & Compliance: 120-second automated KYB verification, $10,000 monthly transaction cap for Tier-1 unverified merchants, 180-day dormant merchant deactivation.
- DOC-INFRA-008 (Policy INFRA-805) High Availability & Disaster Recovery: 99.995% uptime SLA, RPO=0s (zero data loss), RTO<15s failover across multi-region active-active CockroachDB clusters, 2,000ms circuit breaker timeout.
- DOC-SUB-009 (Policy SUB-904) Recurring Subscriptions & Smart Dunning: 24-hour pre-debit e-mandate notifications, INR 15,000 per-transaction cap without additional OTP, Smart Dunning retry sequence on Day +1, +3, +5, and +7.
- DOC-SLA-010 (Policy SLA-1001) Pricing Matrix & Incident SLAs: 2.0% domestic card fee, 0.0% UPI transaction fee, 3.0% international card fee; Sev-1 critical incident response SLA < 15 minutes, 24/7 dedicated enterprise support.
"""

SYSTEM_PROMPT = f"""You are the official SentinelAI Enterprise AI Assistant for NovaPay.
Your primary role is to answer questions about NovaPay enterprise policies, SLAs, APIs, system architecture, security, and developer integration based on the official NovaPay Enterprise documentation.

{NOVA_PAY_CORPUS_OVERVIEW}

INSTRUCTIONS:
1. Formatting Constraints: NEVER output raw HTML tags like <br>. When creating Markdown tables, put each table row on its own newline \\n and keep cells concise. Prefer clean Markdown tables (with one row per line) or structured bulleted sections.
2. Broad Policy Questions: If the user asks a broad question like "what is the company policy?", "what are the policies?", or "explain NovaPay policies", provide a clean, structured summary of NovaPay's 10 core enterprise policies with their exact Policy Reference codes (FIN-101, GW-204, RISK-309, HOOK-401, SEC-402, FX-603, KYC-702, INFRA-805, SUB-904, SLA-1001).
3. Specific Questions: When answering specific queries, rely directly on the retrieved documentation chunks provided below. Answer directly and precisely, citing the Document ID (e.g., DOC-OPS-001) and Page number.
4. Out-of-Domain Questions: If the user asks an unrelated or out-of-domain question (such as "i need degree" or other irrelevant topics), politely explain that you are the NovaPay Enterprise AI Assistant and list the topics and policies you can assist with.
5. Tone & Style: Be professional, concise, authoritative, and helpful. Use clear markdown formatting.
"""



class DeterministicSimulator:
    """
    Deterministic local LLM simulator that generates contextually relevant
    responses from the provided RAG context. Always available as final fallback.
    """

    def __init__(self, simulated_latency_s: float = 0.08):
        self._call_count = 0
        self._total_latency_ms = 0.0
        self._simulated_latency_s = simulated_latency_s

    def generate(self, prompt: str, context: str = "", query: str = "") -> LLMResponse:
        """
        Generate a deterministic response based on the RAG context.
        """
        start = time.perf_counter()
        if self._simulated_latency_s > 0:
            time.sleep(self._simulated_latency_s)

        self._call_count += 1

        # Extract key information from the context to build a meaningful response
        lower_q = (query or prompt).lower()
        if any(term in lower_q for term in ["policy", "policies", "what is policy", "company policy"]):
            response = (
                "NovaPay Enterprise operates under 10 core operational and compliance policies:\n\n"
                "• **FIN-101 (DOC-OPS-001): Payment Processing & Refunds** — Standard refunds take 5-7 business days; instant UPI refunds complete within 15 minutes at $0.25 fee. Automated reversals trigger within 120 seconds for API timeouts.\n"
                "• **GW-204 (DOC-GW-002): Gateway Rate Limiting** — Token-bucket quotas at 60 RPM (Starter), 600 RPM (Growth), and 3,000 RPM (Enterprise tier).\n"
                "• **RISK-309 (DOC-RISK-003): Chargeback Lifecycle** — 14-day representment window, $15 dispute fee, 0.90% VDMP card-brand threshold.\n"
                "• **HOOK-401 (DOC-HOOK-004): Webhook Delivery & DLQ** — 5,000ms timeout, 8 exponential retries, HMAC-SHA256 signature verification in X-NovaPay-Signature header, 14-day DLQ retention.\n"
                "• **SEC-402 (DOC-SEC-005): Security & Tokenization** — RSA-OAEP-2048 tokenization vault, <1ms Luhn PII redaction, 90-day AWS KMS key rotation.\n"
                "• **FX-603 (DOC-FX-006): Cross-Border FX Settlement** — 135 currencies supported, 1.75% FX conversion markup, T+3 SWIFT settlement cycle.\n"
                "• **KYC-702 (DOC-KYC-007): Automated KYB Verification** — 120s automated KYB, $10,000 monthly transaction cap for Tier-1 unverified merchants, 180-day dormant merchant deactivation.\n"
                "• **INFRA-805 (DOC-INFRA-008): High Availability & DR** — 99.995% uptime SLA, RPO=0s, RTO<15s across multi-region active-active clusters.\n"
                "• **SUB-904 (DOC-SUB-009): Recurring Subscriptions** — 24h pre-debit notifications, INR 15,000 mandate cap, Day +1,+3,+5,+7 Smart Dunning.\n"
                "• **SLA-1001 (DOC-SLA-010): Pricing & Incident SLAs** — 2.0% domestic, 0.0% UPI, 3.0% intl; Sev-1 incident response SLA < 15 minutes."
            )
        elif context.strip():
            # Use the context directly — summarize by taking key sentences
            sentences = [s.strip() for s in context.replace('\n', '. ').split('.') if len(s.strip()) > 20]
            if sentences:
                # Build a structured response from available context
                selected = sentences[:5]  # Top 5 sentences
                response_body = " ".join(selected) + "."
                response = (
                    f"Based on the enterprise documentation:\n\n"
                    f"{response_body}\n\n"
                    f"This information is derived from the NovaPay Enterprise RAG corpus. "
                    f"Please refer to the relevant sections for complete details."
                )
            else:
                response = self._generic_response(query)
        else:
            response = self._generic_response(query)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._total_latency_ms += elapsed_ms

        input_tokens = len(prompt.split()) + len(context.split())
        output_tokens = len(response.split())

        return LLMResponse(
            text=response,
            model="local-simulator",
            provider="local",
            latency_ms=elapsed_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=0.0,
            is_simulated=True,
        )

    def _generic_response(self, query: str) -> str:
        """Generate a generic response when no context is available."""
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        return (
            f"Based on the available documentation, here is the relevant information "
            f"for your query. The NovaPay Enterprise system provides comprehensive "
            f"features for payment processing, transaction management, and enterprise "
            f"integration capabilities. For detailed specifications, please refer to "
            f"the relevant sections of the enterprise documentation. [ref:{query_hash}]"
        )

    async def generate_stream(self, prompt: str, context: str = "", query: str = "") -> AsyncGenerator[str, None]:
        """Stream response token-by-token (simulated)."""
        response = self.generate(prompt, context, query)
        words = response.text.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.001)  # Simulate streaming delay

    @property
    def stats(self) -> dict:
        avg_latency = self._total_latency_ms / max(self._call_count, 1)
        return {
            "total_calls": self._call_count,
            "avg_latency_ms": round(avg_latency, 4),
        }


class LLMRouter:
    """
    Async multi-provider LLM router with failover chain:
    OpenAI -> Groq -> Deterministic Local Simulator

    Features:
    - Per-provider circuit breakers
    - Automatic failover
    - Cost estimation
    - SSE streaming support
    """

    def __init__(self):
        self._providers: List[ProviderConfig] = []
        self._simulator = DeterministicSimulator()

        # Configure providers based on available API keys
        if OPENAI_API_KEY:
            self._providers.append(ProviderConfig(
                name="openai",
                api_key=OPENAI_API_KEY,
                model=OPENAI_MODEL,
                base_url=OPENAI_BASE_URL,
                circuit_breaker=CircuitBreaker(name="openai"),
                priority=0,
            ))

        if GROQ_API_KEY:
            self._providers.append(ProviderConfig(
                name="groq",
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL,
                base_url=GROQ_BASE_URL,
                circuit_breaker=CircuitBreaker(name="groq"),
                priority=1,
            ))

        # Sort by priority
        self._providers.sort(key=lambda p: p.priority)

        # Stats
        self._total_calls = 0
        self._provider_calls = {}
        self._total_cost_usd = 0.0
        self._total_latency_ms = 0.0

        logger.info(
            f"LLM Router initialized with {len(self._providers)} remote providers "
            f"+ local simulator"
        )

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD for a given model and token counts."""
        pricing = LLM_PRICING.get(model, {"input": 0.0, "output": 0.0})
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    async def _call_provider(
        self,
        provider: ProviderConfig,
        prompt: str,
        context: str = "",
    ) -> Optional[LLMResponse]:
        """Call a single LLM provider via httpx."""
        if not provider.circuit_breaker.can_execute():
            logger.warning(f"Circuit breaker OPEN for {provider.name}, skipping")
            return None

        try:
            import httpx

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            if context and context.strip():
                formatted_user_prompt = (
                    f"=== RETRIEVED ENTERPRISE DOCUMENTATION CHUNKS ===\n"
                    f"{context.strip()}\n"
                    f"================================================\n\n"
                    f"User Inquiry: {prompt}"
                )
                messages.append({"role": "user", "content": formatted_user_prompt})
            else:
                messages.append({"role": "user", "content": prompt})

            start = time.perf_counter()

            async with httpx.AsyncClient(timeout=LLM_REQUEST_TIMEOUT) as client:
                response = await client.post(
                    f"{provider.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {provider.api_key}",
                        "Content-Type": "application/json",
                        "User-Agent": "SentinelAI-RAG/1.0",
                    },
                    json={
                        "model": provider.model,
                        "messages": messages,
                        "temperature": 0.3,
                        "max_tokens": LLM_MAX_TOKENS,
                    },
                )
                response.raise_for_status()
                data = response.json()

            elapsed_ms = (time.perf_counter() - start) * 1000.0

            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", AVG_INPUT_TOKENS)
            output_tokens = usage.get("completion_tokens", AVG_OUTPUT_TOKENS)
            cost = self._estimate_cost(provider.model, input_tokens, output_tokens)

            provider.circuit_breaker.record_success()

            return LLMResponse(
                text=text,
                model=provider.model,
                provider=provider.name,
                latency_ms=elapsed_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=cost,
            )

        except Exception as e:
            provider.circuit_breaker.record_failure()
            logger.warning(f"Provider {provider.name} failed: {e}")
            return None

    def _resolve_ephemeral_provider(
        self,
        user_provider: Optional[str] = None,
        user_model: Optional[str] = None,
        user_api_key: Optional[str] = None,
    ) -> Optional[ProviderConfig]:
        """
        Build an ephemeral ProviderConfig if the user supplied an API key.
        Strictly zero-logs the API key per Policy SEC-402.
        """
        if not user_api_key or not user_api_key.strip():
            return None

        clean_key = user_api_key.strip()
        provider_name = (user_provider or "").lower().strip()

        if not provider_name:
            if clean_key.startswith("gsk_"):
                provider_name = "groq"
            elif clean_key.startswith("sk-"):
                provider_name = "openai"
            else:
                provider_name = "groq"

        if provider_name == "groq":
            model = user_model or GROQ_MODEL or "openai/gpt-oss-120b"
            base_url = GROQ_BASE_URL
        elif provider_name == "openai":
            model = user_model or OPENAI_MODEL or "gpt-4o-mini"
            base_url = OPENAI_BASE_URL
        else:
            model = user_model or "custom-model"
            base_url = GROQ_BASE_URL

        return ProviderConfig(
            name=f"ephemeral-{provider_name}",
            api_key=clean_key,
            model=model,
            base_url=base_url,
            circuit_breaker=CircuitBreaker(name=f"ephemeral-{provider_name}"),
            priority=-1,
        )

    async def generate(
        self,
        prompt: str,
        context: str = "",
        query: str = "",
        user_provider: Optional[str] = None,
        user_model: Optional[str] = None,
        user_api_key: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate a response using the failover chain or dynamic user-provided credentials.
        Tries user key -> remote providers in order -> local simulator fallback.
        """
        self._total_calls += 1

        # 1. Force Simulator if explicitly chosen by user
        if user_provider and user_provider.lower().strip() in ("simulator", "local", "offline"):
            response = self._simulator.generate(prompt, context, query)
            if user_model:
                response.model = user_model
            self._provider_calls["local"] = self._provider_calls.get("local", 0) + 1
            self._total_latency_ms += response.latency_ms
            return response

        # 2. Try ephemeral user-provided key if supplied
        ephemeral = self._resolve_ephemeral_provider(user_provider, user_model, user_api_key)
        if ephemeral:
            response = await self._call_provider(ephemeral, prompt, context)
            if response:
                self._provider_calls[ephemeral.name] = self._provider_calls.get(ephemeral.name, 0) + 1
                self._total_cost_usd += response.estimated_cost_usd
                self._total_latency_ms += response.latency_ms
                return response

        # 3. Try each configured remote provider
        for provider in self._providers:
            target_provider = provider
            if user_model and (
                (provider.name == "groq" and any(m in user_model for m in ["llama", "mixtral", "gpt-oss"]))
                or (provider.name == "openai" and "gpt-4" in user_model)
            ):
                target_provider = ProviderConfig(
                    name=provider.name,
                    api_key=provider.api_key,
                    model=user_model,
                    base_url=provider.base_url,
                    circuit_breaker=provider.circuit_breaker,
                    priority=provider.priority,
                )

            response = await self._call_provider(target_provider, prompt, context)
            if response:
                self._provider_calls[provider.name] = self._provider_calls.get(provider.name, 0) + 1
                self._total_cost_usd += response.estimated_cost_usd
                self._total_latency_ms += response.latency_ms
                return response

        # 4. Fallback to local simulator (Render default when .env keys are unset)
        logger.info("All remote providers unavailable or unconfigured, using local simulator")
        response = self._simulator.generate(prompt, context, query)
        if user_model:
            response.model = user_model
        self._provider_calls["local"] = self._provider_calls.get("local", 0) + 1
        self._total_latency_ms += response.latency_ms
        return response

    async def generate_stream(
        self,
        prompt: str,
        context: str = "",
        query: str = "",
        user_provider: Optional[str] = None,
        user_model: Optional[str] = None,
        user_api_key: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response using SSE. Falls back to local simulator streaming.
        Supports dynamic user-provided credentials.
        """
        self._total_calls += 1

        # 1. Force Simulator if explicitly chosen by user
        if user_provider and user_provider.lower().strip() in ("simulator", "local", "offline"):
            self._provider_calls["local"] = self._provider_calls.get("local", 0) + 1
            async for token in self._simulator.generate_stream(prompt, context, query):
                yield token
            return

        # 2. Candidate providers (ephemeral user key prioritized first)
        candidate_providers: List[ProviderConfig] = []
        ephemeral = self._resolve_ephemeral_provider(user_provider, user_model, user_api_key)
        if ephemeral:
            candidate_providers.append(ephemeral)
        candidate_providers.extend(self._providers)

        for provider in candidate_providers:
            if not provider.circuit_breaker.can_execute():
                continue

            try:
                import httpx

                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                if context and context.strip():
                    formatted_user_prompt = (
                        f"=== RETRIEVED ENTERPRISE DOCUMENTATION CHUNKS ===\n"
                        f"{context.strip()}\n"
                        f"================================================\n\n"
                        f"User Inquiry: {prompt}"
                    )
                    messages.append({"role": "user", "content": formatted_user_prompt})
                else:
                    messages.append({"role": "user", "content": prompt})

                model_to_use = user_model if (ephemeral and provider == ephemeral) else provider.model

                async with httpx.AsyncClient(timeout=LLM_REQUEST_TIMEOUT) as client:
                    async with client.stream(
                        "POST",
                        f"{provider.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {provider.api_key}",
                            "Content-Type": "application/json",
                            "User-Agent": "SentinelAI-RAG/1.0",
                        },
                        json={
                            "model": model_to_use,
                            "messages": messages,
                            "temperature": 0.3,
                            "max_tokens": LLM_MAX_TOKENS,
                            "stream": True,
                        },
                    ) as response:
                        response.raise_for_status()
                        provider.circuit_breaker.record_success()
                        self._provider_calls[provider.name] = self._provider_calls.get(provider.name, 0) + 1

                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                data_str = line[6:]
                                if data_str.strip() == "[DONE]":
                                    return
                                try:
                                    data = json.loads(data_str)
                                    delta = data["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                                except (json.JSONDecodeError, KeyError):
                                    continue
                        return

            except Exception as e:
                provider.circuit_breaker.record_failure()
                logger.warning(f"Streaming from {provider.name} failed: {e}")
                continue

        # Fallback: stream from local simulator
        self._provider_calls["local"] = self._provider_calls.get("local", 0) + 1
        async for token in self._simulator.generate_stream(prompt, context, query):
            yield token

    @property
    def stats(self) -> dict:
        avg_latency = self._total_latency_ms / max(self._total_calls, 1)
        return {
            "total_calls": self._total_calls,
            "provider_calls": dict(self._provider_calls),
            "total_estimated_cost_usd": round(self._total_cost_usd, 6),
            "avg_latency_ms": round(avg_latency, 2),
            "providers_configured": len(self._providers),
            "simulator_available": True,
            "circuit_breaker_states": {
                p.name: p.circuit_breaker.stats for p in self._providers
            },
        }
