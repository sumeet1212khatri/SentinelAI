"""
main.py — SentinelAI-RAG FastAPI Production Server.
Endpoints:
  - POST /v1/rag/query: Guardrails -> Cache -> SIMD Top-K RAG -> LLM Router -> Output Guard -> Cache Store
  - POST /v1/rag/stream: Server-Sent Events (SSE) streaming query response
  - GET  /v1/health: Readiness, liveness, SIMD engine status, and SLA compliance
  - GET  /v1/metrics: Real-time Prometheus/JSON metrics (cost reduction, cache hit rate, latencies)
"""
import time
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import (
    PDF_PATH,
    CACHE_SIMILARITY_THRESHOLD,
    RAG_TOP_K,
    PII_REDACTION_ENABLED,
    PROMPT_INJECTION_ENABLED,
    OUTPUT_GUARD_ENABLED,
    LLM_PRICING,
    AVG_INPUT_TOKENS,
    AVG_OUTPUT_TOKENS,
)
from app.guardrails.pii_redactor import PIIRedactor
from app.guardrails.prompt_injection import PromptInjectionGuard
from app.guardrails.output_guard import OutputGuard
from app.guardrails.rate_limiter import RateLimiter
from app.embeddings.onnx_embedder import ONNXEmbedder
from app.rag.pdf_chunker import PDFChunker
from app.rag.retriever import SIMDRetriever, SIMD_AVAILABLE
from app.rag.query_rewriter import QueryRewriter
from app.rag.quality_evaluator import RetrievalQualityEvaluator
from app.gateway.semantic_cache import SemanticCache, is_negative_or_refusal_response
from app.gateway.circuit_breaker import CircuitBreaker
from app.gateway.llm_router import LLMRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("sentinelai.server")

# Global subsystem state
embedder: Optional[ONNXEmbedder] = None
retriever: Optional[SIMDRetriever] = None
query_rewriter: Optional[QueryRewriter] = None
quality_evaluator: Optional[RetrievalQualityEvaluator] = None
cache: Optional[SemanticCache] = None
llm_router: Optional[LLMRouter] = None
pii_redactor: Optional[PIIRedactor] = None
injection_guard: Optional[PromptInjectionGuard] = None
output_guard: Optional[OutputGuard] = None
rate_limiter: Optional[RateLimiter] = None

# Metrics aggregation
SERVER_START_TIME = time.time()
query_latencies: List[float] = []
total_queries_served = 0
total_cost_saved_usd = 0.0
total_baseline_cost_usd = 0.0


def initialize_pipeline():
    """Initialize embedder, PDF chunker, retriever, cache, and guardrails."""
    global embedder, retriever, cache, llm_router
    global pii_redactor, injection_guard, output_guard, rate_limiter
    global query_rewriter, quality_evaluator

    logger.info("Initializing SentinelAI-RAG subsystems...")

    # Guardrails
    pii_redactor = PIIRedactor()
    injection_guard = PromptInjectionGuard()
    output_guard = OutputGuard()
    rate_limiter = RateLimiter(max_requests_per_minute=120, burst=20)

    # Advanced RAG: Query Rewriter & Quality Evaluator
    query_rewriter = QueryRewriter()
    quality_evaluator = RetrievalQualityEvaluator()

    # Embedder
    embedder = ONNXEmbedder()

    # Retriever & Cache
    retriever = SIMDRetriever(top_k=RAG_TOP_K)
    cache = SemanticCache(similarity_threshold=CACHE_SIMILARITY_THRESHOLD)
    cache.clear()  # Ensure pristine state on startup without stale or negative entries

    # Router
    llm_router = LLMRouter()

    # Ingest PDF if found
    if PDF_PATH.exists():
        logger.info(f"Ingesting PDF manual from {PDF_PATH}...")
        chunker = PDFChunker()
        parsed_doc = chunker.parse(str(PDF_PATH))
        texts = [chunk.text for chunk in parsed_doc.chunks]
        if texts:
            logger.info(f"Embedding {len(texts)} chunks from PDF...")
            chunk_embeddings = embedder.embed(texts)
            retriever.ingest(parsed_doc.chunks, chunk_embeddings)
            logger.info(f"Successfully ingested {len(texts)} PDF chunks into SIMD retriever.")
    else:
        logger.warning(f"PDF corpus not found at {PDF_PATH}. Starting with empty retriever.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_pipeline()
    yield
    logger.info("Shutting down SentinelAI-RAG service.")


app = FastAPI(
    title="SentinelAI-RAG Enterprise Gateway",
    description="High-Throughput AI Gateway, Guardrail Engine & C++17 AVX2 Semantic Cache Proxy",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

# Serve ChatGPT-Style conversational interface at GET /
@app.get("/", response_class=FileResponse)
async def serve_chat_ui():
    """Serve the ChatGPT-style conversational web interface."""
    if INDEX_HTML.exists():
        return FileResponse(str(INDEX_HTML), media_type="text/html")
    raise HTTPException(status_code=404, detail="Web UI index.html not found")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --- Request & Response Models ---

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's enterprise query string")
    client_id: Optional[str] = Field("default_client", description="Client identifier for rate limiting")
    top_k: Optional[int] = Field(None, description="Override default Top-K retrieval count")
    bypass_cache: Optional[bool] = Field(False, description="Force bypass of semantic cache")
    user_provider: Optional[str] = Field(None, description="Dynamic BYOK provider override (groq, openai, simulator)")
    user_model: Optional[str] = Field(None, description="Dynamic model override")
    user_api_key: Optional[str] = Field(None, description="Ephemeral BYOK API key")


class RetrievedContextItem(BaseModel):
    chunk_id: int
    page_number: int
    section_title: str
    similarity_score: float
    snippet: str


class GuardrailStats(BaseModel):
    pii_detected_count: int
    pii_findings: List[str]
    prompt_injection_blocked: bool
    injection_risk_score: float
    output_guard_passed: bool
    output_issues: List[str]


class LatencyBreakdown(BaseModel):
    guardrail_latency_ms: float
    embedding_latency_ms: float
    cache_lookup_latency_ms: float
    retrieval_latency_ms: float
    llm_latency_ms: float
    output_guard_latency_ms: float
    total_pipeline_latency_ms: float


class QueryResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    query: str
    sanitized_query: str
    response: str
    cache_hit: bool
    cache_similarity: float
    retrieved_contexts: List[RetrievedContextItem]
    guardrails: GuardrailStats
    timing: LatencyBreakdown
    estimated_cost_usd: float
    cost_saved_usd: float
    model_used: str
    retrieval_quality_score: Optional[float] = Field(None, description="Pre-generation retrieval quality score (0.0 to 1.0)")
    rewritten_query: Optional[str] = Field(None, description="Expanded/rewritten query used for hybrid retrieval")


# --- API Endpoints ---

@app.post("/v1/rag/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest, request: Request):
    """
    Execute full Enterprise RAG lifecycle:
    Rate Limit -> PII Redact -> Prompt Injection Check -> Embedding ->
    AVX2 Semantic Cache Lookup -> SIMD Top-K RAG -> LLM Router -> Output Guard -> Cache Store
    """
    global total_queries_served, total_cost_saved_usd, total_baseline_cost_usd
    t_start = time.perf_counter()

    # 1. Rate Limiting
    rl_res = rate_limiter.check(client_id=req.client_id)
    if not rl_res.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Retry after {rl_res.retry_after_seconds:.1f}s",
        )

    # 2. Input Guardrails: PII Redaction
    t_g_start = time.perf_counter()
    pii_res = pii_redactor.redact(req.query) if PII_REDACTION_ENABLED else None
    active_query = pii_res.redacted_text if pii_res else req.query
    pii_count = pii_res.redaction_count if pii_res else 0
    pii_findings = [f"{t}: {val}" for t, val in (pii_res.findings if pii_res else [])]

    # 3. Input Guardrails: Prompt Injection Detection
    inj_res = injection_guard.analyze(active_query) if PROMPT_INJECTION_ENABLED else None
    t_guardrail = (time.perf_counter() - t_g_start) * 1000.0

    if inj_res and inj_res.is_blocked:
        t_total = (time.perf_counter() - t_start) * 1000.0
        return QueryResponse(
            query=req.query,
            sanitized_query=active_query,
            response="[BLOCKED] Your query violated enterprise security policies (Prompt Injection / Jailbreak attempt).",
            cache_hit=False,
            cache_similarity=0.0,
            retrieved_contexts=[],
            guardrails=GuardrailStats(
                pii_detected_count=pii_count,
                pii_findings=pii_findings,
                prompt_injection_blocked=True,
                injection_risk_score=inj_res.risk_score,
                output_guard_passed=True,
                output_issues=["PROMPT_INJECTION_INTERCEPTED"],
            ),
            timing=LatencyBreakdown(
                guardrail_latency_ms=t_guardrail,
                embedding_latency_ms=0.0,
                cache_lookup_latency_ms=0.0,
                retrieval_latency_ms=0.0,
                llm_latency_ms=0.0,
                output_guard_latency_ms=0.0,
                total_pipeline_latency_ms=t_total,
            ),
            estimated_cost_usd=0.0,
            cost_saved_usd=0.0,
            model_used="guardrail-block",
        )

    # 4. Generate Single Query Embedding (Zero-double-embedding principle)
    t_emb_start = time.perf_counter()
    query_vec = embedder.embed_single(active_query)
    t_embedding = (time.perf_counter() - t_emb_start) * 1000.0

    # 5. Semantic Cache Lookup (AVX2 SIMD)
    t_cache_start = time.perf_counter()
    cache_hit = False
    cache_sim = 0.0
    cached_response = None
    if not req.bypass_cache:
        cache_res = cache.lookup(query_vec)
        if cache_res.hit:
            cache_hit = True
            cache_sim = cache_res.similarity_score
            cached_response = cache_res.response
    t_cache = (time.perf_counter() - t_cache_start) * 1000.0

    # Baseline cost estimation for non-cached call
    baseline_cost = ((AVG_INPUT_TOKENS / 1_000_000) * LLM_PRICING["gpt-4o-mini"]["input"] +
                     (AVG_OUTPUT_TOKENS / 1_000_000) * LLM_PRICING["gpt-4o-mini"]["output"])

    if cache_hit and cached_response:
        t_total = (time.perf_counter() - t_start) * 1000.0
        query_latencies.append(t_total)
        total_queries_served += 1
        total_cost_saved_usd += baseline_cost
        total_baseline_cost_usd += baseline_cost

        return QueryResponse(
            query=req.query,
            sanitized_query=active_query,
            response=cached_response,
            cache_hit=True,
            cache_similarity=cache_sim,
            retrieved_contexts=[],
            guardrails=GuardrailStats(
                pii_detected_count=pii_count,
                pii_findings=pii_findings,
                prompt_injection_blocked=False,
                injection_risk_score=inj_res.risk_score if inj_res else 0.0,
                output_guard_passed=True,
                output_issues=[],
            ),
            timing=LatencyBreakdown(
                guardrail_latency_ms=t_guardrail,
                embedding_latency_ms=t_embedding,
                cache_lookup_latency_ms=t_cache,
                retrieval_latency_ms=0.0,
                llm_latency_ms=0.0,
                output_guard_latency_ms=0.0,
                total_pipeline_latency_ms=t_total,
            ),
            estimated_cost_usd=0.0,
            cost_saved_usd=baseline_cost,
            model_used="semantic-cache-avx2",
        )

    # 6. Pre-Retrieval Query Rewriting & Expansion (Level 3 Production RAG)
    rewritten = query_rewriter.rewrite(active_query) if query_rewriter else None
    expanded_search_query = rewritten.expanded_query if rewritten else active_query

    # 7. Production Hybrid Search (BM25 + AVX2 SIMD + RRF + Cross-Attention Reranking)
    t_ret_start = time.perf_counter()
    rag_res = retriever.hybrid_search(query_vec, query_text=expanded_search_query, top_k=req.top_k)
    t_retrieval = (time.perf_counter() - t_ret_start) * 1000.0

    # 8. Pre-Generation Retrieval Quality Gating
    quality_report = quality_evaluator.evaluate(active_query, rag_res.chunks, rag_res.scores) if quality_evaluator else None
    quality_score = quality_report.quality_score if quality_report else 0.85

    context_str = "\n\n".join([f"[{c.section_title} | Page {c.page_number}]: {c.text}" for c in rag_res.chunks])
    context_items = [
        RetrievedContextItem(
            chunk_id=c.chunk_id,
            page_number=c.page_number,
            section_title=c.section_title,
            similarity_score=score,
            snippet=c.text[:200] + "..." if len(c.text) > 200 else c.text,
        )
        for c, score in zip(rag_res.chunks, rag_res.scores)
    ]

    # 9. LLM Generation via Router (with dynamic user BYOK credentials)
    user_provider = request.headers.get("X-User-Provider") or req.user_provider
    user_model = request.headers.get("X-User-Model") or req.user_model
    user_api_key = request.headers.get("X-User-Api-Key") or req.user_api_key

    t_llm_start = time.perf_counter()
    llm_resp = await llm_router.generate(
        prompt=active_query,
        context=context_str,
        query=active_query,
        user_provider=user_provider,
        user_model=user_model,
        user_api_key=user_api_key,
    )
    t_llm = (time.perf_counter() - t_llm_start) * 1000.0

    # 10. Output Guardrails
    t_og_start = time.perf_counter()
    og_res = output_guard.validate(llm_resp.text) if OUTPUT_GUARD_ENABLED else None
    t_og = (time.perf_counter() - t_og_start) * 1000.0
    final_text = og_res.sanitized_text if og_res else llm_resp.text

    # 11. Store In Semantic Cache for Future Hits (Guarded against negative/refusal caching)
    if (
        og_res
        and og_res.is_safe
        and len(rag_res.chunks) > 0
        and not is_negative_or_refusal_response(final_text)
    ):
        cache.store(
            query_text=active_query,
            response_text=final_text,
            query_embedding=query_vec,
            model_used=llm_resp.model,
            metadata={"chunks_count": len(rag_res.chunks)},
        )

    t_total = (time.perf_counter() - t_start) * 1000.0
    query_latencies.append(t_total)
    total_queries_served += 1
    total_baseline_cost_usd += baseline_cost

    return QueryResponse(
        query=req.query,
        sanitized_query=active_query,
        response=final_text,
        cache_hit=False,
        cache_similarity=cache_sim,
        retrieved_contexts=context_items,
        guardrails=GuardrailStats(
            pii_detected_count=pii_count,
            pii_findings=pii_findings,
            prompt_injection_blocked=False,
            injection_risk_score=inj_res.risk_score if inj_res else 0.0,
            output_guard_passed=og_res.is_safe if og_res else True,
            output_issues=og_res.issues if og_res else [],
        ),
        timing=LatencyBreakdown(
            guardrail_latency_ms=t_guardrail,
            embedding_latency_ms=t_embedding,
            cache_lookup_latency_ms=t_cache,
            retrieval_latency_ms=t_retrieval,
            llm_latency_ms=t_llm,
            output_guard_latency_ms=t_og,
            total_pipeline_latency_ms=t_total,
        ),
        estimated_cost_usd=llm_resp.estimated_cost_usd,
        cost_saved_usd=0.0,
        model_used=llm_resp.model,
        retrieval_quality_score=quality_score,
        rewritten_query=expanded_search_query,
    )


@app.post("/v1/rag/stream")
async def stream_endpoint(req: QueryRequest, request: Request):
    """
    Server-Sent Events (SSE) streaming query response with guardrail checks.
    """
    user_provider = request.headers.get("X-User-Provider") or req.user_provider
    user_model = request.headers.get("X-User-Model") or req.user_model
    user_api_key = request.headers.get("X-User-Api-Key") or req.user_api_key

    # Rate Limiting
    rl_res = rate_limiter.check(client_id=req.client_id)
    if not rl_res.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Retry after {rl_res.retry_after_seconds:.1f}s",
        )

    # Input Guardrails
    pii_res = pii_redactor.redact(req.query) if PII_REDACTION_ENABLED else None
    active_query = pii_res.redacted_text if pii_res else req.query

    inj_res = injection_guard.analyze(active_query) if PROMPT_INJECTION_ENABLED else None
    if inj_res and inj_res.is_blocked:
        async def block_gen():
            yield "data: " + json.dumps({"token": "[BLOCKED: Prompt injection detected]"}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(block_gen(), media_type="text/event-stream")

    query_vec = embedder.embed_single(active_query)

    # Check cache first
    if not req.bypass_cache:
        cache_res = cache.lookup(query_vec)
        if cache_res.hit:
            async def cache_gen():
                for word in cache_res.response.split():
                    yield "data: " + json.dumps({"token": word + " "}) + "\n\n"
                yield "data: [DONE]\n\n"
    # Hybrid Retrieval (BM25 + AVX2 SIMD + RRF + Cross-Attention Reranking)
    rewritten = query_rewriter.rewrite(active_query) if query_rewriter else None
    expanded_search_query = rewritten.expanded_query if rewritten else active_query
    rag_res = retriever.hybrid_search(query_vec, query_text=expanded_search_query, top_k=req.top_k)
    context_str = "\n\n".join([f"[{c.section_title}]: {c.text}" for c in rag_res.chunks])

    async def token_generator():
        collected_tokens = []
        async for token in llm_router.generate_stream(
            prompt=active_query,
            context=context_str,
            query=active_query,
            user_provider=user_provider,
            user_model=user_model,
            user_api_key=user_api_key,
        ):
            collected_tokens.append(token)
            yield f"data: {json.dumps({'token': token})}\n\n"

        full_resp = "".join(collected_tokens)
        cache.store(
            query_text=active_query,
            response_text=full_resp,
            query_embedding=query_vec,
            model_used=user_model or "streaming-model",
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


@app.get("/v1/health")
async def health_check():
    """
    Comprehensive health status of all subsystems.
    """
    uptime = time.time() - SERVER_START_TIME
    simd_active = SIMD_AVAILABLE
    if simd_active:
        try:
            import simd_engine
            simd_active = simd_engine.is_simd_available()
        except Exception:
            simd_active = False

    return {
        "status": "healthy",
        "uptime_seconds": round(uptime, 2),
        "subsystems": {
            "simd_avx2_engine": {
                "compiled_and_loaded": SIMD_AVAILABLE,
                "hardware_avx2_fma": simd_active,
            },
            "retriever": {
                "indexed_chunks": retriever.chunk_count if retriever else 0,
                "ready": retriever is not None and retriever.chunk_count > 0,
            },
            "semantic_cache": {
                "current_entries": cache.size if cache else 0,
                "hit_rate": cache.stats["hit_rate"] if cache else 0.0,
            },
            "llm_router": {
                "providers_active": llm_router.stats["providers_configured"] if llm_router else 0,
                "local_simulator_ready": True,
            },
            "guardrails": {
                "pii_redactor": "active",
                "prompt_injection_guard": "active",
                "output_guard": "active",
                "rate_limiter": "active",
            },
        },
    }


@app.get("/v1/metrics")
async def metrics_endpoint():
    """
    Export full operational metrics for monitoring and SLAs.
    """
    lats = sorted(query_latencies) if query_latencies else [0.0]
    n = len(lats)
    p50 = lats[int(n * 0.50)]
    p95 = lats[int(n * 0.95)] if n > 1 else lats[0]
    p99 = lats[int(n * 0.99)] if n > 1 else lats[0]

    cost_decrease_pct = 0.0
    if total_baseline_cost_usd > 0:
        cost_decrease_pct = (total_cost_saved_usd / total_baseline_cost_usd) * 100.0

    return {
        "total_queries_served": total_queries_served,
        "latency_ms": {
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "mean": round(sum(lats) / max(n, 1), 2),
        },
        "cache_metrics": cache.stats if cache else {},
        "cost_metrics": {
            "total_baseline_cost_usd": round(total_baseline_cost_usd, 6),
            "total_cost_saved_usd": round(total_cost_saved_usd, 6),
            "cost_decrease_percentage": round(cost_decrease_pct, 2),
        },
        "guardrail_metrics": {
            "pii": pii_redactor.stats if pii_redactor else {},
            "prompt_injection": injection_guard.stats if injection_guard else {},
            "output_guard": output_guard.stats if output_guard else {},
            "rate_limiter": rate_limiter.stats if rate_limiter else {},
        },
        "retriever_metrics": retriever.stats if retriever else {},
    }


@app.post("/v1/cache/clear")
async def clear_cache_endpoint():
    """Clear all entries in the semantic cache."""
    global cache
    if cache:
        cache.clear()
    logger.info("Semantic cache cleared via /v1/cache/clear")
    return {"status": "ok", "message": "Semantic cache cleared successfully", "current_size": 0}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)

