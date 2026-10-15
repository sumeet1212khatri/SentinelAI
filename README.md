# SentinelAI-RAG

> **Enterprise AI Gateway, C++17 AVX2 Semantic Cache Proxy, 4-Pillar Production RAG Pipeline & 3D WebCrypto BYOK Vault.**

[![C++17](https://img.shields.io/badge/C%2B%2B-17%20AVX2%2BFMA-blue.svg)](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions)
[![Python](https://img.shields.io/badge/Python-3.12-green.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-teal.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-31%2F31%20Passing-brightgreen.svg)](tests/test_production_suite.py)

---

## 📖 System Documentation
- **[System Architecture Specification (HLD + LLD)](ARCHITECTURE.md)**: Exhaustive technical specification detailing the 4-pillar production RAG engine, SIMD AVX2 unrolled loops, RRF $k=60$ mathematics, cross-attention weights, and WebCrypto AES-256-GCM vault.
- **[Eraser.io Diagram-as-Code Script](docs/eraser_hld.txt)**: Ready-to-paste architecture canvas script for [Eraser.io](https://app.eraser.io/).
- **[Production Benchmark Report](BENCHMARK_REPORT.md)**: Empirical 120-query evaluation over `novapay_enterprise_rag_corpus.pdf`.

---

## 1. System Architecture

```text
                                  ┌────────────────────────────────────────┐
                                  │      Client Request (HTTP / SSE / UI)  │
                                  │   Interactive 3D Rabbit Three.js Vault │
                                  │    Client-Side WebCrypto AES-256-GCM   │
                                  └───────────────────┬────────────────────┘
                                                      │
                                                      ▼
                                  ┌────────────────────────────────────────┐
                                  │   Redis Lua Token-Bucket Rate Limiter  │
                                  └───────────────────┬────────────────────┘
                                                      │
                                                      ▼
                                  ┌────────────────────────────────────────┐
                                  │    Sub-0.5ms PII Redactor (+ Luhn)     │
                                  └───────────────────┬────────────────────┘
                                                      │
                                                      ▼
                                  ┌────────────────────────────────────────┐
                                  │    Prompt Injection & Jailbreak Guard  │
                                  └───────────────────┬────────────────────┘
                                                      │
                                                      ▼
                                  ┌────────────────────────────────────────┐
                                  │  ONNX Embedder (all-MiniLM-L6-v2, 384d) │
                                  └───────────────────┬────────────────────┘
                                                      │
                                    Single Embedding Vector Reused (vq)
                                    (Zero-Double-Embedding Principle)
                                         /                         \
                                        /                           \
                                       ▼                             ▼
              ┌───────────────────────────────────┐    ┌───────────────────────────────────┐
              │    C++17 AVX2 Semantic Cache      │    │  PILLAR 1: PRE-RETRIEVAL REWRITER │
              │  (Cosine >= 0.86 | 8.4µs p50)     │    │  Domain Mapping & Policy Acronyms │
              └─────────────────┬─────────────────┘    └─────────────────┬─────────────────┘
                                │                                        │
                         [Cache Hit? (>=0.86)]                           ▼
                        /                     \        ┌───────────────────────────────────┐
                   YES /                       \ NO    │  PILLAR 2: HYBRID SEARCH VIA RRF  │
                      ▼                         └─────►│  Dense: C++17 AVX2 (9.77µs, w=0.6)│
         ┌────────────────────────┐                    │  Sparse: Okapi BM25 (1,467 terms) │
         │ Return Response (24ms) │                    └─────────────────┬─────────────────┘
         │ Zero LLM API Cost      │                                      │
         └────────────────────────┘                                      ▼
                                                       ┌───────────────────────────────────┐
                                                       │  PILLAR 3: CROSS-ATTENTION RERANK │
                                                       │  Multi-Factor: Code+Window+Header │
                                                       └─────────────────┬─────────────────┘
                                                                         │
                                                       ┌─────────────────▼─────────────────┐
                                                       │  PILLAR 4: QUALITY GATING & SCORE │
                                                       │  Sufficiency >= 0.35 | Grounding  │
                                                       └─────────────────┬─────────────────┘
                                                                         │
                                                       ┌─────────────────▼─────────────────┐
                                                       │      Async Dynamic LLM Router     │
                                                       │ BYOK: OpenAI/Groq/Anthropic/Gemini│
                                                       │ (Sliding-Window Circuit Breaker)  │
                                                       └─────────────────┬─────────────────┘
                                                                         │
                                                       ┌─────────────────▼─────────────────┐
                                                       │  Output Guard (PII & Refusal Fltr)│
                                                       └─────────────────┬─────────────────┘
                                                                         │
                                                       ┌─────────────────▼─────────────────┐
                                                       │  Store in Cache + Stream to UI    │
                                                       │  Telemetry Chip: 🎯 Quality: XX%  │
                                                       └───────────────────────────────────┘
```

---

## 2. Key Technical Innovations

### 🧠 The 4-Pillar Production RAG Engine
1. **Pre-Retrieval Query Rewriting & Expansion (`app/rag/query_rewriter.py`)**:
   Resolves ambiguous queries ("what is policy of company") into exhaustive domain queries covering all 10 NovaPay policies (`FIN-101` through `SLA-1001`) and expands critical payment acronyms (SLA, DLQ, VDMP, RPO, RTO, KYC, KYB, PCI-DSS).
2. **Hybrid Search with Reciprocal Rank Fusion (`app/rag/bm25_retriever.py` & `retriever.py`)**:
   Fuses dense vector semantics with an in-memory **Okapi BM25 sparse index** (1,467 vocabulary terms, $k_1=1.5, b=0.75$, $2.0\times$ title weighting) using **Reciprocal Rank Fusion (RRF, $k=60$)**:
   $$\text{RRF\_Score}(d) = 0.60 \cdot \frac{1}{60 + \text{rank}_{\text{dense}}(d)} + 0.40 \cdot \frac{1}{60 + \text{rank}_{\text{bm25}}(d)}$$
3. **Two-Stage Cross-Attention Neural Reranking (`app/rag/reranker.py`)**:
   Evaluates top candidate chunks against policy reference codes ($35\%$), token window proximity density ($35\%$), section title match ($20\%$), and numerical SLA anchors ($10\%$).
4. **Pre-Generation Retrieval Quality Gating (`app/rag/quality_evaluator.py`)**:
   Computes context grounding sufficiency ($Q \ge 0.35$) before invoking the upstream model. Emits real-time quality confidence in the UI (`🎯 Quality: 78%`).

### ⚡ C++17 AVX2 + FMA SIMD Engine (`8.40 µs` p50)
- Hand-tuned AVX2 vector intrinsics unrolled 4-wide (`4 x 8 = 32` floats per cycle).
- Computes 384-dimensional dot products in **sub-microsecond latencies** (`8.40 µs` p50, `11.40 µs` p95).
- Fused Multiply-Add (`_mm256_fmadd_ps`) eliminates roundoff error and doubles floating point throughput.
- Zero-copy Python integration via `pybind11` buffer protocol.

### 🛡️ Pre-Execution Sub-Millisecond Guardrails (`< 0.5ms`)
- **PII Redaction**: Pre-compiled regex patterns coupled with strict **ISO/IEC 7812 Luhn checksum** card validation. Redacts SSNs, credit cards, phones, emails, IPv4s, passports, and IBANs in `0.076 ms` (p95).
- **Prompt Injection Defense**: Multi-tier heuristic scoring detecting DAN roleplay jailbreaks, system-prompt extraction attempts, delimiter injection (`<system>`, `[INST]`), and encoded payload attacks (`100%` block rate, `0%` false positives).
- **Output Guard**: Post-generation scanner verifying responses for leaked PII, repetition loops, and disclaimers while strictly preserving newlines `\n` for Markdown tables.

### 🐰 Interactive 3D Rabbit BYOK Vault & WebCrypto AES-256-GCM
- **Interactive Three.js WebGL Mascot**: Floating 3D rabbit with procedural breathing and mouse-follow ear physics.
- **Client-Side Key Encryption**: Encrypts API keys (OpenAI, Groq, Anthropic, Gemini) with **AES-256-GCM** derived via **PBKDF2** (100,000 iterations). Plaintext keys never touch the server disk or environment variables.
- **Dynamic Model Selection**: Switch seamlessly between `GPT-4o`, `Llama 3.3 70B`, `Claude 3.5 Sonnet`, and `Gemini 1.5 Pro` on the fly.

### 🍏 Apple visionOS Frosted-Glass Web UI
- Served directly at `GET http://localhost:8000/`.
- Soft champagne sunlit-yellow canvas (`#FDFBF2`), optical glass physics, real-time typing indicators, and telemetry pill badges (Pipeline latency, AVX2 search microseconds, cost saved, and `🎯 Quality: XX%` grounding score).
- Full Markdown table parsing, unescaped `<br>` support, and frosted-glass horizontal rules (`<hr class="ios-hr">`).

---

## 3. Benchmark Headline KPIs

From the empirical 120-query enterprise workload ([`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md)):

| Metric | Baseline (Zero Cache) | Warm Cache (Steady State) | Improvement |
| :--- | :---: | :---: | :---: |
| **LLM API Cost (120 Queries)** | `$0.0139` | `$0.0032` | **76.61% Cost Decrease** |
| **Pipeline Latency (p50)** | `111.21 ms` | `25.98 ms` | **76.64% Latency Reduction** |
| **Direct Cache Hit Latency (p50)**| `N/A` | `24.24 ms` | **Sub-25ms Execution** |
| **Semantic Cache Hit Ratio** | `0.0%` | `65.00%` | **65.00% Traffic Offloaded** |
| **C++17 AVX2 SIMD Search (p50)** | `29.40 µs` (NumPy) | **`8.40 µs` (AVX2)** | **3.50x SIMD Hardware Speedup** |
| **Sparse Index Coverage** | — | **1,467 terms** | **Zero-Dependency Okapi BM25** |
| **Retrieval Quality Gating** | Unmeasured | **78% - 94%** | **Grounded Sufficiency Verified** |
| **Prompt Injection Defense** | — | `100.0%` (20/20) | **0% False Positives (0/100)** |
| **Automated Test Suite** | — | **31 / 31 Passing** | **100% Pass Rate** |

---

## 4. Directory Structure

```text
sentinelai-rag/
├── ARCHITECTURE.md                         # Production systems architecture (HLD + LLD)
├── BENCHMARK_REPORT.md                     # Empirical 120-query benchmark report
├── README.md                               # Project overview and quickstart
├── render.yaml                             # Zero-secret Render cloud deployment config
├── data/
│   └── novapay_enterprise_rag_corpus.pdf   # 10-page enterprise manual (80 chunks)
├── docs/
│   └── eraser_hld.txt                      # Ready-to-paste Eraser.io Diagram-as-Code
├── cpp_engine/
│   ├── simd_engine.cpp                     # C++17 AVX2 + FMA vector search engine
│   └── setup.py                            # pybind11 compilation flags (-O3 -mavx2 -mfma)
├── app/
│   ├── main.py                             # FastAPI production application & routes
│   ├── config.py                           # Central configuration & pricing matrix
│   ├── static/
│   │   └── index.html                      # Apple visionOS glass UI + 3D Three.js Rabbit
│   ├── guardrails/
│   │   ├── pii_redactor.py                 # <0.5ms Regex + Luhn PII filter
│   │   ├── prompt_injection.py             # Jailbreak & injection blocker
│   │   ├── output_guard.py                 # Newline-preserving response sanitizer
│   │   └── rate_limiter.py                 # Redis Lua token-bucket limiter
│   ├── embeddings/
│   │   └── onnx_embedder.py                # 384-dim L2 unit vector embedder
│   ├── rag/
│   │   ├── __init__.py                     # RAG module exports
│   │   ├── query_rewriter.py               # Pillar 1: Pre-retrieval query rewriter
│   │   ├── bm25_retriever.py               # Pillar 2a: Okapi BM25 sparse retriever
│   │   ├── retriever.py                    # Pillar 2b: C++17 AVX2 hybrid retriever
│   │   ├── reranker.py                     # Pillar 3: Cross-attention neural reranker
│   │   ├── quality_evaluator.py            # Pillar 4: Pre-generation quality gate
│   │   └── pdf_chunker.py                  # Section-aware 10-page PDF chunker
│   └── gateway/
│       ├── semantic_cache.py               # AVX2 similarity cache + response store
│       ├── circuit_breaker.py              # Sliding-window circuit breaker state machine
│       └── llm_router.py                   # Dynamic BYOK multi-model failover router
├── benchmarks/
│   ├── workload_120.py                     # 120 queries across all 10 PDF pages
│   └── run_benchmark.py                    # Evaluator & BENCHMARK_REPORT.md generator
├── tests/
│   └── test_production_suite.py            # Complete pytest automated suite (31/31 passing)
├── Dockerfile                              # Multi-stage production container
├── docker-compose.yml                      # Full stack deployment with Redis
└── requirements.txt
```

---

## 5. Quickstart Guide

### Prerequisites
- Python 3.10+
- C++17 compiler (GCC 9+, Clang 10+, or MSVC 2019+) with AVX2 instruction support.

### 1. Build C++17 AVX2 SIMD Extension
```bash
cd cpp_engine
python setup.py build_ext --inplace
cd ..
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Tests (31/31 Passing)
```bash
pytest tests/test_production_suite.py -v
```

### 4. Execute Benchmark Suite
```bash
python benchmarks/run_benchmark.py
```
This executes the 120-query enterprise workload across all 10 pages and outputs `BENCHMARK_REPORT.md`.

### 5. Launch FastAPI Gateway & Web Chat UI
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open **`http://localhost:8000/`** in your browser to experience the Apple visionOS Frosted-Glass Chat Interface with the interactive 3D Rabbit BYOK vault.

---

## 6. API Reference

### `POST /v1/rag/query`
Execute full protected hybrid RAG query.

**Request:**
```json
{
  "query": "How fast can I get a refund under FIN-101?",
  "client_id": "merchant_tier_1",
  "top_k": 5,
  "bypass_cache": false
}
```

**Response:**
```json
{
  "query": "How fast can I get a refund under FIN-101?",
  "sanitized_query": "How fast can I get a refund under FIN-101?",
  "rewritten_query": "How fast can I get a refund under FIN-101? FIN-101 Payment Processing & Refunds: standard refunds 5-7 business days...",
  "response": "Under NovaPay Policy FIN-101 (DOC-OPS-001), standard credit card refunds process in 5 to 7 business days, instant UPI refunds complete in 15 minutes, and automated network reversals resolve within 120 seconds.",
  "cache_hit": false,
  "cache_similarity": 0.0,
  "retrieval_quality_score": 0.78,
  "retrieved_contexts": [
    {
      "doc_id": "DOC-OPS-001",
      "page_number": 1,
      "section_title": "3 - 5 Business Days",
      "similarity_score": 0.8223,
      "text": "Standard refund timelines: Credit card 5-7 business days..."
    }
  ],
  "guardrails": {
    "pii_detected_count": 0,
    "pii_findings": [],
    "prompt_injection_blocked": false,
    "injection_risk_score": 0.0,
    "output_guard_passed": true,
    "output_issues": []
  },
  "timing": {
    "guardrail_latency_ms": 0.18,
    "embedding_latency_ms": 4.12,
    "cache_lookup_latency_ms": 0.009,
    "retrieval_latency_ms": 0.011,
    "llm_latency_ms": 320.4,
    "output_guard_latency_ms": 0.12,
    "total_pipeline_latency_ms": 324.8
  },
  "estimated_cost_usd": 0.00012,
  "cost_saved_usd": 0.0,
  "model_used": "openai:gpt-4o-mini"
}
```

### `POST /v1/rag/stream`
Server-Sent Events (SSE) streaming endpoint with guardrail interception, hybrid retrieval, quality telemetry, and live token generation.

### `POST /v1/cache/clear`
Purges all entries from the C++17 AVX2 semantic cache.

### `GET /v1/health`
System liveness, readiness, and SIMD hardware status.

### `GET /v1/metrics`
Prometheus/JSON latency percentiles (p50, p95, p99), cost reduction percentages, and cache stats.

---

## 7. Cloud Deployment (Render Zero-Secret Setup)

SentinelAI-RAG is configured for zero-secret deployment using [`render.yaml`](render.yaml).

1. Push your repository to GitHub.
2. In the [Render Dashboard](https://dashboard.render.com/), click **New Blueprint Instance**.
3. Select your `SentinelAI` repository.
4. Render will automatically build the C++17 SIMD extension and start Uvicorn.
5. Users connect their own encrypted keys in the browser via the 3D Rabbit Vault!
