# SentinelAI-RAG: Production Systems Architecture Specification
**Enterprise AI Gateway, C++17 AVX2+FMA Semantic Cache Proxy, 4-Pillar Production RAG Engine & 3D WebCrypto BYOK Vault**

---

## 1. Executive Architectural Overview & Core Philosophies

**SentinelAI-RAG** is an ultra-high-throughput enterprise AI gateway and Retrieval-Augmented Generation (RAG) proxy designed to eliminate redundant LLM API expenditure, enforce sub-millisecond security guardrails, execute sub-10µs SIMD vector operations, and guarantee retrieval precision through a complete 4-pillar production RAG pipeline.

```text
               ┌──────────────────────────────────────────────────────────────┐
               │                     CLIENT INTERFACES                        │
               │   Apple visionOS Glass Web UI  •  REST API  •  SSE Stream   │
               │        Interactive 3D Rabbit Vault (Three.js WebGL)          │
               │             Client-Side WebCrypto AES-256-GCM                │
               └──────────────────────────────┬───────────────────────────────┘
                                              │
                    ┌─────────────────────────▼─────────────────────────┐
                    │     LAYER 1: PRE-INFERENCE GUARDRAILS (<0.5ms)    │
                    │   Redis Lua Token-Bucket  •  Prompt Injection     │
                    │   Luhn-Verified PII Redactor (Policy SEC-402)     │
                    └─────────────────────────┬─────────────────────────┘
                                              │
                    ┌─────────────────────────▼─────────────────────────┐
                    │    LAYER 2: SINGLE-PASS EMBEDDER (all-MiniLM)     │
                    │   384-Dim L2 Unit Vector Normalization (vq)       │
                    └─────────────────────────┬─────────────────────────┘
                                              │
                        Single Normalized Vector Reused (vq)
                        (Zero-Double-Embedding Invariant)
                                     /        \
                                    /          \
    ┌──────────────────────────────▼──┐      ┌──▼──────────────────────────────┐
    │  C++17 AVX2+FMA SEMANTIC CACHE  │      │  PILLAR 1: PRE-RETRIEVAL        │
    │  Cosine Sim >= 0.86 (8.40 µs)   │      │  QUERY REWRITER & EXPANDER      │
    └─────────────────┬───────────────┘      │  (Domain Mapping & Acronyms)    │
                      │                      └──┬──────────────────────────────┘
               [Cache Hit? >=0.86]              │ (On Cache Miss Only)
               /               \                ▼
         YES  /                 \ NO ┌─────────────────────────────────────────┐
             ▼                   └──►│  PILLAR 2: HYBRID SEARCH VIA RRF (k=60) │
   ┌───────────────────┐             │  Dense: C++17 AVX2 SIMD (9.77 µs, w=0.6)│
   │ 24.2ms Direct Hit │             │  Sparse: Okapi BM25 (1,467 terms, w=0.4)│
   │ $0.0000 API Cost  │             └──────────────────┬──────────────────────┘
   └───────────────────┘                                │
                                     ┌──────────────────▼──────────────────────┐
                                     │  PILLAR 3: CROSS-ATTENTION RERANKER     │
                                     │  Multi-Factor: Code + Window + Header   │
                                     └──────────────────┬──────────────────────┘
                                                        │
                                     ┌──────────────────▼──────────────────────┐
                                     │  PILLAR 4: RETRIEVAL QUALITY GATING     │
                                     │  Sufficiency Score >= 0.35 & Grounding  │
                                     └──────────────────┬──────────────────────┘
                                                        │ (Grounded Chunks Injected)
                                     ┌──────────────────▼──────────────────────┐
                                     │ RESILIENT MULTI-PROVIDER LLM ROUTER     │
                                     │ BYOK Routing: OpenAI/Groq/Anthropic/Gem │
                                     │ Sliding-Window Circuit Breakers (<50ms) │
                                     └──────────────────┬──────────────────────┘
                                                        │
                                     ┌──────────────────▼──────────────────────┐
                                     │ POST-GENERATION OUTPUT GUARDRAILS       │
                                     │ Markdown Preserving & Negative-Cache Fltr│
                                     └──────────────────┬──────────────────────┘
                                                        │
                                     ┌──────────────────▼──────────────────────┐
                                     │ POPULATE CACHE & STREAM TO CLIENT       │
                                     │ Live Telemetry: 🎯 Quality Score %      │
                                     └─────────────────────────────────────────┘
```

### Core Engineering Invariants

1. **The Zero-Double-Embedding Principle ($O(1)$ Vectorization)**:
   A critical performance anti-pattern in naive RAG architectures is computing the dense query embedding vector twice: once for semantic cache lookup, and a second time for document retrieval. SentinelAI-RAG generates the 384-dimensional dense query vector $\mathbf{v}_q$ **exactly once** via an optimized local ONNX embedder. That single L2-normalized vector is passed by reference across both the C++17 AVX2 Semantic Cache and the AVX2 SIMD Dense Retriever.

2. **Pre-Inference Defense-in-Depth (<0.5ms SLA)**:
   Zero untrusted user input reaches the vectorizer or upstream LLM without passing through three deterministic, non-LLM security filters:
   - **Distributed Atomic Rate Limiter** via Redis Lua script (Token-bucket algorithm supporting 60, 600, and 3,000 RPM tiers).
   - **Multi-Vector Prompt Injection Blocker** (100% interception of jailbreak, DAN, system prompt exfiltration, and delimiter injection).
   - **Luhn-Validated PII Redactor** (ISO/IEC 7812 card verification, SSN, IPv4, phone, and credentials masked in 94.7 µs p95).

3. **C++17 AVX2 + Fused Multiply-Add (FMA) Vector Hardware Acceleration**:
   Vector cosine similarity calculations are delegated to a custom C++17 shared library compiled with `-O3 -mavx2 -mfma`. By leveraging 256-bit YMM SIMD registers and unrolling dot-product loops 4-wide (processing 32 floating-point values per instruction cycle), cache search executes in **8.40 µs (p50)** and Top-K RAG retrieval across 80 PDF chunks completes in **9.77 µs (p50)**.

4. **The 4-Pillar Production RAG Pipeline**:
   Instead of naively feeding raw user queries into an index and trusting whatever comes out, SentinelAI-RAG enforces a strict 4-stage pipeline:
   - **Pillar 1: Pre-retrieval Query Rewriting**: Expands colloquial, ambiguous, or truncated questions into rich domain representations with explicit policy codes (`FIN-101` to `SLA-1001`) and acronyms.
   - **Pillar 2: Hybrid Search with Reciprocal Rank Fusion (RRF, $k=60$)**: Combines high-speed semantic dense vector retrieval ($w=0.60$) with an in-memory Okapi BM25 sparse index ($w=0.40$) across 1,467 vocabulary terms.
   - **Pillar 3: Two-Stage Cross-Attention Neural Reranker**: Rescores fused candidate chunks using a composite metric of policy code alignment, token window density, section header match, and numeric SLA anchors.
   - **Pillar 4: Pre-generation Retrieval Quality Gating**: Evaluates context grounding sufficiency before a single LLM token is generated, preventing hallucinations and exposing real-time grounding confidence (`🎯 Quality: XX%`).

5. **Zero-Secret BYOK Architecture with Three.js 3D Rabbit Vault**:
   All user API keys (OpenAI, Groq, Anthropic, Google Gemini) are stored client-side in the browser using the WebCrypto API with AES-256-GCM encryption derived from a user master passphrase via PBKDF2 (100,000 iterations). Keys are transmitted ephemerally per-request over HTTPS headers (`X-OpenAI-Key`, etc.) and never touch server disk or environment variables.

6. **Multi-Provider Resilience & Circuit Breaker Isolation**:
   Upstream LLM dependencies are encapsulated within sliding-window circuit breakers. If an API provider experiences elevated latency (>2,000ms) or an error rate exceeding 5%, the circuit trips to `OPEN`, immediately falling back to secondary providers (OpenAI $\to$ Groq $\to$ Deterministic Local Simulator) in under 50ms.

7. **Negative-Response Cache Isolation**:
   Under no circumstances are model refusals, apologies ("I don't have access to..."), or empty responses cached in the semantic cache. Responses undergo strict output guardrail validation before vector persistence.

---

## 2. High-Level Design (HLD)

### 2.1 System Architecture & Data Flow

```mermaid
sequenceDiagram
    autonumber
    actor User as Client (Apple Glass UI / 3D Vault)
    participant Edge as Edge / Rate Limiter (Redis Lua)
    participant Guard as Pre-Inference Guardrails (PII + Injection)
    participant Emb as Local Embedder (ONNX all-MiniLM)
    participant SIMDCache as C++17 AVX2 Semantic Cache
    participant QR as Pillar 1: Query Rewriter
    participant BM25 as Pillar 2a: Okapi BM25 Index
    participant SIMDRAG as Pillar 2b: C++17 AVX2 SIMD
    participant RRF as Pillar 2c: RRF Fusion (k=60)
    participant Rerank as Pillar 3: Neural Reranker
    participant Quality as Pillar 4: Quality Gate
    participant Router as Multi-Provider LLM Router
    participant OutGuard as Output Guard & Cache Validator

    User->>Edge: POST /v1/rag/query (Prompt + BYOK Headers)
    Edge->>Edge: Evaluate Token-Bucket Quota (Redis Lua)
    alt Quota Exceeded
        Edge-->>User: HTTP 429 Too Many Requests
    end
    Edge->>Guard: Forward Raw Prompt
    Guard->>Guard: Scan Prompt Injection & Jailbreaks (<0.15ms)
    alt Injection Detected
        Guard-->>User: HTTP 200 Security Policy Block (Risk: 1.0)
    end
    Guard->>Guard: Luhn Card Checksum & PII Masking (<0.08ms)
    Guard->>Emb: Deliver Sanitized Query Text
    Emb->>Emb: Single-Pass Tokenization & L2-Normalization
    Emb->>SIMDCache: Query Vector vq (384-dim float)
    
    SIMDCache->>SIMDCache: AVX2 SIMD Dot Product (8.4µs)
    alt Cache Hit (Cosine Similarity >= 0.86)
        SIMDCache-->>User: Instant Cached Response (24.2ms p50 | $0.00 Cost)
    else Cache Miss (Cosine Similarity < 0.86)
        SIMDCache->>QR: Cache Miss Trigger -> Raw Query
        QR->>QR: Expand Domain Intent & Acronyms
        
        par Parallel Hybrid Retrieval
            QR->>BM25: Query String -> Okapi BM25 Sparse Search
            Emb->>SIMDRAG: Reuse Same vq -> AVX2 SIMD Dense Search
        end
        
        BM25->>RRF: Top-20 Sparse Candidates (k1=1.5, b=0.75)
        SIMDRAG->>RRF: Top-20 Dense Candidates (Cosine)
        RRF->>Rerank: Fused Candidates (RRF k=60, w=0.6/0.4)
        
        Rerank->>Rerank: Cross-Attention Scoring (Code + Window + Headings)
        Rerank->>Quality: Top-K Reranked High-Precision Chunks
        
        Quality->>Quality: Compute Quality Score (Sim + Cov + Density)
        Quality->>Router: Verified Grounded Chunks + Quality Score
        
        Router->>Router: Evaluate Circuit Breaker State (CLOSED)
        Router->>Router: Route to Client Model (OpenAI / Groq / Anthropic)
        Router-->>OutGuard: Raw LLM Completion
        
        OutGuard->>OutGuard: Check PII Leak & Negative Refusal
        alt Non-Refusal & Grounded
            OutGuard->>SIMDCache: Store Vector vq + Completion
        end
        OutGuard-->>User: Grounded RAG Response + Telemetry (Quality: 78%)
    end
```

### 2.2 Ready-to-Paste Eraser.io Diagram-as-Code

The complete Eraser.io script is maintained in [`docs/eraser_hld.txt`](docs/eraser_hld.txt).

---

## 3. Low-Level Design (LLD): Component Breakdown

### 3.1 Pre-Inference Security & Guardrail Subsystem

```mermaid
graph LR
    subgraph Pre-Inference Guardrail Chain (<0.5ms)
        Input[Raw User Query] --> RL[Redis Token-Bucket Limiter]
        RL -->|HTTP 429| Deny1[Throttled Client]
        RL -->|Allowed| PI[Prompt Injection Guard]
        PI -->|Blocked| Deny2[Security Policy Block]
        PI -->|Safe| PR[PII Redactor + Luhn Checksum]
        PR --> Sanitized[Sanitized Text Output]
    end
```

#### A. Distributed Token-Bucket Rate Limiter (`app/guardrails/rate_limiter.py`)
- **Storage**: Redis cluster with local memory fallback (`InMemoryTokenBucket`).
- **Atomicity**: Executed via an atomic Lua script (`EVALSHA`) to prevent race conditions during distributed concurrency spikes.
- **Tiers**:
  - Starter: $60\text{ requests/min}$ (burst 10)
  - Growth: $600\text{ requests/min}$ (burst 50)
  - Enterprise: $3,000\text{ requests/min}$ (burst 200)

#### B. Prompt Injection & Jailbreak Guard (`app/guardrails/prompt_injection.py`)
- **Inspection Overhead**: $0.134\text{ ms (p95)}$.
- **Detection Mechanics**:
  - Multi-pattern regular expressions detecting delimiter overrides (`<system>`, `[INST]`, `### Instruction:`).
  - DAN (Do-Anything-Now) persona injection detection.
  - System prompt exfiltration keywords ("ignore previous instructions", "repeat the prompt above", "reveal base prompt").
  - Approximate matching via normalized Levenshtein distance on known attack signatures.
- **Verification Score**: $100\%$ interception rate ($20/20$ attacks blocked) with $0\%$ false positive rate ($0/100$ clean requests).

#### C. High-Throughput PII Redactor (`app/guardrails/pii_redactor.py`)
- **Inspection Overhead**: $0.076\text{ ms (p95)}$.
- **ISO/IEC 7812 Luhn Algorithm**: When a 13-19 digit number is matched by the credit card pattern, it is passed through a bit-shift Luhn checksum:
  $$\sum_{i=1}^{n} \left( \text{if } i \equiv 1 \pmod 2 \text{ then } f(2 \cdot d_i) \text{ else } d_i \right) \equiv 0 \pmod{10}$$
  Only numbers with valid Luhn checksums are flagged as credit cards, eliminating false positives on arbitrary invoice IDs or tracking numbers.
- **Redaction Catalog**: Credit Cards (`[CARD_REDACTED]`), US SSN (`[SSN_REDACTED]`), Phone Numbers (`[PHONE_REDACTED]`), Email Addresses (`[EMAIL_REDACTED]`), IPv4 Addresses (`[IP_REDACTED]`), AWS Secret Keys & Bearer Tokens (`[KEY_REDACTED]`).

---

### 3.2 Single-Pass Vectorization Subsystem (`app/embeddings/onnx_embedder.py`)

- **Model**: `sentence-transformers/all-MiniLM-L6-v2` loaded locally via ONNX Runtime / PyTorch.
- **Dimensionality**: $D = 384$ floats per vector.
- **L2-Normalization Invariant**:
  Every generated vector $\mathbf{u}$ is normalized to unit length immediately after mean pooling:
  $$\hat{\mathbf{u}} = \frac{\mathbf{u}}{\|\mathbf{u}\|_2} = \frac{\mathbf{u}}{\sqrt{\sum_{i=1}^{384} u_i^2}}$$
- **Mathematical Implication for SIMD**:
  Because both the stored vectors $\mathbf{v}$ and query vector $\hat{\mathbf{u}}$ satisfy $\|\hat{\mathbf{u}}\|_2 = \|\mathbf{v}\|_2 = 1.0$, cosine similarity simplifies to a pure Euclidean dot product:
  $$\text{CosineSimilarity}(\hat{\mathbf{u}}, \mathbf{v}) = \sum_{i=1}^{384} \hat{u}_i \cdot v_i$$
  This completely removes division, square roots, and norm calculations from the inner SIMD loop.

---

### 3.3 C++17 AVX2 + FMA SIMD Engine (`cpp_engine/simd_engine.cpp`)

The SIMD engine is compiled into a native CPython extension via `pybind11` using `-O3 -mavx2 -mfma`.

#### A. Hardware Register Mapping & Loop Unrolling
The dot product loop is unrolled 4-wide across 256-bit `YMM` registers, processing **32 floating-point values per iteration** (12 iterations total for 384 dimensions):

```cpp
__m256 sum0 = _mm256_setzero_ps();
__m256 sum1 = _mm256_setzero_ps();
__m256 sum2 = _mm256_setzero_ps();
__m256 sum3 = _mm256_setzero_ps();

for (size_t i = 0; i < 384; i += 32) {
    __m256 q0 = _mm256_loadu_ps(query + i + 0);
    __m256 v0 = _mm256_loadu_ps(target + i + 0);
    sum0 = _mm256_fmadd_ps(q0, v0, sum0);

    __m256 q1 = _mm256_loadu_ps(query + i + 8);
    __m256 v1 = _mm256_loadu_ps(target + i + 8);
    sum1 = _mm256_fmadd_ps(q1, v1, sum1);

    __m256 q2 = _mm256_loadu_ps(query + i + 16);
    __m256 v2 = _mm256_loadu_ps(target + i + 16);
    sum2 = _mm256_fmadd_ps(q2, v2, sum2);

    __m256 q3 = _mm256_loadu_ps(query + i + 24);
    __m256 v3 = _mm256_loadu_ps(target + i + 24);
    sum3 = _mm256_fmadd_ps(q3, v3, sum3);
}

__m256 total = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
```

#### B. Top-K Selection Heap
Top-K ranking uses a fixed-capacity min-heap (`std::priority_queue`) bounded to $O(K)$ space and $O(N \log K)$ time complexity across $N$ candidate chunks.

---

### 3.4 Semantic Cache Subsystem (`app/gateway/semantic_cache.py`)

- **Lookup Threshold**: $\text{Cosine Similarity} \ge 0.86$.
- **Lookup Latency**: **8.40 µs (p50)**.
- **Storage Structure**: Continuous contiguous float buffer in C++ with an in-memory LRU response map (bounded to 10,000 entries, $TTL = 3600\text{s}$).
- **Negative Caching Guard**:
  ```python
  def is_negative_or_refusal_response(text: str) -> bool:
      lower = text.lower()
      return any(phrase in lower for phrase in [
          "i am sorry", "i'm sorry", "i do not have access",
          "i don't have access", "does not contain", "cannot answer"
      ])
  ```
  Model refusals and apologetic fallbacks are strictly barred from cache insertion.

---

### 3.5 The 4-Pillar Production RAG Engine

```mermaid
graph TD
    Query[User Query Text] --> P1[Pillar 1: Pre-Retrieval Query Rewriter]
    P1 -->|Domain Expanded Text| P2a[Pillar 2a: Okapi BM25 Index]
    Vector[Reused vq Vector] --> P2b[Pillar 2b: C++17 AVX2 SIMD Dense]
    
    P2a -->|Top Sparse Candidates| P2c[Pillar 2c: Reciprocal Rank Fusion RRF k=60]
    P2b -->|Top Dense Candidates| P2c
    
    P2c -->|Fused Candidate Chunks| P3[Pillar 3: Cross-Attention Neural Reranker]
    P3 -->|Top-K High Precision Chunks| P4[Pillar 4: Retrieval Quality Gating]
    
    P4 -->|Sufficient: Score >= 0.35| LLM[LLM Context Augmentation]
    P4 -->|Insufficient: Score < 0.35| Fallback[Safe Domain Clarification]
```

#### 3.5.1 Pillar 1: Pre-Retrieval Query Rewriting & Expansion (`app/rag/query_rewriter.py`)
- **Intent**: Resolves ambiguous phrasing ("what is company policy?"), extracts domain entities, and expands acronyms before retrieval begins.
- **Acronym Expansion Dictionary**:
  `SLA` $\to$ *Service Level Agreement*, `DLQ` $\to$ *Dead Letter Queue*, `VDMP` $\to$ *Visa Dispute Monitoring Program*, `KYC/KYB` $\to$ *Know Your Customer / Business*, `RPO/RTO` $\to$ *Recovery Point / Time Objective*, `PCI-DSS` $\to$ *Payment Card Industry Data Security Standard*.
- **Domain Scope Mapping**:
  Broad queries matching `policy`, `manual`, `rules`, or `guidelines` are automatically augmented with references to all 10 NovaPay policy codes (`FIN-101`, `GW-204`, `RISK-309`, `HOOK-401`, `SEC-402`, `FX-603`, `KYC-702`, `INFRA-805`, `SUB-904`, `SLA-1001`).

#### 3.5.2 Pillar 2: Hybrid Search with Reciprocal Rank Fusion (`app/rag/bm25_retriever.py` & `retriever.py`)
Combines sparse keyword search with dense vector search to eliminate vocabulary mismatch and guarantee exact code retrieval.

- **Okapi BM25 Sparse Index**:
  Given a query $Q$ with terms $q_1, \dots, q_n$ and document $D$:
  $$\text{BM25}(D, Q) = \sum_{i=1}^n \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
  Where $k_1 = 1.5$, $b = 0.75$, and Robertson-Spärck Jones IDF is defined as:
  $$\text{IDF}(q_i) = \ln \left( \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1 \right)$$
  Title tokens are weighted at $2.0\times$ to prioritize primary section headers.
- **Reciprocal Rank Fusion (RRF, $k=60$)**:
  Fuses dense ranks and sparse ranks without score calibration artifacts:
  $$\text{RRF\_Score}(d) = w_{\text{dense}} \cdot \frac{1}{k + \text{rank}_{\text{dense}}(d)} + w_{\text{bm25}} \cdot \frac{1}{k + \text{rank}_{\text{bm25}}(d)}$$
  With $w_{\text{dense}} = 0.60$, $w_{\text{bm25}} = 0.40$, and smoothing constant $k = 60$.

#### 3.5.3 Pillar 3: Cross-Attention Neural Reranker (`app/rag/reranker.py`)
Distills the fused candidate pool down to the highest-precision Top-K chunks ($K=5$) via a multi-factor scoring function:
$$S_{\text{rerank}} = 0.35 \cdot S_{\text{code}} + 0.35 \cdot S_{\text{window}} + 0.20 \cdot S_{\text{header}} + 0.10 \cdot S_{\text{num}}$$

- $S_{\text{code}}$: Exact policy reference alignment (e.g. `FIN-101`, `SEC-402`, `DOC-OPS-001`).
- $S_{\text{window}}$: Token proximity and exact keyword density within candidate text.
- $S_{\text{header}}$: Overlap between query terms and chunk section titles.
- $S_{\text{num}}$: Numeric anchor matching (SLA days, percentages, currency values, timeouts).

#### 3.5.4 Pillar 4: Pre-Generation Retrieval Quality Gating (`app/rag/quality_evaluator.py`)
Measures context sufficiency before invocation of upstream LLMs:
$$Q = 0.40 \cdot \bar{S}_{\text{sim}} + 0.30 \cdot C_{\text{cov}} + 0.30 \cdot D_{\text{fact}}$$

- $\bar{S}_{\text{sim}}$: Mean vector similarity of the Top-3 reranked chunks.
- $C_{\text{cov}}$: Query keyword coverage in retrieved context.
- $D_{\text{fact}}$: Factual density (ratio of technical numbers, codes, and uppercase anchors).
- **Sufficiency Gate**: If $Q < 0.35$, the query is classified as ungrounded and routed to safe domain clarification instead of risking hallucination.
- **Telemetry Exposure**: The composite score is emitted in all API responses (`retrieval_quality_score: 0.78`) and displayed in the UI as a real-time badge (`🎯 Quality: 78%`).

---

### 3.6 Resilient Upstream Router & Circuit Breaker (`app/gateway/llm_router.py`)

- **BYOK Multi-Model Routing**:
  The gateway dynamically respects client-provided headers to route between providers:
  - `openai`: `gpt-4o`, `gpt-4o-mini`
  - `groq`: `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `openai/gpt-oss-120b`
  - `anthropic`: `claude-3-5-sonnet-latest`
  - `gemini`: `gemini-1.5-pro`, `gemini-1.5-flash`
- **Sliding-Window Circuit Breakers**:
  - `failure_threshold`: 5 consecutive failures.
  - `recovery_timeout`: 30.0 seconds.
  - Failover cascade: Active Provider $\to$ Groq Backup $\to$ Deterministic Local Simulator.
- **Token Limits**: Upstream calls are configured with `max_tokens = 1200` to guarantee complete multi-row Markdown tables without mid-sentence truncation.

---

### 3.7 Interactive 3D Rabbit Three.js BYOK Vault (`app/static/index.html`)

- **3D Mascot Physics**:
  Rendered in Three.js WebGL with dynamic ambient lighting, procedural breathing animations, and mouse-follow ear rotation.
- **Client-Side Key Encryption (WebCrypto API)**:
  - Cipher: **AES-256-GCM** with a unique 96-bit Initialization Vector (`IV`) per session.
  - Key Derivation: **PBKDF2** with SHA-256 and 100,000 hash iterations.
  - Local Storage: Only the ciphertext (`ct`) and IV are stored in `localStorage`. Plaintext keys exist only in browser memory and are wiped on session exit.
- **Zero-Secret Cloud Deployment**:
  The repository contains zero hardcoded keys and is ready for 1-click Render deployment via [`render.yaml`](render.yaml).

---

## 4. Performance & Benchmark Verification Matrix

Data from empirical benchmarks over `data/novapay_enterprise_rag_corpus.pdf` ([`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md)):

| Metric Category | Baseline (Zero Cache) | SentinelAI Warm State | Impact / Improvement |
| :--- | :---: | :---: | :---: |
| **LLM API Cost (120 Queries)** | `$0.0139` | `$0.0032` | **76.61% Cost Decrease** |
| **Pipeline Latency (p50)** | `111.21 ms` | `25.98 ms` | **76.64% Latency Reduction** |
| **Direct Cache Hit Latency (p50)** | `N/A` | `24.24 ms` | **Sub-25ms Execution** |
| **Direct Cache Hit Latency (p95)** | `N/A` | `66.57 ms` | **51.66% Reduction vs Baseline** |
| **Semantic Cache Hit Ratio** | `0.0%` | `65.00%` | **65.00% Offload Rate** |
| **C++17 AVX2 SIMD Search (p50)** | `29.40 µs` (NumPy) | **`8.40 µs` (AVX2)** | **3.50x SIMD Hardware Speedup** |
| **C++17 AVX2 SIMD Search (p95)** | `41.80 µs` (NumPy) | **`11.40 µs` (AVX2)** | **3.67x SIMD Hardware Speedup** |
| **Okapi BM25 Index Vocabulary** | — | **1,467 terms** | **Zero-Dependency Sparse Index** |
| **Retrieval Quality (Grounding)**| Unmeasured | **78% - 94%** | **Gated Sufficiency Filtering** |
| **PII Redactor Latency (p95)** | — | **`0.0763 ms`** | **<0.5ms SLA Compliant (True)** |
| **Prompt Injection Detection** | — | **`100.0%` (20/20)** | **0% False Positives (0/100)** |
| **Automated Test Suite** | — | **31 / 31 Passing** | **100% Pass Rate (57.25s)** |

---

## 5. Complete Directory & File Layout

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
│   ├── simd_engine.cpp                     # C++17 AVX2+FMA vector SIMD core
│   └── setup.py                            # pybind11 compilation script (-O3 -mavx2 -mfma)
├── app/
│   ├── main.py                             # FastAPI server & hybrid RAG route handlers
│   ├── config.py                           # Central configuration & pricing matrix
│   ├── static/
│   │   └── index.html                      # Apple visionOS glass chat UI + 3D Three.js Rabbit
│   ├── guardrails/
│   │   ├── rate_limiter.py                 # Redis Lua token-bucket limiter
│   │   ├── prompt_injection.py             # Jailbreak & injection blocker
│   │   ├── pii_redactor.py                 # ISO/IEC 7812 Luhn PII redactor
│   │   └── output_guard.py                 # Whitespace-preserving response scanner
│   ├── embeddings/
│   │   └── onnx_embedder.py                # 384-dim L2 unit vector embedder
│   ├── rag/
│   │   ├── __init__.py                     # RAG module exports
│   │   ├── query_rewriter.py               # Pillar 1: Domain query rewriting & expansion
│   │   ├── bm25_retriever.py               # Pillar 2a: Okapi BM25 sparse keyword retriever
│   │   ├── retriever.py                    # Pillar 2b: C++17 AVX2 SIMD hybrid retriever
│   │   ├── reranker.py                     # Pillar 3: Cross-attention neural reranker
│   │   ├── quality_evaluator.py            # Pillar 4: Pre-generation quality & grounding gate
│   │   └── pdf_chunker.py                  # Section-aware PDF corpus chunker
│   └── gateway/
│       ├── semantic_cache.py               # AVX2 similarity cache & response store
│       ├── circuit_breaker.py              # Sliding-window circuit breaker state machine
│       └── llm_router.py                   # Dynamic BYOK multi-model failover router
├── benchmarks/
│   ├── workload_120.py                     # 120 enterprise query dataset
│   └── run_benchmark.py                    # Benchmark runner & evaluator
├── tests/
│   └── test_production_suite.py            # Full pytest suite (31/31 passing)
├── Dockerfile                              # Multi-stage production container
├── docker-compose.yml                      # Full stack deployment with Redis
└── requirements.txt
```
