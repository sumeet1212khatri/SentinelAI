"""
retriever.py — Production Hybrid C++17 AVX2 SIMD & Okapi BM25 RAG Retriever with RRF & Reranking.
Combines:
  1. Dense vector similarity via C++17 AVX2 SIMD dot-product engine.
  2. Sparse keyword matching via Okapi BM25 engine.
  3. Reciprocal Rank Fusion (RRF, k=60) for balanced multi-modal candidate selection.
  4. Cross-Attention neural reranker for precision context distillation.
"""
import numpy as np
import time
import logging
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field

from ..config import RAG_TOP_K, RAG_SIMILARITY_THRESHOLD, EMBEDDING_DIM
from .pdf_chunker import DocumentChunk
from .bm25_retriever import BM25Retriever, BM25Match
from .reranker import CrossAttentionReranker, RerankedChunk

logger = logging.getLogger(__name__)

# Try importing the compiled SIMD engine; fall back to NumPy
try:
    import simd_engine
    SIMD_AVAILABLE = True
    logger.info("C++17 AVX2 SIMD engine loaded successfully")
except ImportError:
    SIMD_AVAILABLE = False
    logger.warning("SIMD engine not available, using NumPy fallback")


@dataclass
class RetrievalResult:
    """Result of a retrieval query."""
    chunks: List[DocumentChunk]
    scores: List[float]
    query_text: str = ""
    search_latency_ms: float = 0.0
    total_chunks_searched: int = 0
    used_simd: bool = False
    used_hybrid: bool = False
    used_reranker: bool = False
    reranked_details: List[RerankedChunk] = field(default_factory=list)


class SIMDRetriever:
    """
    Production Hybrid Retriever fusing C++17 AVX2 SIMD dense search,
    Okapi BM25 sparse keyword search, RRF fusion, and two-stage cross-attention reranking.
    """

    def __init__(
        self,
        top_k: int = RAG_TOP_K,
        similarity_threshold: float = RAG_SIMILARITY_THRESHOLD,
    ):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        # Chunk store
        self._chunks: List[DocumentChunk] = []
        self._embeddings: Optional[np.ndarray] = None  # shape: (n, dim)

        # Hybrid components
        self._bm25 = BM25Retriever()
        self._reranker = CrossAttentionReranker()

        self._total_queries = 0
        self._total_latency_ms = 0.0

    def ingest(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """
        Ingest pre-embedded chunks into both dense AVX2 store and sparse BM25 index.
        Zero-double-embedding principle: each chunk embedded exactly once at ingestion.
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Chunks count ({len(chunks)}) != embeddings count ({embeddings.shape[0]})"
            )

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        embeddings = embeddings / norms

        self._chunks = list(chunks)
        self._embeddings = embeddings

        # Build BM25 sparse keyword index over the chunks
        self._bm25.build_index(chunks)

        logger.info(
            f"Successfully ingested {len(chunks)} chunks into C++17 AVX2 vector store "
            f"and BM25 keyword index."
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: Optional[int] = None,
        query_text: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Search for the top-k most relevant chunks.
        If query_text is provided, executes full Hybrid Search (BM25 + AVX2 SIMD + RRF + Reranking).
        If query_text is None, executes pure AVX2 SIMD dense search for backward compatibility.
        """
        if self._embeddings is None or len(self._chunks) == 0:
            return RetrievalResult(
                chunks=[], scores=[], query_text=query_text or "",
                total_chunks_searched=0, used_simd=False,
            )

        k = top_k or self.top_k
        query_vec = np.ascontiguousarray(query_embedding.flatten(), dtype=np.float32)

        start = time.perf_counter()

        # 1. Dense SIMD / NumPy candidate search
        dense_candidates_count = min(len(self._chunks), max(k * 3, 15))

        if SIMD_AVAILABLE:
            result = simd_engine.top_k_search(
                query_vec,
                self._embeddings,
                dense_candidates_count,
                -1.0,
            )
            dense_indices = list(result.indices)
            dense_scores = list(result.scores)
            used_simd = True
        else:
            similarities = self._embeddings @ query_vec
            top_idx = np.argsort(similarities)[::-1][:dense_candidates_count]
            dense_indices = top_idx.tolist()
            dense_scores = similarities[top_idx].tolist()
            used_simd = False

        dense_scores_map = {idx: float(score) for idx, score in zip(dense_indices, dense_scores)}

        # 2. If query_text is provided, perform Hybrid BM25 Fusion + Cross-Attention Reranking
        if query_text and query_text.strip():
            # Sparse BM25 Search
            tokens = BM25Retriever.tokenize(query_text)
            sparse_matches = self._bm25.search(tokens, top_k=dense_candidates_count)

            # Reciprocal Rank Fusion (RRF) with standard constant 60
            rrf_constant = 60
            rrf_scores: Dict[int, float] = {}

            # Add Dense ranks (weight 0.60)
            for rank, idx in enumerate(dense_indices):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (0.60 / (rrf_constant + rank + 1))

            # Add Sparse BM25 ranks (weight 0.40)
            for rank, bm25_match in enumerate(sparse_matches):
                idx = bm25_match.chunk_index
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (0.40 / (rrf_constant + rank + 1))

            # Sort fused candidates by RRF score
            sorted_candidate_ids = sorted(
                rrf_scores.keys(), key=lambda doc_id: rrf_scores[doc_id], reverse=True
            )[:dense_candidates_count]

            candidate_pairs = [
                (self._chunks[idx], dense_scores_map.get(idx, 0.35))
                for idx in sorted_candidate_ids
            ]

            # 3. Two-Stage Cross-Attention Reranking
            reranked = self._reranker.rerank(query=query_text, candidate_chunks=candidate_pairs, top_k=k)

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._total_queries += 1
            self._total_latency_ms += elapsed_ms

            result_chunks = [r.chunk for r in reranked]
            result_scores = [r.rerank_score for r in reranked]

            return RetrievalResult(
                chunks=result_chunks,
                scores=result_scores,
                query_text=query_text,
                search_latency_ms=elapsed_ms,
                total_chunks_searched=len(self._chunks),
                used_simd=used_simd,
                used_hybrid=True,
                used_reranker=True,
                reranked_details=reranked,
            )

        # Fallback: Pure Dense Vector Search (e.g. for vector-only tests)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._total_queries += 1
        self._total_latency_ms += elapsed_ms

        result_chunks = [self._chunks[i] for i in dense_indices[:k]]
        result_scores = [float(s) for s in dense_scores[:k]]

        return RetrievalResult(
            chunks=result_chunks,
            scores=result_scores,
            query_text="",
            search_latency_ms=elapsed_ms,
            total_chunks_searched=len(self._chunks),
            used_simd=used_simd,
            used_hybrid=False,
            used_reranker=False,
        )

    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """Explicit entrypoint for full Hybrid (BM25 + AVX2 + RRF + Reranker) search."""
        return self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            query_text=query_text,
        )

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> RetrievalResult:
        """Legacy helper for backward compatibility."""
        return self.search(query_embedding=query_embedding, top_k=top_k)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def stats(self) -> dict:
        avg_latency = self._total_latency_ms / max(self._total_queries, 1)
        return {
            "total_queries": self._total_queries,
            "avg_search_latency_ms": round(avg_latency, 4),
            "total_chunks": len(self._chunks),
            "simd_available": SIMD_AVAILABLE,
            "hybrid_bm25_ready": True,
            "reranker_ready": True,
        }
