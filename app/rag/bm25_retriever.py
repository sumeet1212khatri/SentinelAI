"""
bm25_retriever.py — High-Performance In-Memory Okapi BM25 Sparse Search Engine.
Provides sub-millisecond keyword-exact retrieval for policy codes, thresholds,
fees, headers, and acronyms to complement dense semantic vector search.
"""
import math
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

from .pdf_chunker import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class BM25Match:
    chunk_index: int
    score: float
    chunk: DocumentChunk


class BM25Retriever:
    """
    Okapi BM25 Sparse In-Memory Keyword Search Engine.
    Tuned with k1=1.5 and b=0.75. Gives 2.0x weight to terms matching chunk section headers.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        self._chunks: List[DocumentChunk] = []
        self._doc_lengths: List[int] = []
        self._avg_doc_length: float = 0.0
        self._total_docs: int = 0

        # Inverted index: term -> {doc_id: term_frequency}
        self._inverted_index: Dict[str, Dict[int, float]] = {}
        # Document frequency: term -> count of docs containing term
        self._doc_freqs: Dict[str, int] = {}
        # Precomputed IDF table
        self._idf_cache: Dict[str, float] = {}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into normalized lowercase alphanumeric words and symbols."""
        if not text:
            return []
        # Support hyphenated policy codes and percentages e.g. FIN-101, 0.90%, HMAC-SHA256
        raw_tokens = re.findall(r"[A-Za-z0-9]+(?:[\-\.][A-Za-z0-9]+)*", text.lower())
        return [t for t in raw_tokens if len(t) > 1]

    def build_index(self, chunks: List[DocumentChunk]):
        """
        Build inverted index over document chunks.
        Applies a 2x boost to tokens found in section_title.
        """
        self._chunks = list(chunks)
        self._total_docs = len(chunks)
        self._doc_lengths = []
        self._inverted_index.clear()
        self._doc_freqs.clear()
        self._idf_cache.clear()

        if self._total_docs == 0:
            return

        total_length = 0

        for idx, chunk in enumerate(chunks):
            body_tokens = self.tokenize(chunk.text)
            title_tokens = self.tokenize(chunk.section_title or "")

            term_counts: Dict[str, float] = {}
            for t in body_tokens:
                term_counts[t] = term_counts.get(t, 0.0) + 1.0

            # Boost section title tokens by 2x
            for t in title_tokens:
                term_counts[t] = term_counts.get(t, 0.0) + 2.0

            doc_len = len(body_tokens) + len(title_tokens) * 2
            self._doc_lengths.append(doc_len)
            total_length += doc_len

            for term, count in term_counts.items():
                if term not in self._inverted_index:
                    self._inverted_index[term] = {}
                    self._doc_freqs[term] = 0
                self._inverted_index[term][idx] = count
                self._doc_freqs[term] += 1

        self._avg_doc_length = total_length / max(self._total_docs, 1)

        # Precompute Robertson-Spärck Jones IDF for all terms
        for term, df in self._doc_freqs.items():
            self._idf_cache[term] = math.log(
                1.0 + (self._total_docs - df + 0.5) / (df + 0.5)
            )

        logger.info(
            f"Built BM25 index over {self._total_docs} chunks "
            f"({len(self._inverted_index)} distinct terms, avg_dl={self._avg_doc_length:.1f})"
        )

    def search(self, query_tokens: List[str], top_k: int = 15) -> List[BM25Match]:
        """
        Execute Okapi BM25 score calculation for candidate documents.
        
        Args:
            query_tokens: List of normalized query terms.
            top_k: Number of highest scoring chunks to return.
            
        Returns:
            List of BM25Match sorted by score descending.
        """
        if self._total_docs == 0 or not query_tokens:
            return []

        scores: Dict[int, float] = {}

        for token in query_tokens:
            token = token.lower()
            if token not in self._inverted_index:
                continue

            idf = self._idf_cache.get(token, 0.0)
            doc_postings = self._inverted_index[token]

            for doc_id, tf in doc_postings.items():
                dl = self._doc_lengths[doc_id]
                # Okapi BM25 formula
                num = tf * (self.k1 + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self._avg_doc_length))
                score = idf * (num / denom)
                scores[doc_id] = scores.get(doc_id, 0.0) + score

        if not scores:
            return []

        # Sort documents by BM25 score descending
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            BM25Match(
                chunk_index=doc_id,
                score=float(score),
                chunk=self._chunks[doc_id],
            )
            for doc_id, score in sorted_docs
        ]
