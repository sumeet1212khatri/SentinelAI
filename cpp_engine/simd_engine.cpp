// simd_engine.cpp — C++17 AVX2 + FMA Vector Engine
// Provides: cosine_similarity, top_k_search, cache_lookup
// Compiled with: -O3 -mavx2 -mfma -std=c++17

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <tuple>
#include <numeric>
#include <mutex>
#include <unordered_map>

#ifdef __AVX2__
#include <immintrin.h>
#define SIMD_AVAILABLE true
#else
#define SIMD_AVAILABLE false
#endif

namespace py = pybind11;

// ─────────────────────────────────────────────────
// AVX2 + FMA Dot Product (384-dim optimized)
// ─────────────────────────────────────────────────
static inline float avx2_dot_product(const float* a, const float* b, int n) {
#ifdef __AVX2__
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    // Process 32 floats per iteration (4 x 8-wide AVX2 registers)
    for (; i + 31 < n; i += 32) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va0, vb0, sum);

        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        sum = _mm256_fmadd_ps(va1, vb1, sum);

        __m256 va2 = _mm256_loadu_ps(a + i + 16);
        __m256 vb2 = _mm256_loadu_ps(b + i + 16);
        sum = _mm256_fmadd_ps(va2, vb2, sum);

        __m256 va3 = _mm256_loadu_ps(a + i + 24);
        __m256 vb3 = _mm256_loadu_ps(b + i + 24);
        sum = _mm256_fmadd_ps(va3, vb3, sum);
    }
    // Process remaining 8-wide chunks
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    // Horizontal reduction
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float result = _mm_cvtss_f32(s);
    // Scalar tail
    for (; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
#else
    // Scalar fallback
    float result = 0.0f;
    for (int i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
#endif
}

// ─────────────────────────────────────────────────
// L2 Norm (AVX2 accelerated)
// ─────────────────────────────────────────────────
static inline float avx2_l2_norm(const float* a, int n) {
    return std::sqrt(avx2_dot_product(a, a, n));
}

// ─────────────────────────────────────────────────
// Cosine Similarity (for L2-normalized vectors, this is just dot product)
// ─────────────────────────────────────────────────
float cosine_similarity(py::array_t<float> a, py::array_t<float> b) {
    auto buf_a = a.request();
    auto buf_b = b.request();
    if (buf_a.ndim != 1 || buf_b.ndim != 1)
        throw std::runtime_error("Input arrays must be 1-dimensional");
    if (buf_a.shape[0] != buf_b.shape[0])
        throw std::runtime_error("Input arrays must have the same length");

    int n = static_cast<int>(buf_a.shape[0]);
    const float* pa = static_cast<const float*>(buf_a.ptr);
    const float* pb = static_cast<const float*>(buf_b.ptr);

    float dot = avx2_dot_product(pa, pb, n);
    float norm_a = avx2_l2_norm(pa, n);
    float norm_b = avx2_l2_norm(pb, n);

    if (norm_a < 1e-9f || norm_b < 1e-9f) return 0.0f;
    return dot / (norm_a * norm_b);
}

// ─────────────────────────────────────────────────
// Top-K Search: returns indices and scores of top-k most similar vectors
// ─────────────────────────────────────────────────
struct SearchResult {
    std::vector<int> indices;
    std::vector<float> scores;
};

SearchResult top_k_search(
    py::array_t<float> query,        // shape: (dim,)
    py::array_t<float> matrix,       // shape: (num_vectors, dim)
    int k,
    float threshold = -1.0f          // optional minimum similarity threshold
) {
    auto q_buf = query.request();
    auto m_buf = matrix.request();

    if (q_buf.ndim != 1)
        throw std::runtime_error("Query must be 1-dimensional");
    if (m_buf.ndim != 2)
        throw std::runtime_error("Matrix must be 2-dimensional");
    if (q_buf.shape[0] != m_buf.shape[1])
        throw std::runtime_error("Query dimension must match matrix column dimension");

    int dim = static_cast<int>(q_buf.shape[0]);
    int num_vectors = static_cast<int>(m_buf.shape[0]);
    const float* q_ptr = static_cast<const float*>(q_buf.ptr);
    const float* m_ptr = static_cast<const float*>(m_buf.ptr);

    if (k <= 0) k = 1;
    if (k > num_vectors) k = num_vectors;

    // Compute query norm once
    float q_norm = avx2_l2_norm(q_ptr, dim);
    if (q_norm < 1e-9f) {
        return {{}, {}};
    }

    // Compute all similarities
    std::vector<std::pair<float, int>> scores_idx;
    scores_idx.reserve(num_vectors);

    for (int i = 0; i < num_vectors; i++) {
        const float* vec_ptr = m_ptr + (static_cast<size_t>(i) * dim);
        float dot = avx2_dot_product(q_ptr, vec_ptr, dim);
        float v_norm = avx2_l2_norm(vec_ptr, dim);
        float sim = (v_norm < 1e-9f) ? 0.0f : dot / (q_norm * v_norm);

        if (threshold < 0.0f || sim >= threshold) {
            scores_idx.emplace_back(sim, i);
        }
    }

    // Partial sort for top-k
    if (static_cast<int>(scores_idx.size()) > k) {
        std::partial_sort(
            scores_idx.begin(),
            scores_idx.begin() + k,
            scores_idx.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; }
        );
        scores_idx.resize(k);
    } else {
        std::sort(
            scores_idx.begin(),
            scores_idx.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; }
        );
    }

    SearchResult result;
    result.indices.reserve(scores_idx.size());
    result.scores.reserve(scores_idx.size());
    for (const auto& [score, idx] : scores_idx) {
        result.indices.push_back(idx);
        result.scores.push_back(score);
    }
    return result;
}

// ─────────────────────────────────────────────────
// Cache Lookup: find best match above threshold, return (index, score, hit)
// ─────────────────────────────────────────────────
struct CacheLookupResult {
    int best_index;
    float best_score;
    bool hit;
};

CacheLookupResult cache_lookup(
    py::array_t<float> query,
    py::array_t<float> cache_matrix,
    float similarity_threshold
) {
    auto q_buf = query.request();
    auto c_buf = cache_matrix.request();

    if (q_buf.ndim != 1)
        throw std::runtime_error("Query must be 1-dimensional");

    // Empty cache
    if (c_buf.ndim == 1 && c_buf.shape[0] == 0) {
        return {-1, 0.0f, false};
    }
    if (c_buf.ndim != 2)
        throw std::runtime_error("Cache matrix must be 2-dimensional or empty 1-d");
    if (q_buf.shape[0] != c_buf.shape[1])
        throw std::runtime_error("Query dim must match cache matrix column dim");

    int dim = static_cast<int>(q_buf.shape[0]);
    int num_entries = static_cast<int>(c_buf.shape[0]);
    const float* q_ptr = static_cast<const float*>(q_buf.ptr);
    const float* c_ptr = static_cast<const float*>(c_buf.ptr);

    float q_norm = avx2_l2_norm(q_ptr, dim);
    if (q_norm < 1e-9f) return {-1, 0.0f, false};

    float best_score = -2.0f;
    int best_idx = -1;

    for (int i = 0; i < num_entries; i++) {
        const float* v_ptr = c_ptr + (static_cast<size_t>(i) * dim);
        float dot = avx2_dot_product(q_ptr, v_ptr, dim);
        float v_norm = avx2_l2_norm(v_ptr, dim);
        float sim = (v_norm < 1e-9f) ? 0.0f : dot / (q_norm * v_norm);
        if (sim > best_score) {
            best_score = sim;
            best_idx = i;
        }
    }

    bool hit = (best_score >= similarity_threshold);
    return {best_idx, best_score, hit};
}

// ─────────────────────────────────────────────────
// Batch Cosine Similarities: query vs matrix
// ─────────────────────────────────────────────────
py::array_t<float> batch_cosine_similarities(
    py::array_t<float> query,
    py::array_t<float> matrix
) {
    auto q_buf = query.request();
    auto m_buf = matrix.request();

    if (q_buf.ndim != 1) throw std::runtime_error("Query must be 1-d");
    if (m_buf.ndim != 2) throw std::runtime_error("Matrix must be 2-d");
    if (q_buf.shape[0] != m_buf.shape[1])
        throw std::runtime_error("Dimension mismatch");

    int dim = static_cast<int>(q_buf.shape[0]);
    int n = static_cast<int>(m_buf.shape[0]);
    const float* q_ptr = static_cast<const float*>(q_buf.ptr);
    const float* m_ptr = static_cast<const float*>(m_buf.ptr);

    float q_norm = avx2_l2_norm(q_ptr, dim);

    auto result = py::array_t<float>(n);
    auto r_buf = result.request();
    float* r_ptr = static_cast<float*>(r_buf.ptr);

    for (int i = 0; i < n; i++) {
        const float* v_ptr = m_ptr + (static_cast<size_t>(i) * dim);
        float dot = avx2_dot_product(q_ptr, v_ptr, dim);
        float v_norm = avx2_l2_norm(v_ptr, dim);
        r_ptr[i] = (q_norm < 1e-9f || v_norm < 1e-9f) ? 0.0f : dot / (q_norm * v_norm);
    }
    return result;
}

// ─────────────────────────────────────────────────
// Benchmarking utilities
// ─────────────────────────────────────────────────
double benchmark_search_latency_us(
    py::array_t<float> query,
    py::array_t<float> matrix,
    int k,
    int iterations = 1000
) {
    std::vector<double> latencies;
    latencies.reserve(iterations);

    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        top_k_search(query, matrix, k);
        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start).count();
        latencies.push_back(us);
    }

    std::sort(latencies.begin(), latencies.end());
    // Return p50
    return latencies[iterations / 2];
}

struct LatencyStats {
    double p50;
    double p95;
    double p99;
    double mean;
    double min_val;
    double max_val;
};

LatencyStats benchmark_search_latency_full(
    py::array_t<float> query,
    py::array_t<float> matrix,
    int k,
    int iterations = 1000
) {
    std::vector<double> latencies;
    latencies.reserve(iterations);

    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        top_k_search(query, matrix, k);
        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start).count();
        latencies.push_back(us);
    }

    std::sort(latencies.begin(), latencies.end());

    int n = static_cast<int>(latencies.size());
    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);

    return {
        latencies[n / 2],                     // p50
        latencies[static_cast<int>(n * 0.95)], // p95
        latencies[static_cast<int>(n * 0.99)], // p99
        sum / n,                                // mean
        latencies[0],                           // min
        latencies[n - 1]                        // max
    };
}

// ─────────────────────────────────────────────────
// Module: is_simd_available helper
// ─────────────────────────────────────────────────
bool is_simd_available() {
    return SIMD_AVAILABLE;
}

// ─────────────────────────────────────────────────
// Pybind11 Module Definition
// ─────────────────────────────────────────────────
PYBIND11_MODULE(simd_engine, m) {
    m.doc() = "C++17 AVX2 + FMA SIMD Vector Search Engine for SentinelAI-RAG";

    m.def("cosine_similarity", &cosine_similarity,
          py::arg("a"), py::arg("b"),
          "Compute cosine similarity between two vectors using AVX2+FMA");

    py::class_<SearchResult>(m, "SearchResult")
        .def_readonly("indices", &SearchResult::indices)
        .def_readonly("scores", &SearchResult::scores);

    m.def("top_k_search", &top_k_search,
          py::arg("query"), py::arg("matrix"), py::arg("k"),
          py::arg("threshold") = -1.0f,
          "AVX2-accelerated top-k cosine similarity search");

    py::class_<CacheLookupResult>(m, "CacheLookupResult")
        .def_readonly("best_index", &CacheLookupResult::best_index)
        .def_readonly("best_score", &CacheLookupResult::best_score)
        .def_readonly("hit", &CacheLookupResult::hit);

    m.def("cache_lookup", &cache_lookup,
          py::arg("query"), py::arg("cache_matrix"), py::arg("similarity_threshold"),
          "AVX2-accelerated cache lookup with similarity threshold");

    m.def("batch_cosine_similarities", &batch_cosine_similarities,
          py::arg("query"), py::arg("matrix"),
          "Compute cosine similarity of query against all rows in matrix");

    m.def("benchmark_search_latency_us", &benchmark_search_latency_us,
          py::arg("query"), py::arg("matrix"), py::arg("k"),
          py::arg("iterations") = 1000,
          "Benchmark top-k search latency, returns p50 in microseconds");

    py::class_<LatencyStats>(m, "LatencyStats")
        .def_readonly("p50", &LatencyStats::p50)
        .def_readonly("p95", &LatencyStats::p95)
        .def_readonly("p99", &LatencyStats::p99)
        .def_readonly("mean", &LatencyStats::mean)
        .def_readonly("min_val", &LatencyStats::min_val)
        .def_readonly("max_val", &LatencyStats::max_val);

    m.def("benchmark_search_latency_full", &benchmark_search_latency_full,
          py::arg("query"), py::arg("matrix"), py::arg("k"),
          py::arg("iterations") = 1000,
          "Full latency benchmark returning p50/p95/p99/mean/min/max in µs");

    m.def("is_simd_available", &is_simd_available,
          "Check if AVX2 SIMD instructions are available at compile time");
}
