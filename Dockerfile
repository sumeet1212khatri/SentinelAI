# Multi-stage production Dockerfile for SentinelAI-RAG
# Compiles C++17 AVX2+FMA SIMD extension, pre-downloads all-MiniLM-L6-v2, and serves via FastAPI + Uvicorn

# Stage 1: Build C++17 SIMD extension & Python dependencies
FROM python:3.12-slim AS builder

WORKDIR /build

# Install C++17 build tools & SIMD support
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install slim CPU-only PyTorch first (avoids pulling 2GB CUDA binaries)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python build dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Compile C++17 AVX2 SIMD engine
COPY cpp_engine/ ./cpp_engine/
WORKDIR /build/cpp_engine
RUN python setup.py build_ext --inplace

# Stage 2: Production runtime stage
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install runtime system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set persistent HuggingFace cache directory baked into image
ENV HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/cpp_engine \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=8000 \
    USE_LOCAL_SIMULATOR=true

# Pre-download all-MiniLM-L6-v2 model during image build so runtime has 0s download & zero network reliance
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')"

# Copy compiled SIMD extension
COPY --from=builder /build/cpp_engine/simd_engine*.so /app/
COPY --from=builder /build/cpp_engine/simd_engine*.so /app/cpp_engine/

# Copy application source code & data
COPY app/ /app/app/
COPY data/ /app/data/
COPY benchmarks/ /app/benchmarks/
COPY tests/ /app/tests/

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/v1/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
