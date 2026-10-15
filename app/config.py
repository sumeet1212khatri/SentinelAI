"""
config.py — Central configuration for SentinelAI-RAG.
All environment variables, thresholds, pricing, and SLA timeouts.
"""
import os
from pathlib import Path

# ─────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_PATH = DATA_DIR / "novapay_enterprise_rag_corpus.pdf"

# ─────────────────────────────────────────────────
# Embedding Model
# ─────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# ─────────────────────────────────────────────────
# RAG Pipeline
# ─────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.25"))

# ─────────────────────────────────────────────────
# Semantic Cache
# ─────────────────────────────────────────────────
CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.75"))
CACHE_MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "10000"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))

# ─────────────────────────────────────────────────
# LLM Provider Configuration
# ─────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

# Deterministic Local Simulator (always-available fallback)
USE_LOCAL_SIMULATOR = os.getenv("USE_LOCAL_SIMULATOR", "true").lower() == "true"

# ─────────────────────────────────────────────────
# LLM Pricing (per 1M tokens, USD)
# ─────────────────────────────────────────────────
LLM_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
    "openai/gpt-oss-120b": {"input": 0.50, "output": 0.80},
    "deterministic-rag-synth-v1": {"input": 0.0, "output": 0.0},
    "local-simulator": {"input": 0.0, "output": 0.0},
}

# Average tokens per query/response (for cost estimation)
AVG_INPUT_TOKENS = int(os.getenv("AVG_INPUT_TOKENS", "800"))
AVG_OUTPUT_TOKENS = int(os.getenv("AVG_OUTPUT_TOKENS", "300"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1200"))

# ─────────────────────────────────────────────────
# Guardrails
# ─────────────────────────────────────────────────
PII_REDACTION_ENABLED = os.getenv("PII_REDACTION_ENABLED", "true").lower() == "true"
PROMPT_INJECTION_ENABLED = os.getenv("PROMPT_INJECTION_ENABLED", "true").lower() == "true"
OUTPUT_GUARD_ENABLED = os.getenv("OUTPUT_GUARD_ENABLED", "true").lower() == "true"

# ─────────────────────────────────────────────────
# Rate Limiting
# ─────────────────────────────────────────────────
RATE_LIMIT_REQUESTS_PER_MINUTE = int(os.getenv("RATE_LIMIT_RPM", "60"))
RATE_LIMIT_TOKENS_PER_MINUTE = int(os.getenv("RATE_LIMIT_TPM", "100000"))
RATE_LIMIT_BURST = int(os.getenv("RATE_LIMIT_BURST", "10"))

# ─────────────────────────────────────────────────
# Circuit Breaker
# ─────────────────────────────────────────────────
CB_FAILURE_THRESHOLD = int(os.getenv("CB_FAILURE_THRESHOLD", "5"))
CB_RECOVERY_TIMEOUT = float(os.getenv("CB_RECOVERY_TIMEOUT", "30.0"))
CB_HALF_OPEN_MAX_CALLS = int(os.getenv("CB_HALF_OPEN_MAX_CALLS", "3"))
CB_WINDOW_SIZE = int(os.getenv("CB_WINDOW_SIZE", "60"))

# ─────────────────────────────────────────────────
# Redis
# ─────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_ENABLED = os.getenv("REDIS_ENABLED", "false").lower() == "true"

# ─────────────────────────────────────────────────
# SLA & Timeouts
# ─────────────────────────────────────────────────
SLA_P99_LATENCY_MS = float(os.getenv("SLA_P99_LATENCY_MS", "2000.0"))
LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "30.0"))
HEALTH_CHECK_INTERVAL = float(os.getenv("HEALTH_CHECK_INTERVAL", "10.0"))

# ─────────────────────────────────────────────────
# Server
# ─────────────────────────────────────────────────
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
SERVER_WORKERS = int(os.getenv("SERVER_WORKERS", "1"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
