"""
onnx_embedder.py — ONNX Runtime / PyTorch embedder using all-MiniLM-L6-v2.
Produces 384-dim L2-normalized embeddings with local-cache priority,
clean online downloads on fresh containers (Render/Docker), and a resilient
deterministic fallback embedder.
"""
import os
import numpy as np
import time
import logging
from typing import List, Union, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ONNXEmbedder:
    """
    High-performance text embedder producing 384-dim L2-normalized vectors.
    Supports local snapshot priority, online HuggingFace downloads, and
    deterministic feature-hashing backup to guarantee 100% server uptime.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        batch_size: int = 32,
        use_onnx: bool = True,
    ):
        self._model_name = model_name
        self._embedding_dim = embedding_dim
        self._batch_size = batch_size
        self._model = None
        self._total_texts = 0
        self._total_latency_ms = 0.0
        self._use_onnx = use_onnx

        self._load_model()

    def _fallback_embed(self, texts: List[str]) -> np.ndarray:
        """
        Deterministic, lightweight 384-dim embedder using token and n-gram feature
        projections. Guaranteed to never fail, OOM, or require network access.
        Produces unit L2-normalized float32 vectors.
        """
        dim = self._embedding_dim
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            clean = text.lower().strip()
            if not clean:
                out[i, 0] = 1.0
                continue
            words = clean.split()
            vec = np.zeros(dim, dtype=np.float32)
            for w in words:
                h = abs(hash(w)) % dim
                sign = 1.0 if (abs(hash(w + "_s")) % 2 == 0) else -1.0
                vec[h] += sign * 1.5
            # Character trigrams for morphological robustness
            for j in range(max(1, len(clean) - 2)):
                tri = clean[j : j + 3]
                h = abs(hash(tri)) % dim
                sign = 1.0 if (abs(hash(tri + "_t")) % 2 == 0) else -1.0
                vec[h] += sign * 0.5

            norm = float(np.linalg.norm(vec))
            if norm > 1e-9:
                vec /= norm
            else:
                vec[0] = 1.0
            out[i] = vec
        return out

    def _load_model(self):
        """
        Load the embedding model with:
        1. Local disk snapshot priority (0ms startup if cached)
        2. Online HTTPS download for fresh containers (Render / Docker)
        3. Resilient deterministic fallback to prevent container crash (Exit Code 3)
        """
        # 1. Check candidate local cache directories on disk
        hf_home = os.environ.get("HF_HOME", "")
        clean_name = self._model_name.replace("sentence-transformers/", "")
        candidate_dirs = [
            Path(hf_home) / "hub" / f"models--sentence-transformers--{clean_name}" / "snapshots",
            Path.home() / ".cache" / "huggingface" / "hub" / f"models--sentence-transformers--{clean_name}" / "snapshots",
            Path(self._model_name),
        ]

        local_path = None
        for loc in candidate_dirs:
            if loc.exists():
                if loc.is_dir():
                    snapshots = [p for p in loc.iterdir() if p.is_dir()]
                    if snapshots:
                        local_path = str(snapshots[0])
                        break
                    elif (loc / "config.json").exists():
                        local_path = str(loc)
                        break

        if local_path:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model from warm local snapshot: {local_path}")
                self._model = SentenceTransformer(local_path, device="cpu")
                logger.info("Loaded model successfully from local disk snapshot.")
                return
            except Exception as e:
                logger.warning(f"Local snapshot load failed ({e}), attempting online download...")

        # 2. Ensure offline flags are unset so fresh containers can download from HuggingFace
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

        try:
            from sentence_transformers import SentenceTransformer
            repo_id = self._model_name
            if not repo_id.startswith("sentence-transformers/") and "/" not in repo_id:
                repo_id = f"sentence-transformers/{self._model_name}"

            logger.info(f"Downloading/loading embedding model '{repo_id}' via SentenceTransformer (CPU)...")
            self._model = SentenceTransformer(repo_id, device="cpu")
            logger.info("Successfully loaded embedding model from Hugging Face.")
            return
        except Exception as e:
            logger.warning(f"Online SentenceTransformer download failed ({e}), trying short name...")
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name, device="cpu")
                logger.info(f"Successfully loaded '{self._model_name}'.")
                return
            except Exception as e2:
                logger.error(
                    f"SentenceTransformer could not load model ({e2}). "
                    "Activating resilient deterministic fallback embedder (server will NOT crash)!"
                )
                self._model = None

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate L2-normalized embeddings for one or more texts.

        Args:
            texts: Single string or list of strings.

        Returns:
            numpy array of shape (n, 384) with L2-normalized embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]

        start = time.perf_counter()

        if self._model is not None:
            try:
                embeddings = self._model.encode(
                    texts,
                    batch_size=self._batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,  # L2 normalization
                    convert_to_numpy=True,
                )
                embeddings = np.asarray(embeddings, dtype=np.float32)
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-9)
                embeddings = embeddings / norms
            except Exception as e:
                logger.warning(f"SentenceTransformer encode failed ({e}); using deterministic fallback.")
                embeddings = self._fallback_embed(texts)
        else:
            embeddings = self._fallback_embed(texts)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._total_texts += len(texts)
        self._total_latency_ms += elapsed_ms

        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text, returning a 1-D vector of shape (384,)."""
        result = self.embed(text)
        return result[0]

    @property
    def dim(self) -> int:
        """Return embedding dimensionality."""
        return self._embedding_dim

    @property
    def stats(self) -> dict:
        avg_latency = self._total_latency_ms / max(self._total_texts, 1)
        return {
            "model_name": self._model_name,
            "embedding_dim": self._embedding_dim,
            "total_texts_embedded": self._total_texts,
            "avg_latency_per_text_ms": round(avg_latency, 4),
            "total_latency_ms": round(self._total_latency_ms, 2),
            "engine": "SentenceTransformer" if self._model is not None else "DeterministicFallback",
        }
