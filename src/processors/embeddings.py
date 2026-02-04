"""Embedding providers and utilities for semantic similarity."""

import hashlib
import logging
import struct
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

from config.settings import get_settings

# Lazy numpy import for optional dependency
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


def _check_numpy():
    """Check if numpy is available, raise helpful error if not."""
    if not NUMPY_AVAILABLE:
        raise ImportError(
            "numpy is required for embeddings. Install with: pip install numpy"
        )


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def encode(self, text: str):
        """Encode text to embedding vector."""
        pass

    @abstractmethod
    def encode_batch(self, texts: list[str]):
        """Encode multiple texts to embedding vectors."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class SentenceTransformersProvider(EmbeddingProvider):
    """Local embedding provider using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        _check_numpy()
        self.model_name = model_name
        self._model = None
        self._dimension = None

    def _get_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                # Suppress verbose loading messages from transformers/sentence-transformers
                import io
                import logging as _logging
                import os
                import warnings
                from contextlib import redirect_stderr, redirect_stdout

                # Disable tokenizers parallelism warning
                os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

                # Suppress tqdm progress bars from transformers and huggingface
                os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
                os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
                os.environ.setdefault("TQDM_DISABLE", "1")

                # Suppress verbose loggers
                for logger_name in [
                    "sentence_transformers",
                    "transformers",
                    "huggingface_hub",
                    "filelock",
                    "mlx",
                ]:
                    _logging.getLogger(logger_name).setLevel(_logging.WARNING)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)

                    # Also try to disable transformers' internal progress bars
                    try:
                        import transformers

                        transformers.logging.set_verbosity_error()
                        transformers.logging.disable_progress_bar()
                    except (ImportError, AttributeError):
                        pass

                    from sentence_transformers import SentenceTransformer

                    logger.info(f"Loading embedding model: {self.model_name}")

                    # Suppress stdout/stderr during model loading to hide MLX progress bars
                    # and "LOAD REPORT" messages that bypass the logging system
                    devnull = io.StringIO()
                    with redirect_stdout(devnull), redirect_stderr(devnull):
                        self._model = SentenceTransformer(self.model_name)

                self._dimension = self._model.get_sentence_embedding_dimension()
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
        return self._model

    def encode(self, text: str):
        """Encode text to embedding vector."""
        model = self._get_model()
        return model.encode(
            text, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )

    def encode_batch(self, texts: list[str]):
        """Encode multiple texts to embedding vectors."""
        model = self._get_model()
        return model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        if self._dimension is None:
            self._get_model()
        return self._dimension


class OpenAIProvider(EmbeddingProvider):
    """OpenAI embedding provider using the API."""

    def __init__(self, model_name: str = "text-embedding-3-small"):
        _check_numpy()
        self.model_name = model_name
        self._client = None
        # text-embedding-3-small is 1536 dimensions
        self._dimension = 1536 if "small" in model_name else 3072

    def _get_client(self):
        """Lazy load the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI()
                logger.info(f"OpenAI client initialized with model: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "openai package not installed. "
                    "Run: pip install openai"
                )
        return self._client

    def encode(self, text: str):
        """Encode text to embedding vector."""
        client = self._get_client()
        response = client.embeddings.create(
            input=text,
            model=self.model_name
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def encode_batch(self, texts: list[str]):
        """Encode multiple texts to embedding vectors."""
        client = self._get_client()
        response = client.embeddings.create(
            input=texts,
            model=self.model_name
        )
        embeddings = np.array([d.embedding for d in response.data], dtype=np.float32)
        # Normalize each vector
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        return embeddings / norms

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension


class EmbeddingManager:
    """
    Manages embeddings with caching and database persistence.

    Features:
    - In-memory LRU cache for fast repeated lookups
    - Database persistence for embeddings
    - Lazy provider initialization
    - Cosine similarity computation
    """

    _instance: Optional["EmbeddingManager"] = None

    def __init__(self):
        self.settings = get_settings().embedding
        self._provider: Optional[EmbeddingProvider] = None
        self._cache: dict[str, np.ndarray] = {}

    @classmethod
    def get_instance(cls) -> "EmbeddingManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_provider(self) -> EmbeddingProvider:
        """Get or create embedding provider."""
        _check_numpy()
        if self._provider is None:
            if self.settings.provider == "sentence-transformers":
                self._provider = SentenceTransformersProvider(
                    model_name=self.settings.sentence_transformers_model
                )
            elif self.settings.provider == "openai":
                self._provider = OpenAIProvider(
                    model_name=self.settings.openai_model
                )
            else:
                raise ValueError(f"Unknown embedding provider: {self.settings.provider}")
        return self._provider

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self.settings.dimension > 0:
            return self.settings.dimension
        return self._get_provider().dimension

    def _cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    @lru_cache(maxsize=1024)
    def get_embedding(self, text: str):
        """
        Get embedding for text with caching.

        Args:
            text: Text to encode

        Returns:
            Normalized embedding vector (numpy array)
        """
        _check_numpy()
        if not self.settings.enabled:
            return np.zeros(self.dimension, dtype=np.float32)

        # Truncate very long text
        if len(text) > 2000:
            text = text[:2000]

        provider = self._get_provider()
        return provider.encode(text)

    def get_embeddings_batch(self, texts: list[str]):
        """
        Get embeddings for multiple texts.

        Args:
            texts: List of texts to encode

        Returns:
            Array of embedding vectors (len(texts) x dimension)
        """
        _check_numpy()
        if not self.settings.enabled:
            return np.zeros((len(texts), self.dimension), dtype=np.float32)

        # Truncate long texts
        texts = [t[:2000] if len(t) > 2000 else t for t in texts]

        provider = self._get_provider()
        return provider.encode_batch(texts)

    def cosine_similarity(self, vec1, vec2) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector (should be normalized)
            vec2: Second vector (should be normalized)

        Returns:
            Cosine similarity (-1 to 1, higher is more similar)
        """
        _check_numpy()
        # Vectors are already normalized, so dot product = cosine similarity
        return float(np.dot(vec1, vec2))

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0 to 1)
        """
        if not self.settings.enabled:
            return 0.0

        vec1 = self.get_embedding(text1)
        vec2 = self.get_embedding(text2)

        # Cosine similarity, mapped from [-1, 1] to [0, 1]
        similarity = self.cosine_similarity(vec1, vec2)
        return max(0.0, (similarity + 1) / 2)

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        self.get_embedding.cache_clear()


def embedding_to_bytes(embedding) -> bytes:
    """Convert numpy embedding to bytes for database storage."""
    _check_numpy()
    return embedding.astype(np.float32).tobytes()


def bytes_to_embedding(data: bytes, dimension: int):
    """Convert bytes back to numpy embedding."""
    _check_numpy()
    return np.frombuffer(data, dtype=np.float32).reshape(dimension)


def compute_centroid(embeddings: list):
    """
    Compute centroid of multiple embeddings.

    Args:
        embeddings: List of embedding vectors

    Returns:
        Normalized centroid vector
    """
    _check_numpy()
    if not embeddings:
        raise ValueError("Cannot compute centroid of empty list")

    stacked = np.vstack(embeddings)
    centroid = np.mean(stacked, axis=0)

    # Normalize
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm

    return centroid


def compute_hybrid_similarity(
    rule_score: float,
    semantic_score: float,
    rule_weight: float = 0.4,
    semantic_weight: float = 0.6,
    embeddings_available: bool = True,
) -> float:
    """
    Compute hybrid similarity combining rule-based and semantic scores.

    When embeddings are not available (semantic_score is 0 and embeddings_available is False),
    the function uses 100% rule-based scoring to avoid penalizing matches due to missing embeddings.

    Args:
        rule_score: Rule-based similarity (0-1)
        semantic_score: Semantic/embedding similarity (0-1)
        rule_weight: Weight for rule-based score (default 0.4)
        semantic_weight: Weight for semantic score (default 0.6)
        embeddings_available: Whether embeddings were actually computed (default True)

    Returns:
        Combined similarity score (0-1)
    """
    # Dynamic fallback: when embeddings unavailable, use 100% rule-based scoring
    # This prevents the hybrid score from being artificially low (e.g., 0.4 * rule_score + 0.6 * 0)
    if not embeddings_available and semantic_score == 0.0:
        return rule_score

    return rule_weight * rule_score + semantic_weight * semantic_score
