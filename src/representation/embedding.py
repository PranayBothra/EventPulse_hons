"""Embedding providers supporting sentence-transformers and offline TF-IDF."""

import abc
import hashlib
import logging
from typing import List, Optional, Union
import numpy as np

logger = logging.getLogger("eventpulse.embedding")


class BaseEmbeddingProvider(abc.ABC):
    """Abstract interface for text embedding providers."""

    @property
    @abc.abstractmethod
    def model_id(self) -> str:
        """Return the unique model identifier string."""
        pass

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Return the vector dimensionality."""
        pass

    @abc.abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Compute L2-normalized embeddings for a list of texts.

        Args:
            texts: List of input strings.

        Returns:
            2D numpy array of shape (len(texts), dimension) and dtype float32.
        """
        pass

    def embed_single(self, text: str) -> np.ndarray:
        """Compute L2-normalized embedding for a single text."""
        return self.embed_texts([text])[0]


class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    """Dense semantic embedding provider using SentenceTransformers on CPU."""

    _instance = None
    _loaded_model_id = None

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 16,
        device: str = "cpu",
    ):
        self._model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = self._get_or_load_model(model_name, device)
        self._dim = self._model.get_sentence_embedding_dimension()

    @classmethod
    def _get_or_load_model(cls, model_name: str, device: str):
        """Reuse existing model instance to avoid redundant weight loading."""
        if cls._instance is None or cls._loaded_model_id != model_name:
            from sentence_transformers import SentenceTransformer
            import torch
            logger.info("Loading SentenceTransformer model '%s' on %s...", model_name, device)
            model_kwargs = {}
            if any(k in model_name.lower() for k in ["harrier", "qwen"]):
                # On CPU, native float32 is up to 6x faster than software-emulated bfloat16
                model_kwargs = {"dtype": torch.float32}
            cls._instance = SentenceTransformer(model_name, device=device, model_kwargs=model_kwargs)
            cls._loaded_model_id = model_name
        return cls._instance

    @property
    def model_id(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        # Handle empty/whitespace strings
        sanitized = [t if t and t.strip() else "empty article" for t in texts]

        embeddings = self._model.encode(
            sanitized,
            batch_size=self.batch_size,
            device=self.device,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)


class TfidfEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic, lightweight TF-IDF / Hashing embedding provider for testing and fallback."""

    def __init__(self, dimension: int = 384):
        from sklearn.feature_extraction.text import HashingVectorizer
        self._dim = dimension
        # HashingVectorizer is stateless and deterministic across instances
        self.vectorizer = HashingVectorizer(
            n_features=dimension,
            norm="l2",
            alternate_sign=False,
            token_pattern=r"(?u)\b\w+\b",
        )

    @property
    def model_id(self) -> str:
        return f"tfidf_hashing_{self._dim}"

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        sanitized = [t if t and t.strip() else "empty text" for t in texts]
        sparse_mat = self.vectorizer.transform(sanitized)
        dense_arr = sparse_mat.toarray().astype(np.float32)

        # Ensure unit L2 norm
        norms = np.linalg.norm(dense_arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dense_arr / norms


def get_embedding_provider(
    provider_type: str = "sentence-transformers",
    model_name: str = "all-MiniLM-L6-v2",
) -> BaseEmbeddingProvider:
    """Factory to instantiate the configured embedding provider.

    Args:
        provider_type: 'sentence-transformers' or 'tfidf' / 'fallback'.
        model_name: HuggingFace model identifier if using sentence-transformers.
    """
    normalized = provider_type.lower().strip()
    if normalized in ("tfidf", "fallback", "hashing"):
        return TfidfEmbeddingProvider(dimension=384)
    else:
        try:
            return SentenceTransformerEmbeddingProvider(model_name=model_name)
        except Exception as e:
            logger.warning(
                "Failed to initialize SentenceTransformer provider (%s). Falling back to TfidfEmbeddingProvider.",
                e,
            )
            return TfidfEmbeddingProvider(dimension=384)
