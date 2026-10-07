"""Text embedders behind one protocol, so tests and offline dev never download a model.

All embedders return L2-normalised float32 matrices: cosine similarity == dot product,
matching the cosine space of the ChromaDB collection.
"""

import hashlib
import re
import threading
from functools import lru_cache
from typing import Protocol

import numpy as np

from d2s.config import get_settings


class Embedder(Protocol):
    dim: int
    name: str

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        """Short, noisy inputs: job-post sentences, search boxes."""
        ...

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Canonical entries: taxonomy skill and occupation labels/descriptions."""
        ...


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class SentenceTransformerEmbedder:
    """Sentence-Transformers v5 model; uses the asymmetric query/document encoders."""

    def __init__(self, model_name: str, backend: str = "torch", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        self.name = model_name
        self._batch = batch_size
        self._model = SentenceTransformer(model_name, backend=backend)
        get_dim = getattr(self._model, "get_embedding_dimension", None) or \
            self._model.get_sentence_embedding_dimension  # renamed in sentence-transformers 6
        self.dim = get_dim()
        self._lock = threading.Lock()  # torch models aren't safe to call from many threads at once

    def _encode(self, fn, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        with self._lock:
            out = fn(texts, batch_size=self._batch, normalize_embeddings=True, convert_to_numpy=True)
        return _normalize(out)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(self._model.encode_query, texts)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(self._model.encode_document, texts)


class HashingEmbedder:
    """Deterministic character-n-gram hashing embedder (no model, no network).

    Captures spelling overlap only, not meaning; good enough for tests and wiring, never for prod.
    """

    def __init__(self, dim: int = 512, ngram: tuple[int, int] = (3, 5)):
        self.dim = dim
        self.name = f"hashing-{dim}"
        self._ngram = ngram

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"[a-z0-9+#.]+", text.lower()):
            padded = f" {token} "
            for n in range(self._ngram[0], self._ngram[1] + 1):
                for i in range(max(1, len(padded) - n + 1)):
                    h = int.from_bytes(hashlib.blake2b(padded[i:i + n].encode(), digest_size=8).digest())
                    v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return v

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.stack([self._vector(t) for t in texts]))

    embed_documents = embed_queries


@lru_cache
def get_embedder() -> Embedder:
    """Process-wide embedder chosen by settings; loaded once per worker."""
    s = get_settings()
    if s.embedding_backend == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(s.embedding_model, s.embedding_backend, s.embedding_batch_size)
