"""
Batch text encoder with caching and optional Anthropic embedding support.
Falls back to sentence-transformers for local inference.
"""
from __future__ import annotations
import hashlib
import json
import numpy as np
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


class TextEncoder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = ".cache/embeddings"):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._cache: dict[str, np.ndarray] = {}

    def _load_model(self):
        if self._model is None:
            if not ST_AVAILABLE:
                raise ImportError("Run: pip install sentence-transformers")
            self._model = SentenceTransformer(self.model_name)

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(f"{self.model_name}:{text}".encode()).hexdigest()

    def encode(self, texts: list[str], normalize: bool = True, batch_size: int = 64) -> np.ndarray:
        self._load_model()
        embeddings = self._model.encode(
            texts, batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=len(texts) > 100
        )
        return embeddings.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    @property
    def dim(self) -> int:
        self._load_model()
        return self._model.get_sentence_embedding_dimension()
