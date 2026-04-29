"""
Embedding generation using Sentence-Transformers (all-MiniLM-L6-v2).
Provides both batch document embedding and single-query embedding.
"""

from __future__ import annotations

import logging
from typing import List

from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Lazy-loaded global model instance
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Return (and lazily initialise) the embedding model."""
    global _model
    if _model is None:
        logger.info("Loading embedding model '%s' …", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _model


def embed_chunks(
    chunks: List[dict],
    *,
    batch_size: int = 32,
    show_progress: bool = True,
) -> List[dict]:
    """Add an ``"embedding"`` key to each chunk dict.

    Parameters
    ----------
    chunks : list[dict]
        Must contain at least a ``"text"`` key.
    batch_size : int
        Batch size for the encoder.
    show_progress : bool
        Whether to display a progress bar.

    Returns
    -------
    list[dict]
        The same list, mutated in-place with ``"embedding"`` added.
    """
    if not chunks:
        return chunks

    model = _get_model()
    texts = [c["text"] for c in chunks]
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )

    for chunk, vec in zip(chunks, vectors):
        chunk["embedding"] = vec.tolist()

    logger.info("Embedded %d chunks (dim=%d)", len(chunks), len(vectors[0]))
    return chunks


def embed_query(query: str) -> List[float]:
    """Embed a single query string and return the vector.

    Parameters
    ----------
    query : str
        The search query to embed.

    Returns
    -------
    list[float]
        384-dimensional embedding vector.
    """
    model = _get_model()
    vector = model.encode(query, convert_to_numpy=True)
    return vector.tolist()
