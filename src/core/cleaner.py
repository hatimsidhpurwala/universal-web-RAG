"""
Text cleaning utilities – normalisation, whitespace collapsing,
and chunk-level deduplication.
"""

from __future__ import annotations

import re
import logging
from typing import List

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize *text* by collapsing whitespace and cleaning artefacts.

    • Collapses multiple spaces into one.
    • Collapses 3+ consecutive newlines into two.
    • Strips leading/trailing whitespace.
    """
    # Collapse horizontal whitespace (tabs, spaces) but keep newlines
    text = re.sub(r"[^\S\n]+", " ", text)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def deduplicate_chunks(
    chunks: List[dict],
    *,
    similarity_ratio: float = 0.90,
) -> List[dict]:
    """Remove near-duplicate chunks based on exact text overlap.

    Each chunk is expected to have a ``"text"`` key.  Two chunks are
    considered duplicates if the shorter text is contained in the longer
    one (substring check), **or** if they share ≥ *similarity_ratio* of
    their words.

    Parameters
    ----------
    chunks : list[dict]
        Chunk dicts with at least a ``"text"`` key.
    similarity_ratio : float
        Word-overlap ratio above which chunks are considered duplicates.

    Returns
    -------
    list[dict]
        De-duplicated list (order preserved).
    """
    if not chunks:
        return []

    unique: list[dict] = []
    seen_texts: list[set[str]] = []

    for chunk in chunks:
        text = chunk.get("text", "")
        words = set(text.lower().split())
        is_dup = False

        for idx, existing_words in enumerate(seen_texts):
            if not words or not existing_words:
                continue
            overlap = len(words & existing_words)
            smaller = min(len(words), len(existing_words))
            if smaller > 0 and overlap / smaller >= similarity_ratio:
                is_dup = True
                break

        if not is_dup:
            unique.append(chunk)
            seen_texts.append(words)

    removed = len(chunks) - len(unique)
    if removed:
        logger.info("Deduplication removed %d / %d chunks", removed, len(chunks))

    return unique
