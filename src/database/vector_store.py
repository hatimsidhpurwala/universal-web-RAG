"""
Qdrant vector-store operations – create collection, store chunks,
search by similarity, and clear site-specific data.
"""

from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from config.settings import (
    COLLECTION_NAME,
    EMBEDDING_DIMENSION,
    QDRANT_API_KEY,
    QDRANT_LOCAL_DIR,
    QDRANT_URL,
)

logger = logging.getLogger(__name__)


class VectorStore:
    """Thin wrapper around the Qdrant client for the RAG knowledge base."""

    # Class-level cache so Streamlit reruns reuse the same QdrantClient
    # instead of fighting over the local storage lock file.
    _shared_client: Optional[QdrantClient] = None

    def __init__(self) -> None:
        if VectorStore._shared_client is not None:
            self.client = VectorStore._shared_client
        elif QDRANT_URL and QDRANT_API_KEY:
            logger.info("Connecting to Qdrant Cloud at %s", QDRANT_URL)
            self.client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                timeout=30,  # 30-second timeout to prevent hanging
            )
            VectorStore._shared_client = self.client
        else:
            logger.info("Using local Qdrant storage at %s", QDRANT_LOCAL_DIR)
            QDRANT_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(QDRANT_LOCAL_DIR))
            VectorStore._shared_client = self.client

        self._ensure_collection()

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if COLLECTION_NAME not in collections:
                self.client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIMENSION,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Created Qdrant collection '%s'", COLLECTION_NAME)
            else:
                logger.info("Qdrant collection '%s' already exists", COLLECTION_NAME)
        except Exception as exc:
            logger.error("Failed to ensure collection: %s", exc)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store_chunks_for_site(
        self,
        chunks: List[dict],
        site_name: str,
    ) -> int:
        """Upsert embedded chunks into Qdrant.

        Parameters
        ----------
        chunks : list[dict]
            Each chunk must have ``text``, ``embedding``, ``source_url``,
            and ``chunk_index`` keys.
        site_name : str
            A human-friendly label for the source (e.g. domain or filename).

        Returns
        -------
        int
            Number of chunks stored.
        """
        if not chunks:
            return 0

        points = []
        for chunk in chunks:
            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=chunk["embedding"],
                    payload={
                        "text": chunk["text"],
                        "source_url": chunk.get("source_url", ""),
                        "site_name": site_name,
                        "chunk_index": chunk.get("chunk_index", 0),
                    },
                )
            )

        # Upsert in batches of 100 to avoid large payloads
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=COLLECTION_NAME, points=batch)

        logger.info(
            "Stored %d chunks for site '%s' in '%s'",
            len(points),
            site_name,
            COLLECTION_NAME,
        )
        return len(points)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_chunks(
        self,
        query_vector: List[float],
        top_k: int = 8,
        score_threshold: Optional[float] = None,
    ) -> List[dict]:
        """Search for the *top_k* most similar chunks.

        Returns
        -------
        list[dict]
            Each dict contains ``text``, ``source_url``, ``site_name``,
            ``chunk_index``, and ``score``.
        """
        try:
            results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
            )
        except Exception as exc:
            logger.error("Qdrant search failed: %s", exc)
            return []

        chunks = []
        for hit in results:
            chunks.append(
                {
                    "text": hit.payload.get("text", ""),
                    "source_url": hit.payload.get("source_url", ""),
                    "site_name": hit.payload.get("site_name", ""),
                    "chunk_index": hit.payload.get("chunk_index", 0),
                    "score": hit.score,
                }
            )

        logger.info("Search returned %d results (top_k=%d)", len(chunks), top_k)
        return chunks

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_indexed_sites(self) -> List[str]:
        """Return a list of unique site_name values in the collection."""
        try:
            # Scroll through a sample to find unique site names
            results, _ = self.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=500,
                with_payload=["site_name"],
                with_vectors=False,
            )
            sites = set()
            for point in results:
                name = point.payload.get("site_name", "")
                if name:
                    sites.add(name)
            return sorted(sites)
        except Exception as exc:
            logger.error("Failed to list indexed sites: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def clear_site(self, site_name: str) -> None:
        """Delete all chunks belonging to *site_name*."""
        self.client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="site_name",
                        match=MatchValue(value=site_name),
                    )
                ]
            ),
        )
        logger.info("Cleared all chunks for site '%s'", site_name)

    def get_collection_info(self) -> dict:
        """Return basic stats about the collection."""
        try:
            info = self.client.get_collection(COLLECTION_NAME)
            return {
                "name": COLLECTION_NAME,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
            }
        except Exception as exc:
            logger.error("Failed to get collection info: %s", exc)
            return {"name": COLLECTION_NAME, "vectors_count": 0, "points_count": 0}

    def has_site(self, site_name: str) -> bool:
        """Return True if *site_name* already has chunks stored."""
        try:
            results = self.client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="site_name",
                            match=MatchValue(value=site_name),
                        )
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            return len(results[0]) > 0
        except Exception:
            return False
