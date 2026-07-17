"""
ACA Knowledge Layer — Local Numpy Vector Store
================================================

A lightweight, edge-friendly implementation of ``AbstractVectorStore``
that computes cosine similarity using pure numpy operations.

Architecture
~~~~~~~~~~~~
- Embeddings are stored in a 2-D numpy array
  (shape ``[N, D]`` where ``N`` = number of chunks and ``D`` =
  embedding dimensionality).
- Chunks are stored in a parallel ``dict[str, KnowledgeChunk]`` keyed
  by ``chunk_id``, with an ordered list tracking the correspondence
  between dict entries and matrix rows.
- ``search()`` computes cosine similarity between the query vector and
  every stored embedding in a single vectorised operation — no loops,
  no external dependencies beyond numpy.
- A ``threading.RLock`` serialises mutations.  Reads also acquire the
  lock to guarantee snapshot consistency of the parallel structures.

Design Decisions
~~~~~~~~~~~~~~~~
- **No external vector DB** — runs entirely locally on edge hardware.
- The embedding matrix is rebuilt (``np.vstack``) on every
  ``add_chunks`` call.  This is optimal for stores up to ~100 k
  chunks, which is appropriate for on-device agricultural knowledge
  bases.  For larger corpora a memory-mapped approach would be
  preferred.
- Zero-magnitude embeddings are handled gracefully: cosine similarity
  defaults to 0.0 to avoid division-by-zero.
"""

from __future__ import annotations

import threading
from typing import Dict, List

import numpy as np

from aca.knowledge.interfaces import AbstractVectorStore
from aca.knowledge.schemas import KnowledgeChunk, QueryResult
from aca.logging_config import get_logger

logger = get_logger("knowledge.local_store")


class NumpyVectorStore(AbstractVectorStore):
    """
    In-memory vector store backed by a numpy embedding matrix.

    Args:
        (none — all state is initialised internally)

    Example::

        store = NumpyVectorStore()
        store.add_chunks([chunk_with_embedding])
        results = store.search(query_vector, top_k=3)
    """

    def __init__(self) -> None:
        # ── Internal storage ──────────────────────────────────────────
        # Ordered list of chunk_ids mirroring rows in _matrix.
        self._ids: List[str] = []
        # chunk_id → KnowledgeChunk lookup.
        self._chunks: Dict[str, KnowledgeChunk] = {}
        # Embedding matrix: shape [N, D].  None when store is empty.
        self._matrix: np.ndarray | None = None

        # ── Thread safety ─────────────────────────────────────────────
        self._lock = threading.RLock()

        logger.info("NumpyVectorStore initialised (empty)")

    # ══════════════════════════════════════════════════════════════════
    #  Insertion
    # ══════════════════════════════════════════════════════════════════

    def add_chunks(self, chunks: List[KnowledgeChunk]) -> int:
        """
        Insert (or upsert) knowledge chunks with their embeddings.

        Each chunk must have a non-``None`` ``embedding`` field.
        If a chunk with the same ``chunk_id`` already exists, it is
        replaced.

        Returns:
            The number of chunks successfully stored.

        Raises:
            ValueError: If any chunk has a ``None`` embedding.
        """
        if not chunks:
            return 0

        # ── Validate all chunks before mutating state ─────────────────
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(
                    f"KnowledgeChunk '{chunk.chunk_id}' has no embedding. "
                    f"All chunks must be embedded before insertion."
                )

        with self._lock:
            inserted = 0
            for chunk in chunks:
                chunk_id = chunk.chunk_id

                if chunk_id in self._chunks:
                    # Upsert: remove old entry so we can re-append.
                    old_idx = self._ids.index(chunk_id)
                    self._ids.pop(old_idx)
                    # Defer matrix rebuild to after all insertions.

                self._chunks[chunk_id] = chunk
                self._ids.append(chunk_id)
                inserted += 1

            # ── Rebuild the embedding matrix ──────────────────────────
            self._rebuild_matrix()

            logger.info(
                "Added %d chunk(s) — store now contains %d total.",
                inserted,
                len(self._ids),
            )
            return inserted

    # ══════════════════════════════════════════════════════════════════
    #  Search
    # ══════════════════════════════════════════════════════════════════

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[QueryResult]:
        """
        Find the top-*k* most similar chunks by cosine similarity.

        Cosine similarity is computed as::

            sim(q, v) = (q · v) / (‖q‖ × ‖v‖)

        Returns an empty list if the store has no chunks.
        """
        with self._lock:
            if self._matrix is None or len(self._ids) == 0:
                return []

            # Convert query to numpy column vector.
            q = np.asarray(query_embedding, dtype=np.float64)

            # ── Cosine similarity in one vectorised shot ──────────────
            # _matrix shape: [N, D],  q shape: [D]
            dot_products = self._matrix @ q                  # [N]
            query_norm = np.linalg.norm(q)
            store_norms = np.linalg.norm(self._matrix, axis=1)  # [N]

            # Guard against zero-magnitude vectors.
            denominator = store_norms * query_norm
            # Use np.divide with out/where to avoid RuntimeWarning
            # when denominator is zero.  Elements where the condition
            # is False are left at the initialised value (0.0).
            similarities = np.zeros_like(dot_products)
            mask = denominator > 0.0
            np.divide(dot_products, denominator, out=similarities, where=mask)

            # ── Select top-k ─────────────────────────────────────────
            # Clamp top_k to available count.
            k = min(top_k, len(self._ids))
            # argpartition is O(N) vs O(N log N) for full sort, but
            # we still need to sort the top-k slice for deterministic
            # ordering.
            if k >= len(self._ids):
                top_indices = np.argsort(-similarities)[:k]
            else:
                partitioned = np.argpartition(-similarities, k)[:k]
                top_indices = partitioned[
                    np.argsort(-similarities[partitioned])
                ]

            results: List[QueryResult] = []
            for idx in top_indices:
                chunk_id = self._ids[idx]
                results.append(
                    QueryResult(
                        chunk=self._chunks[chunk_id],
                        similarity_score=float(similarities[idx]),
                    )
                )

            return results

    # ══════════════════════════════════════════════════════════════════
    #  Introspection
    # ══════════════════════════════════════════════════════════════════

    def count(self) -> int:
        """Return the number of chunks currently stored."""
        with self._lock:
            return len(self._ids)

    def clear(self) -> None:
        """Remove all chunks and free the embedding matrix."""
        with self._lock:
            self._ids.clear()
            self._chunks.clear()
            self._matrix = None
            logger.info("NumpyVectorStore cleared.")

    # ══════════════════════════════════════════════════════════════════
    #  Internal helpers
    # ══════════════════════════════════════════════════════════════════

    def _rebuild_matrix(self) -> None:
        """
        Reconstruct the embedding matrix from the current chunk list.

        Must be called under ``_lock``.
        """
        if not self._ids:
            self._matrix = None
            return

        rows: List[np.ndarray] = []
        for chunk_id in self._ids:
            emb = self._chunks[chunk_id].embedding
            # Embedding existence is guaranteed by add_chunks validation.
            rows.append(np.asarray(emb, dtype=np.float64))

        self._matrix = np.vstack(rows)  # [N, D]
