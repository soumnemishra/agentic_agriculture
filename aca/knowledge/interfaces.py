"""
ACA Knowledge Layer — Abstract Interfaces
===========================================

Defines the contracts for the two pluggable components in the
knowledge-retrieval pipeline:

1. ``AbstractEmbedder``    – Converts text into a dense vector
                             representation.
2. ``AbstractVectorStore`` – Persists ``KnowledgeChunk`` objects with
                             their embeddings and supports
                             similarity-based retrieval.

Design Decisions
~~~~~~~~~~~~~~~~
- ``abc.ABCMeta`` is used so that missing-method errors surface at
  class-definition time rather than at first call-site.
- The interfaces are deliberately minimal, making it easy to swap
  between local-only implementations (TF-IDF, ONNX sentence
  transformers) and cloud-backed alternatives without touching
  consumer code.
- ``AbstractEmbedder`` returns a plain ``List[float]`` rather than a
  numpy array to keep the contract framework-agnostic.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import List

from aca.knowledge.schemas import KnowledgeChunk, QueryResult


class AbstractEmbedder(metaclass=ABCMeta):
    """
    Contract for a text-to-vector embedding component.

    Implementations may use anything from bag-of-characters heuristics
    (for testing) to full ONNX sentence-transformer models (for
    production on edge hardware).
    """

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Convert *text* into a dense floating-point vector.

        Args:
            text: The input string to embed.

        Returns:
            A list of floats representing the embedding vector.
            The dimensionality must be consistent across calls within
            the same embedder instance.
        """

    @abstractmethod
    def dimension(self) -> int:
        """
        Return the fixed dimensionality of embeddings produced by
        this embedder.

        This is used by vector stores to pre-validate chunk embeddings
        at insertion time.
        """


class AbstractVectorStore(metaclass=ABCMeta):
    """
    Contract for a vector-indexed knowledge store.

    Implementations manage a collection of ``KnowledgeChunk`` objects,
    each associated with an embedding vector.  The primary operation is
    ``search``, which finds the chunks whose embeddings are most
    similar to a query vector.
    """

    @abstractmethod
    def add_chunks(self, chunks: List[KnowledgeChunk]) -> int:
        """
        Insert one or more knowledge chunks into the store.

        Each chunk **must** carry a non-``None`` ``embedding``.
        Chunks whose ``chunk_id`` already exists in the store should
        be silently replaced (upsert semantics).

        Args:
            chunks: List of ``KnowledgeChunk`` instances to store.

        Returns:
            The number of chunks that were successfully inserted or
            updated.

        Raises:
            ValueError: If any chunk lacks an embedding.
        """

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[QueryResult]:
        """
        Retrieve the top-*k* most similar chunks to the query vector.

        Args:
            query_embedding: The dense query vector (same
                             dimensionality as stored embeddings).
            top_k: Maximum number of results to return.

        Returns:
            A list of ``QueryResult`` objects sorted by descending
            similarity score.  May contain fewer than *top_k* entries
            if the store has fewer chunks.
        """

    @abstractmethod
    def count(self) -> int:
        """Return the number of chunks currently stored."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all chunks from the store."""
