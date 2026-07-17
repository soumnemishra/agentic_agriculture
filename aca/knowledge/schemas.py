"""
ACA Knowledge Layer — Data Schemas
====================================

Immutable, typed data structures for the knowledge retrieval subsystem.

Structures
----------
- ``KnowledgeChunk`` – A discrete unit of agricultural knowledge
                       (text content, metadata, optional embedding).
- ``QueryResult``    – A single search hit pairing a ``KnowledgeChunk``
                       with its cosine-similarity score.

Design Decisions
~~~~~~~~~~~~~~~~
- **Frozen dataclasses** guarantee that knowledge chunks cannot be
  silently mutated once stored — any update requires explicit
  replacement.
- ``KnowledgeChunk.metadata`` is stored as a *tuple of key-value
  pairs* (like ``EntityNode.properties``) so the dataclass remains
  truly immutable and hashable.
- ``KnowledgeChunk.embedding`` is stored as a *tuple of floats*
  rather than a list, preserving frozen semantics.
- Both structures provide ``to_dict()`` helpers for JSON
  serialisation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


# ── Knowledge chunk dataclass ────────────────────────────────────────────

@dataclass(frozen=True)
class KnowledgeChunk:
    """
    An atomic unit of agricultural knowledge suitable for vector
    retrieval.

    Attributes:
        chunk_id: Globally unique identifier for this chunk.
        content: The textual content of the knowledge fragment
                 (e.g. a paragraph from a crop-management guide).
        metadata: Immutable tuple of ``(key, value)`` metadata pairs
                  (e.g. ``("source", "FAO_guide")``,
                  ``("topic", "irrigation")``).
        embedding: Optional pre-computed embedding vector stored as a
                   tuple of floats.  ``None`` if not yet embedded.
    """

    chunk_id: str
    content: str
    metadata: Tuple[Tuple[str, Any], ...] = ()
    embedding: Optional[Tuple[float, ...]] = None

    def __post_init__(self) -> None:
        """Validate invariants immediately after construction."""
        if not self.chunk_id:
            raise ValueError(
                "KnowledgeChunk.chunk_id must be a non-empty string."
            )
        if not isinstance(self.content, str):
            raise TypeError(
                f"KnowledgeChunk.content must be a string, "
                f"got {type(self.content).__name__}."
            )

    # ── Convenience helpers ───────────────────────────────────────────

    def metadata_as_dict(self) -> Dict[str, Any]:
        """Return the frozen metadata tuple as a mutable dict copy."""
        return dict(self.metadata)

    @classmethod
    def from_dict(
        cls,
        chunk_id: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> "KnowledgeChunk":
        """
        Construct a ``KnowledgeChunk`` from plain Python collections.

        Args:
            chunk_id: Unique identifier.
            content: Text content of the chunk.
            metadata: Mutable dict that will be frozen into a tuple.
            embedding: Mutable list of floats that will be frozen into
                       a tuple.  Pass ``None`` if not yet embedded.

        Returns:
            A new, frozen ``KnowledgeChunk``.
        """
        meta = metadata or {}
        frozen_meta = tuple(
            (k, copy.deepcopy(v)) for k, v in sorted(meta.items())
        )
        frozen_emb = tuple(embedding) if embedding is not None else None
        return cls(
            chunk_id=chunk_id,
            content=content,
            metadata=frozen_meta,
            embedding=frozen_emb,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the chunk to a JSON-compatible dict."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": self.metadata_as_dict(),
            "embedding": list(self.embedding) if self.embedding else None,
        }


# ── Query result dataclass ───────────────────────────────────────────────

@dataclass(frozen=True)
class QueryResult:
    """
    A single search result from the vector store.

    Attributes:
        chunk: The matched ``KnowledgeChunk``.
        similarity_score: Cosine-similarity score ∈ [-1.0, 1.0].
                          Higher values indicate stronger semantic
                          matches.
    """

    chunk: KnowledgeChunk
    similarity_score: float

    def __post_init__(self) -> None:
        """Validate invariants immediately after construction."""
        if not isinstance(self.chunk, KnowledgeChunk):
            raise TypeError(
                "QueryResult.chunk must be a KnowledgeChunk instance."
            )
        if not isinstance(self.similarity_score, (int, float)):
            raise TypeError(
                f"QueryResult.similarity_score must be numeric, "
                f"got {type(self.similarity_score).__name__}."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the result to a JSON-compatible dict."""
        return {
            "chunk": self.chunk.to_dict(),
            "similarity_score": self.similarity_score,
        }
