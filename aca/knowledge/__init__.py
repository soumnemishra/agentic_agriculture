"""
ACA Knowledge Layer
===================

Local-first knowledge retrieval subsystem for the Agricultural
Cognitive Architecture.  Provides vector-indexed storage and semantic
search over agricultural knowledge chunks.

- **interfaces**   — Abstract contracts (``AbstractEmbedder``,
                     ``AbstractVectorStore``).
- **schemas**      — Frozen data structures (``KnowledgeChunk``,
                     ``QueryResult``).
- **local_store**  — Concrete numpy-backed implementation
                     (``NumpyVectorStore``).

Downstream consumers should depend on the abstract interfaces only.
"""

from aca.knowledge.interfaces import AbstractEmbedder, AbstractVectorStore
from aca.knowledge.local_store import NumpyVectorStore
from aca.knowledge.schemas import KnowledgeChunk, QueryResult

__all__ = [
    "AbstractEmbedder",
    "AbstractVectorStore",
    "KnowledgeChunk",
    "NumpyVectorStore",
    "QueryResult",
]
