"""
ACA Tools — Agronomy Knowledge Tool
=====================================

A ``BaseTool`` implementation that enables agents to query the
agricultural knowledge base using natural-language questions.

Pipeline
~~~~~~~~
1. The agent calls ``execute(query="How to treat nitrogen deficiency")``.
2. The tool embeds the query via the injected ``AbstractEmbedder``.
3. It searches the injected ``AbstractVectorStore`` for the *top-k*
   most relevant ``KnowledgeChunk`` objects.
4. It formats the results into a clean, numbered text block suitable
   for LLM consumption and returns a ``ToolResult``.

Design Decisions
~~~~~~~~~~~~~~~~
- Both ``AbstractVectorStore`` and ``AbstractEmbedder`` are injected
  via the constructor — no global singletons, no import-time side
  effects.
- ``top_k`` is configurable at construction time with a sensible
  default of 3 (ideal for context-window-limited LLM agents running
  on edge hardware).
- The output format is deliberately plain-text (not JSON) so that an
  LLM agent can read it directly without parsing overhead.
"""

from __future__ import annotations

from typing import Any, List

from aca.knowledge.interfaces import AbstractEmbedder, AbstractVectorStore
from aca.logging_config import get_logger
from aca.tools.base_tool import (
    BaseTool,
    ToolParameter,
    ToolResult,
    ToolSchema,
)

logger = get_logger("tools.agronomy_knowledge")


class AgronomyKnowledgeTool(BaseTool):
    """
    Tool for semantic search over the agricultural knowledge base.

    Args:
        store: The vector store holding embedded knowledge chunks.
        embedder: The text-to-vector embedder for query encoding.
        top_k: Maximum number of results to return per query
               (default ``3``).

    Example::

        tool = AgronomyKnowledgeTool(store, embedder, top_k=5)
        result = tool.execute(query="optimal soil pH for tomatoes")
        print(result.data)  # formatted knowledge snippets
    """

    def __init__(
        self,
        store: AbstractVectorStore,
        embedder: AbstractEmbedder,
        top_k: int = 3,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._top_k = top_k
        logger.info(
            "AgronomyKnowledgeTool initialised (top_k=%d)", self._top_k
        )

    # ── BaseTool contract ─────────────────────────────────────────────

    @property
    def schema(self) -> ToolSchema:
        """Declarative schema for agent discovery."""
        return ToolSchema(
            name="agronomy_knowledge",
            description=(
                "Search the local agricultural knowledge base for "
                "information relevant to a natural-language query. "
                "Returns the most relevant knowledge snippets ranked "
                "by semantic similarity."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    description=(
                        "Natural-language question or topic to search "
                        "for (e.g. 'nitrogen deficiency symptoms in "
                        "wheat')."
                    ),
                    param_type="str",
                    required=True,
                ),
            ],
            returns=(
                "A formatted text block containing the top-k most "
                "relevant knowledge snippets with similarity scores."
            ),
        )

    def execute(self, **kwargs: Any) -> ToolResult:
        """
        Embed the query, search the knowledge store, and return
        formatted results.

        Kwargs:
            query (str): The natural-language search query.

        Returns:
            A ``ToolResult`` with ``data`` containing a formatted
            string of the top-k knowledge snippets.
        """
        # ── Parameter extraction ──────────────────────────────────────
        query: str | None = kwargs.get("query")
        if not query or not isinstance(query, str) or not query.strip():
            return ToolResult(
                success=False,
                error="Parameter 'query' must be a non-empty string.",
            )

        query = query.strip()

        try:
            # ── 1. Embed the query ────────────────────────────────────
            query_embedding: List[float] = self._embedder.embed_text(query)

            # ── 2. Search the vector store ────────────────────────────
            results = self._store.search(
                query_embedding=query_embedding,
                top_k=self._top_k,
            )

            # ── 3. Format the output ──────────────────────────────────
            if not results:
                formatted = (
                    "No relevant knowledge found for the query: "
                    f"'{query}'."
                )
                return ToolResult(
                    success=True,
                    data=formatted,
                    metadata={"query": query, "result_count": 0},
                )

            formatted = self._format_results(query, results)

            logger.info(
                "Knowledge query '%s' returned %d result(s).",
                query,
                len(results),
            )

            return ToolResult(
                success=True,
                data=formatted,
                metadata={
                    "query": query,
                    "result_count": len(results),
                    "top_score": results[0].similarity_score,
                },
            )

        except Exception as exc:
            logger.exception(
                "Error executing agronomy knowledge query: %s", exc
            )
            return ToolResult(
                success=False,
                error=f"Knowledge query failed: {exc}",
            )

    # ── Formatting helper ─────────────────────────────────────────────

    @staticmethod
    def _format_results(query: str, results: list) -> str:
        """
        Format search results into a clean, numbered text block
        suitable for LLM consumption.

        Output format::

            === Knowledge Results for: "<query>" ===

            [1] (score: 0.9234)
            <content>
            — Source: <source_metadata>

            [2] (score: 0.8712)
            <content>
            — Source: <source_metadata>
        """
        lines: List[str] = [
            f'=== Knowledge Results for: "{query}" ===',
            "",
        ]

        for rank, result in enumerate(results, start=1):
            chunk = result.chunk
            meta = chunk.metadata_as_dict()
            source = meta.get("source", "unknown")

            lines.append(
                f"[{rank}] (score: {result.similarity_score:.4f})"
            )
            lines.append(chunk.content)
            lines.append(f"  — Source: {source}")
            lines.append("")

        return "\n".join(lines).rstrip()
