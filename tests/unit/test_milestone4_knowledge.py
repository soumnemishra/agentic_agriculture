"""
Unit Tests for ACA Knowledge Layer — Milestone 4, Phase 1
==========================================================

Tests cover:
  - Schema validation (KnowledgeChunk, QueryResult)
  - Abstract interface enforcement (ABCMeta contracts)
  - NumpyVectorStore:
      · Insertion and count
      · Upsert (replace existing chunk_id)
      · Search returns correct top-k ordering
      · Cosine similarity correctness (hand-computed values)
      · Empty-store search returns []
      · Zero-vector handling (no division-by-zero)
      · Dimensionality consistency
      · Clear empties store
      · Chunk without embedding raises ValueError
  - AgronomyKnowledgeTool:
      · End-to-end execute pipeline
      · Formatted output structure
      · Empty result handling
      · Missing / invalid query parameter
      · Integration with ToolRegistry

Dummy Embedder
~~~~~~~~~~~~~~
``CharCountEmbedder`` produces a deterministic, reproducible embedding
based on character-frequency counts.  This is intentionally *not*
semantically meaningful — it exists purely for unit testing the
vector-store mechanics and tool pipeline.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import pytest

from aca.knowledge.interfaces import AbstractEmbedder, AbstractVectorStore
from aca.knowledge.local_store import NumpyVectorStore
from aca.knowledge.schemas import KnowledgeChunk, QueryResult
from aca.tools.agronomy_tool import AgronomyKnowledgeTool
from aca.tools.base_tool import ToolResult
from aca.tools.registry import ToolRegistry


# ══════════════════════════════════════════════════════════════════════
#  Dummy Embedder (for testing only)
# ══════════════════════════════════════════════════════════════════════

class CharCountEmbedder(AbstractEmbedder):
    """
    Test-only embedder that produces a fixed-dimension vector based on
    character-frequency counts over a small alphabet.

    The embedding dimension is 26 (one bin per lowercase letter a-z).
    Each component is the normalised count of that character in the
    input text.  This gives deterministic, reproducible embeddings
    where texts with similar character distributions will naturally
    have higher cosine similarity — sufficient for ranking tests.
    """

    _DIM = 26

    def embed_text(self, text: str) -> List[float]:
        """Produce a 26-d char-frequency vector from *text*."""
        counts = [0.0] * self._DIM
        lower = text.lower()
        for ch in lower:
            idx = ord(ch) - ord("a")
            if 0 <= idx < self._DIM:
                counts[idx] += 1.0

        # L2-normalise so cosine similarity is just the dot product.
        norm = math.sqrt(sum(c * c for c in counts))
        if norm > 0:
            counts = [c / norm for c in counts]
        return counts

    def dimension(self) -> int:
        return self._DIM


# ══════════════════════════════════════════════════════════════════════
#  Fixtures
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture
def embedder() -> CharCountEmbedder:
    """Fresh dummy embedder."""
    return CharCountEmbedder()


@pytest.fixture
def store() -> NumpyVectorStore:
    """Fresh empty vector store."""
    return NumpyVectorStore()


@pytest.fixture
def embedded_chunks(embedder: CharCountEmbedder) -> List[KnowledgeChunk]:
    """A small corpus of pre-embedded knowledge chunks."""
    texts = [
        ("c1", "Irrigation scheduling for wheat requires monitoring soil moisture"),
        ("c2", "Nitrogen deficiency causes yellowing of lower leaves in crops"),
        ("c3", "Soil pH affects nutrient availability and microbial activity"),
        ("c4", "Drip irrigation reduces water waste and improves root zone moisture"),
        ("c5", "Potassium supports disease resistance and drought tolerance"),
    ]
    chunks: List[KnowledgeChunk] = []
    for chunk_id, content in texts:
        emb = embedder.embed_text(content)
        chunks.append(
            KnowledgeChunk.from_dict(
                chunk_id=chunk_id,
                content=content,
                metadata={"source": "test_corpus", "topic": "agronomy"},
                embedding=emb,
            )
        )
    return chunks


@pytest.fixture
def populated_store(
    store: NumpyVectorStore,
    embedded_chunks: List[KnowledgeChunk],
) -> NumpyVectorStore:
    """Vector store pre-loaded with the test corpus."""
    store.add_chunks(embedded_chunks)
    return store


@pytest.fixture
def tool(
    populated_store: NumpyVectorStore,
    embedder: CharCountEmbedder,
) -> AgronomyKnowledgeTool:
    """Agronomy tool wired to populated store and embedder."""
    return AgronomyKnowledgeTool(populated_store, embedder, top_k=3)


# ══════════════════════════════════════════════════════════════════════
#  Schema Tests
# ══════════════════════════════════════════════════════════════════════

class TestKnowledgeChunk:
    """Verify KnowledgeChunk frozen dataclass."""

    def test_creation_from_dict(self) -> None:
        chunk = KnowledgeChunk.from_dict(
            "k1", "Soil moisture management",
            metadata={"source": "manual"},
            embedding=[0.1, 0.2, 0.3],
        )
        assert chunk.chunk_id == "k1"
        assert chunk.content == "Soil moisture management"
        assert chunk.metadata_as_dict() == {"source": "manual"}
        assert chunk.embedding == (0.1, 0.2, 0.3)

    def test_frozen(self) -> None:
        chunk = KnowledgeChunk.from_dict("k1", "text")
        with pytest.raises(AttributeError):
            chunk.content = "changed"  # type: ignore[misc]

    def test_empty_id_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            KnowledgeChunk(chunk_id="", content="text")

    def test_none_embedding_default(self) -> None:
        chunk = KnowledgeChunk.from_dict("k1", "text")
        assert chunk.embedding is None

    def test_to_dict(self) -> None:
        chunk = KnowledgeChunk.from_dict(
            "k1", "hello", embedding=[1.0, 2.0]
        )
        d = chunk.to_dict()
        assert d["chunk_id"] == "k1"
        assert d["content"] == "hello"
        assert d["embedding"] == [1.0, 2.0]


class TestQueryResult:
    """Verify QueryResult frozen dataclass."""

    def test_creation(self) -> None:
        chunk = KnowledgeChunk.from_dict("k1", "text")
        qr = QueryResult(chunk=chunk, similarity_score=0.95)
        assert qr.similarity_score == 0.95

    def test_frozen(self) -> None:
        chunk = KnowledgeChunk.from_dict("k1", "text")
        qr = QueryResult(chunk=chunk, similarity_score=0.5)
        with pytest.raises(AttributeError):
            qr.similarity_score = 0.9  # type: ignore[misc]

    def test_to_dict(self) -> None:
        chunk = KnowledgeChunk.from_dict("k1", "text")
        qr = QueryResult(chunk=chunk, similarity_score=0.8)
        d = qr.to_dict()
        assert d["similarity_score"] == 0.8
        assert d["chunk"]["chunk_id"] == "k1"


# ══════════════════════════════════════════════════════════════════════
#  Interface Contract Tests
# ══════════════════════════════════════════════════════════════════════

class TestAbstractContracts:
    """Ensure ABCMeta prevents incomplete implementations."""

    def test_cannot_instantiate_abstract_embedder(self) -> None:
        with pytest.raises(TypeError):
            AbstractEmbedder()  # type: ignore[abstract]

    def test_cannot_instantiate_abstract_store(self) -> None:
        with pytest.raises(TypeError):
            AbstractVectorStore()  # type: ignore[abstract]

    def test_incomplete_embedder_raises(self) -> None:
        class Bad(AbstractEmbedder):
            pass  # missing embed_text and dimension
        with pytest.raises(TypeError):
            Bad()  # type: ignore[abstract]

    def test_incomplete_store_raises(self) -> None:
        class Bad(AbstractVectorStore):
            pass  # missing add_chunks, search, count, clear
        with pytest.raises(TypeError):
            Bad()  # type: ignore[abstract]

    def test_numpy_store_is_subclass(self) -> None:
        assert issubclass(NumpyVectorStore, AbstractVectorStore)

    def test_char_count_embedder_is_subclass(self) -> None:
        assert issubclass(CharCountEmbedder, AbstractEmbedder)


# ══════════════════════════════════════════════════════════════════════
#  Dummy Embedder Tests
# ══════════════════════════════════════════════════════════════════════

class TestCharCountEmbedder:
    """Sanity-check the test embedder itself."""

    def test_dimension(self, embedder: CharCountEmbedder) -> None:
        assert embedder.dimension() == 26

    def test_output_length(self, embedder: CharCountEmbedder) -> None:
        vec = embedder.embed_text("hello world")
        assert len(vec) == 26

    def test_normalised(self, embedder: CharCountEmbedder) -> None:
        vec = embedder.embed_text("agriculture")
        norm = math.sqrt(sum(v * v for v in vec))
        assert norm == pytest.approx(1.0, abs=1e-9)

    def test_empty_string_produces_zero_vector(
        self, embedder: CharCountEmbedder
    ) -> None:
        vec = embedder.embed_text("")
        assert all(v == 0.0 for v in vec)

    def test_deterministic(self, embedder: CharCountEmbedder) -> None:
        v1 = embedder.embed_text("test")
        v2 = embedder.embed_text("test")
        assert v1 == v2

    def test_similar_texts_closer(
        self, embedder: CharCountEmbedder
    ) -> None:
        """Texts with overlapping characters should be more similar
        than texts with disjoint characters."""
        v_irrigation = np.array(embedder.embed_text("irrigation"))
        v_irrigating = np.array(embedder.embed_text("irrigating"))
        v_zzzzz = np.array(embedder.embed_text("zzzzzzzzz"))

        sim_close = float(np.dot(v_irrigation, v_irrigating))
        sim_far = float(np.dot(v_irrigation, v_zzzzz))
        assert sim_close > sim_far


# ══════════════════════════════════════════════════════════════════════
#  NumpyVectorStore Tests
# ══════════════════════════════════════════════════════════════════════

class TestNumpyVectorStoreInsertion:
    """Test chunk insertion mechanics."""

    def test_add_single_chunk(
        self, store: NumpyVectorStore, embedder: CharCountEmbedder
    ) -> None:
        chunk = KnowledgeChunk.from_dict(
            "k1", "text", embedding=embedder.embed_text("text")
        )
        count = store.add_chunks([chunk])
        assert count == 1
        assert store.count() == 1

    def test_add_multiple_chunks(
        self, store: NumpyVectorStore, embedded_chunks: List[KnowledgeChunk]
    ) -> None:
        count = store.add_chunks(embedded_chunks)
        assert count == 5
        assert store.count() == 5

    def test_add_empty_list(self, store: NumpyVectorStore) -> None:
        count = store.add_chunks([])
        assert count == 0
        assert store.count() == 0

    def test_upsert_replaces_chunk(
        self, store: NumpyVectorStore, embedder: CharCountEmbedder
    ) -> None:
        v1 = embedder.embed_text("original content")
        v2 = embedder.embed_text("updated content")
        chunk1 = KnowledgeChunk.from_dict("k1", "original", embedding=v1)
        chunk2 = KnowledgeChunk.from_dict("k1", "updated", embedding=v2)

        store.add_chunks([chunk1])
        assert store.count() == 1

        store.add_chunks([chunk2])
        assert store.count() == 1  # still 1, not 2

        results = store.search(v2, top_k=1)
        assert results[0].chunk.content == "updated"

    def test_chunk_without_embedding_raises(
        self, store: NumpyVectorStore
    ) -> None:
        chunk = KnowledgeChunk.from_dict("k1", "text")  # no embedding
        with pytest.raises(ValueError, match="no embedding"):
            store.add_chunks([chunk])

    def test_clear_empties_store(
        self, populated_store: NumpyVectorStore
    ) -> None:
        assert populated_store.count() == 5
        populated_store.clear()
        assert populated_store.count() == 0


class TestNumpyVectorStoreSearch:
    """Test similarity search correctness."""

    def test_empty_store_returns_empty_list(
        self, store: NumpyVectorStore
    ) -> None:
        results = store.search([0.1] * 26, top_k=5)
        assert results == []

    def test_search_returns_query_results(
        self,
        populated_store: NumpyVectorStore,
        embedder: CharCountEmbedder,
    ) -> None:
        query_vec = embedder.embed_text("irrigation water moisture")
        results = populated_store.search(query_vec, top_k=3)
        assert len(results) == 3
        assert all(isinstance(r, QueryResult) for r in results)

    def test_results_sorted_descending(
        self,
        populated_store: NumpyVectorStore,
        embedder: CharCountEmbedder,
    ) -> None:
        query_vec = embedder.embed_text("soil moisture irrigation")
        results = populated_store.search(query_vec, top_k=5)
        scores = [r.similarity_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_larger_than_store(
        self,
        populated_store: NumpyVectorStore,
        embedder: CharCountEmbedder,
    ) -> None:
        query_vec = embedder.embed_text("test")
        results = populated_store.search(query_vec, top_k=100)
        # Should return all 5, not crash.
        assert len(results) == 5

    def test_top_k_one(
        self,
        populated_store: NumpyVectorStore,
        embedder: CharCountEmbedder,
    ) -> None:
        query_vec = embedder.embed_text("nitrogen")
        results = populated_store.search(query_vec, top_k=1)
        assert len(results) == 1

    def test_similarity_ranking_correctness(
        self, store: NumpyVectorStore
    ) -> None:
        """
        Insert chunks with known embeddings and verify the ranking
        is correct by hand-computing cosine similarity.
        """
        # Query: [1, 0, 0]
        # Chunk A: [1, 0, 0]  → sim = 1.0  (identical)
        # Chunk B: [0, 1, 0]  → sim = 0.0  (orthogonal)
        # Chunk C: [1, 1, 0]  → sim = 1/√2 ≈ 0.7071
        chunk_a = KnowledgeChunk.from_dict(
            "a", "exact match", embedding=[1.0, 0.0, 0.0]
        )
        chunk_b = KnowledgeChunk.from_dict(
            "b", "orthogonal", embedding=[0.0, 1.0, 0.0]
        )
        chunk_c = KnowledgeChunk.from_dict(
            "c", "partial match", embedding=[1.0, 1.0, 0.0]
        )
        store.add_chunks([chunk_a, chunk_b, chunk_c])

        results = store.search([1.0, 0.0, 0.0], top_k=3)
        assert len(results) == 3

        # First result: exact match (sim = 1.0).
        assert results[0].chunk.chunk_id == "a"
        assert results[0].similarity_score == pytest.approx(1.0, abs=1e-9)

        # Second result: partial match (sim ≈ 0.7071).
        assert results[1].chunk.chunk_id == "c"
        assert results[1].similarity_score == pytest.approx(
            1.0 / math.sqrt(2.0), abs=1e-6
        )

        # Third result: orthogonal (sim = 0.0).
        assert results[2].chunk.chunk_id == "b"
        assert results[2].similarity_score == pytest.approx(0.0, abs=1e-9)

    def test_zero_query_vector_handled(
        self, populated_store: NumpyVectorStore
    ) -> None:
        """A zero-magnitude query should return results with sim = 0."""
        results = populated_store.search([0.0] * 26, top_k=3)
        assert len(results) == 3
        assert all(r.similarity_score == 0.0 for r in results)

    def test_zero_embedding_chunk_handled(
        self, store: NumpyVectorStore
    ) -> None:
        """A chunk with a zero-magnitude embedding should get sim = 0."""
        zero_chunk = KnowledgeChunk.from_dict(
            "zero", "empty", embedding=[0.0, 0.0, 0.0]
        )
        normal_chunk = KnowledgeChunk.from_dict(
            "normal", "content", embedding=[1.0, 0.0, 0.0]
        )
        store.add_chunks([zero_chunk, normal_chunk])

        results = store.search([1.0, 0.0, 0.0], top_k=2)
        # normal_chunk should rank first.
        assert results[0].chunk.chunk_id == "normal"
        assert results[0].similarity_score == pytest.approx(1.0, abs=1e-9)
        # zero_chunk should have sim = 0.
        assert results[1].chunk.chunk_id == "zero"
        assert results[1].similarity_score == pytest.approx(0.0, abs=1e-9)


# ══════════════════════════════════════════════════════════════════════
#  AgronomyKnowledgeTool Tests
# ══════════════════════════════════════════════════════════════════════

class TestAgronomyKnowledgeTool:
    """Test the tool's full execute pipeline."""

    def test_schema_name(self, tool: AgronomyKnowledgeTool) -> None:
        assert tool.schema.name == "agronomy_knowledge"
        assert tool.name == "agronomy_knowledge"

    def test_schema_has_query_parameter(
        self, tool: AgronomyKnowledgeTool
    ) -> None:
        param_names = [p.name for p in tool.schema.parameters]
        assert "query" in param_names

    def test_execute_returns_success(
        self, tool: AgronomyKnowledgeTool
    ) -> None:
        result = tool.execute(query="soil moisture irrigation")
        assert result.success is True
        assert isinstance(result.data, str)
        assert result.metadata["result_count"] > 0

    def test_execute_output_format(
        self, tool: AgronomyKnowledgeTool
    ) -> None:
        result = tool.execute(query="nitrogen deficiency")
        assert "=== Knowledge Results for:" in result.data
        assert "[1]" in result.data
        assert "score:" in result.data
        assert "Source:" in result.data

    def test_execute_respects_top_k(
        self,
        populated_store: NumpyVectorStore,
        embedder: CharCountEmbedder,
    ) -> None:
        tool_k2 = AgronomyKnowledgeTool(populated_store, embedder, top_k=2)
        result = tool_k2.execute(query="irrigation")
        assert result.metadata["result_count"] == 2

    def test_execute_empty_store(
        self, embedder: CharCountEmbedder
    ) -> None:
        empty_store = NumpyVectorStore()
        tool = AgronomyKnowledgeTool(empty_store, embedder)
        result = tool.execute(query="anything")
        assert result.success is True
        assert "No relevant knowledge found" in result.data
        assert result.metadata["result_count"] == 0

    def test_execute_missing_query(
        self, tool: AgronomyKnowledgeTool
    ) -> None:
        result = tool.execute()
        assert result.success is False
        assert "query" in result.error.lower()

    def test_execute_empty_query(
        self, tool: AgronomyKnowledgeTool
    ) -> None:
        result = tool.execute(query="")
        assert result.success is False

    def test_execute_whitespace_only_query(
        self, tool: AgronomyKnowledgeTool
    ) -> None:
        result = tool.execute(query="   ")
        assert result.success is False


class TestAgronomyToolRegistration:
    """Verify the tool works correctly within the ToolRegistry."""

    def test_register_and_invoke(
        self,
        populated_store: NumpyVectorStore,
        embedder: CharCountEmbedder,
    ) -> None:
        registry = ToolRegistry()
        tool = AgronomyKnowledgeTool(populated_store, embedder)
        registry.register(tool)

        assert "agronomy_knowledge" in registry.list_tools()

        result = registry.invoke(
            "agronomy_knowledge", query="soil moisture"
        )
        assert result.success is True
        assert result.metadata["result_count"] > 0

    def test_invoke_validates_params(
        self,
        populated_store: NumpyVectorStore,
        embedder: CharCountEmbedder,
    ) -> None:
        registry = ToolRegistry()
        tool = AgronomyKnowledgeTool(populated_store, embedder)
        registry.register(tool)

        # Unknown parameter should be caught by schema validation.
        result = registry.invoke(
            "agronomy_knowledge", bad_param="oops"
        )
        assert result.success is False
        assert "Unknown parameter" in result.error


# ══════════════════════════════════════════════════════════════════════
#  Integration: Embedder → Store → Tool Pipeline
# ══════════════════════════════════════════════════════════════════════

class TestEndToEndPipeline:
    """Full pipeline: embed chunks, store, query, format output."""

    def test_full_pipeline(self, embedder: CharCountEmbedder) -> None:
        # 1. Create and embed chunks.
        raw_texts = [
            ("irr", "Drip irrigation saves water and targets root zones"),
            ("fert", "Over-fertilisation leads to nitrogen runoff pollution"),
            ("pest", "Integrated pest management reduces pesticide use"),
        ]
        chunks = [
            KnowledgeChunk.from_dict(
                cid, text,
                metadata={"source": "handbook"},
                embedding=embedder.embed_text(text),
            )
            for cid, text in raw_texts
        ]

        # 2. Load into store.
        store = NumpyVectorStore()
        store.add_chunks(chunks)
        assert store.count() == 3

        # 3. Query via tool.
        tool = AgronomyKnowledgeTool(store, embedder, top_k=2)
        result = tool.execute(query="irrigation water drip")

        assert result.success is True
        assert result.metadata["result_count"] == 2
        # The irrigation chunk should rank highest for an irrigation query.
        assert "irrigation" in result.data.lower()
