"""
Unit tests for ACA World Model — Milestone 3, Phase 1
======================================================

Tests cover:
  - Schema validation (EntityNode, SpatialEdge, NodeType, GraphSnapshot)
  - Abstract interface enforcement (ABCMeta contract)
  - GraphWorldModel node CRUD
  - GraphWorldModel edge management
  - GraphWorldModel subgraph queries
  - GraphWorldModel state snapshots
  - Observation message ingestion
  - Thread safety under concurrent mutations
  - Error handling for missing nodes, duplicate edges, schema mismatches
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Dict

import pytest

from aca.config import ACAConfig
from aca.orchestration.schemas import (
    ACAMessage,
    MessageType,
    ObservationPayload,
    create_message,
)
from aca.world_model.graph_engine import GraphWorldModel
from aca.world_model.interfaces import AbstractWorldModel
from aca.world_model.schemas import (
    EntityNode,
    GraphSnapshot,
    NodeType,
    SpatialEdge,
)


# ══════════════════════════════════════════════════════════════════════
#  Fixtures
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture
def config() -> ACAConfig:
    """Default ACAConfig for testing."""
    return ACAConfig()


@pytest.fixture
def world(config: ACAConfig) -> GraphWorldModel:
    """Fresh GraphWorldModel instance."""
    return GraphWorldModel(config)


@pytest.fixture
def populated_world(world: GraphWorldModel) -> GraphWorldModel:
    """World with a few nodes and edges already in place."""
    world.update_node("zone-north", {"type": "ZONE", "area_ha": 2.5})
    world.update_node("zone-south", {"type": "ZONE", "area_ha": 3.0})
    world.update_node("sensor-01", {"type": "SENSOR", "model": "TDR-300"})
    world.update_node("sensor-02", {"type": "SENSOR", "model": "Davis-VP2"})
    world.update_node("pump-01", {"type": "ACTUATOR", "power_kw": 5.5})
    world.update_node("tractor-01", {"type": "ASSET", "brand": "John Deere"})
    world.add_edge("zone-north", "sensor-01", "CONTAINS")
    world.add_edge("zone-south", "sensor-02", "CONTAINS")
    world.add_edge("zone-north", "pump-01", "CONTROLS")
    return world


# ══════════════════════════════════════════════════════════════════════
#  Schema Tests
# ══════════════════════════════════════════════════════════════════════

class TestNodeType:
    """Verify NodeType enum members and uniqueness."""

    def test_members(self) -> None:
        assert NodeType.ZONE.value == "ZONE"
        assert NodeType.SENSOR.value == "SENSOR"
        assert NodeType.ACTUATOR.value == "ACTUATOR"
        assert NodeType.ASSET.value == "ASSET"

    def test_member_count(self) -> None:
        assert len(NodeType) == 4

    def test_from_value(self) -> None:
        assert NodeType("SENSOR") is NodeType.SENSOR


class TestEntityNode:
    """Verify EntityNode frozen dataclass behaviour."""

    def test_creation_from_dict(self) -> None:
        node = EntityNode.from_dict(
            "n1", NodeType.ZONE, {"area_ha": 10.0}
        )
        assert node.id == "n1"
        assert node.type == NodeType.ZONE
        assert node.properties_as_dict() == {"area_ha": 10.0}
        assert node.last_updated  # non-empty

    def test_frozen(self) -> None:
        node = EntityNode.from_dict("n1", NodeType.ZONE)
        with pytest.raises(AttributeError):
            node.id = "changed"  # type: ignore[misc]

    def test_empty_id_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            EntityNode(id="", type=NodeType.ZONE)

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="NodeType"):
            EntityNode(id="bad", type="NOT_A_TYPE")  # type: ignore[arg-type]

    def test_to_dict_roundtrip(self) -> None:
        node = EntityNode.from_dict(
            "n1", NodeType.SENSOR, {"model": "TDR", "depth_cm": 30}
        )
        d = node.to_dict()
        assert d["id"] == "n1"
        assert d["type"] == "SENSOR"
        assert d["properties"]["model"] == "TDR"


class TestSpatialEdge:
    """Verify SpatialEdge frozen dataclass behaviour."""

    def test_creation(self) -> None:
        edge = SpatialEdge("a", "b", "CONTAINS", 0.8)
        assert edge.source_id == "a"
        assert edge.target_id == "b"
        assert edge.relation_type == "CONTAINS"
        assert edge.weight == 0.8

    def test_default_weight(self) -> None:
        edge = SpatialEdge("a", "b", "MONITORS")
        assert edge.weight == 1.0

    def test_frozen(self) -> None:
        edge = SpatialEdge("a", "b", "CONTAINS")
        with pytest.raises(AttributeError):
            edge.weight = 2.0  # type: ignore[misc]

    def test_empty_source_raises(self) -> None:
        with pytest.raises(ValueError, match="source_id"):
            SpatialEdge("", "b", "R")

    def test_empty_target_raises(self) -> None:
        with pytest.raises(ValueError, match="target_id"):
            SpatialEdge("a", "", "R")

    def test_empty_relation_raises(self) -> None:
        with pytest.raises(ValueError, match="relation_type"):
            SpatialEdge("a", "b", "")

    def test_to_dict(self) -> None:
        d = SpatialEdge("a", "b", "FEEDS", 0.5).to_dict()
        assert d == {
            "source_id": "a",
            "target_id": "b",
            "relation_type": "FEEDS",
            "weight": 0.5,
        }


class TestGraphSnapshot:
    """Verify GraphSnapshot frozen dataclass."""

    def test_empty_snapshot(self) -> None:
        snap = GraphSnapshot()
        assert snap.node_count == 0
        assert snap.edge_count == 0

    def test_to_dict(self) -> None:
        node = EntityNode.from_dict("n1", NodeType.ZONE)
        edge = SpatialEdge("n1", "n2", "R")
        snap = GraphSnapshot(
            nodes=(node,), edges=(edge,), captured_at="2026-01-01T00:00:00Z"
        )
        d = snap.to_dict()
        assert d["node_count"] == 1
        assert d["edge_count"] == 1
        assert len(d["nodes"]) == 1
        assert len(d["edges"]) == 1


# ══════════════════════════════════════════════════════════════════════
#  Interface Contract Tests
# ══════════════════════════════════════════════════════════════════════

class TestAbstractWorldModel:
    """Ensure ABCMeta prevents incomplete implementations."""

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            AbstractWorldModel()  # type: ignore[abstract]

    def test_missing_method_raises(self) -> None:
        """A subclass missing even one abstract method cannot be instantiated."""

        class PartialModel(AbstractWorldModel):
            def update_node(self, node_id, data):
                pass
            # Missing: get_node, remove_node, add_edge, remove_edge,
            #          query_subgraph, get_state, ingest_observation

        with pytest.raises(TypeError):
            PartialModel()  # type: ignore[abstract]

    def test_graph_world_model_is_subclass(self) -> None:
        assert issubclass(GraphWorldModel, AbstractWorldModel)


# ══════════════════════════════════════════════════════════════════════
#  GraphWorldModel — Node Operations
# ══════════════════════════════════════════════════════════════════════

class TestGraphWorldModelNodes:
    """Test create / read / update / delete for nodes."""

    def test_create_node(self, world: GraphWorldModel) -> None:
        node = world.update_node("z1", {"type": "ZONE", "area_ha": 5.0})
        assert node.id == "z1"
        assert node.type == NodeType.ZONE
        assert node.properties_as_dict()["area_ha"] == 5.0

    def test_get_existing_node(self, world: GraphWorldModel) -> None:
        world.update_node("s1", {"type": "SENSOR"})
        result = world.get_node("s1")
        assert result is not None
        assert result.id == "s1"

    def test_get_missing_node_returns_none(self, world: GraphWorldModel) -> None:
        assert world.get_node("nonexistent") is None

    def test_update_merges_properties(self, world: GraphWorldModel) -> None:
        world.update_node("s1", {"type": "SENSOR", "model": "A"})
        world.update_node("s1", {"firmware": "v2.1"})
        node = world.get_node("s1")
        assert node is not None
        props = node.properties_as_dict()
        assert props["model"] == "A"
        assert props["firmware"] == "v2.1"

    def test_update_overwrites_existing_property(
        self, world: GraphWorldModel
    ) -> None:
        world.update_node("s1", {"type": "SENSOR", "reading": 1.0})
        world.update_node("s1", {"reading": 2.5})
        node = world.get_node("s1")
        assert node is not None
        assert node.properties_as_dict()["reading"] == 2.5

    def test_create_without_type_raises(self, world: GraphWorldModel) -> None:
        with pytest.raises(ValueError, match="type"):
            world.update_node("bad", {"value": 42})

    def test_create_with_invalid_type_raises(
        self, world: GraphWorldModel
    ) -> None:
        with pytest.raises(ValueError, match="not a valid NodeType"):
            world.update_node("bad", {"type": "DRAGON"})

    def test_update_with_conflicting_type_raises(
        self, world: GraphWorldModel
    ) -> None:
        world.update_node("s1", {"type": "SENSOR"})
        with pytest.raises(KeyError, match="Cannot change"):
            world.update_node("s1", {"type": "ACTUATOR"})

    def test_update_with_same_type_succeeds(
        self, world: GraphWorldModel
    ) -> None:
        world.update_node("s1", {"type": "SENSOR"})
        node = world.update_node("s1", {"type": "SENSOR", "v": 1})
        assert node.type == NodeType.SENSOR

    def test_remove_node(self, world: GraphWorldModel) -> None:
        world.update_node("s1", {"type": "SENSOR"})
        assert world.remove_node("s1") is True
        assert world.get_node("s1") is None

    def test_remove_missing_node_returns_false(
        self, world: GraphWorldModel
    ) -> None:
        assert world.remove_node("ghost") is False

    def test_remove_node_cascades_edges(
        self, populated_world: GraphWorldModel
    ) -> None:
        """Removing a node should also remove all incident edges."""
        populated_world.remove_node("sensor-01")
        state = populated_world.get_state()
        edge_endpoints = [
            (e["source_id"], e["target_id"]) for e in state["edges"]
        ]
        # No edge should reference the removed sensor.
        for src, tgt in edge_endpoints:
            assert src != "sensor-01"
            assert tgt != "sensor-01"


# ══════════════════════════════════════════════════════════════════════
#  GraphWorldModel — Edge Operations
# ══════════════════════════════════════════════════════════════════════

class TestGraphWorldModelEdges:
    """Test add / remove / duplicate detection for edges."""

    def test_add_edge(self, world: GraphWorldModel) -> None:
        world.update_node("a", {"type": "ZONE"})
        world.update_node("b", {"type": "SENSOR"})
        edge = world.add_edge("a", "b", "CONTAINS")
        assert edge.source_id == "a"
        assert edge.target_id == "b"
        assert edge.relation_type == "CONTAINS"
        assert edge.weight == 1.0

    def test_add_edge_custom_weight(self, world: GraphWorldModel) -> None:
        world.update_node("a", {"type": "ZONE"})
        world.update_node("b", {"type": "SENSOR"})
        edge = world.add_edge("a", "b", "MONITORS", weight=0.75)
        assert edge.weight == 0.75

    def test_add_edge_missing_source_raises(
        self, world: GraphWorldModel
    ) -> None:
        world.update_node("b", {"type": "SENSOR"})
        with pytest.raises(KeyError, match="Source node"):
            world.add_edge("ghost", "b", "R")

    def test_add_edge_missing_target_raises(
        self, world: GraphWorldModel
    ) -> None:
        world.update_node("a", {"type": "ZONE"})
        with pytest.raises(KeyError, match="Target node"):
            world.add_edge("a", "ghost", "R")

    def test_duplicate_edge_raises(self, world: GraphWorldModel) -> None:
        world.update_node("a", {"type": "ZONE"})
        world.update_node("b", {"type": "SENSOR"})
        world.add_edge("a", "b", "CONTAINS")
        with pytest.raises(ValueError, match="Duplicate edge"):
            world.add_edge("a", "b", "CONTAINS")

    def test_different_relation_same_endpoints_allowed(
        self, world: GraphWorldModel
    ) -> None:
        world.update_node("a", {"type": "ZONE"})
        world.update_node("b", {"type": "ACTUATOR"})
        world.add_edge("a", "b", "CONTAINS")
        edge2 = world.add_edge("a", "b", "CONTROLS")
        assert edge2.relation_type == "CONTROLS"

    def test_remove_edge(self, world: GraphWorldModel) -> None:
        world.update_node("a", {"type": "ZONE"})
        world.update_node("b", {"type": "SENSOR"})
        world.add_edge("a", "b", "CONTAINS")
        assert world.remove_edge("a", "b", "CONTAINS") is True

    def test_remove_nonexistent_edge_returns_false(
        self, world: GraphWorldModel
    ) -> None:
        assert world.remove_edge("x", "y", "Z") is False


# ══════════════════════════════════════════════════════════════════════
#  GraphWorldModel — Queries
# ══════════════════════════════════════════════════════════════════════

class TestGraphWorldModelQuery:
    """Test subgraph filtering."""

    def test_query_by_node_type(
        self, populated_world: GraphWorldModel
    ) -> None:
        result = populated_world.query_subgraph({"node_type": "SENSOR"})
        assert len(result["nodes"]) == 2
        assert all(n.type == NodeType.SENSOR for n in result["nodes"])

    def test_query_by_node_ids(
        self, populated_world: GraphWorldModel
    ) -> None:
        result = populated_world.query_subgraph(
            {"node_ids": ["zone-north", "sensor-01"]}
        )
        assert len(result["nodes"]) == 2
        ids = {n.id for n in result["nodes"]}
        assert ids == {"zone-north", "sensor-01"}

    def test_query_by_properties(
        self, populated_world: GraphWorldModel
    ) -> None:
        result = populated_world.query_subgraph(
            {"properties": {"model": "TDR-300"}}
        )
        assert len(result["nodes"]) == 1
        assert result["nodes"][0].id == "sensor-01"

    def test_query_by_relation_type(
        self, populated_world: GraphWorldModel
    ) -> None:
        result = populated_world.query_subgraph(
            {"node_type": "ZONE", "relation_type": "CONTROLS"}
        )
        # Only zone-north has a CONTROLS edge, but both zones match
        # the node_type filter. Edges are filtered to CONTROLS only.
        controls_edges = result["edges"]
        for edge in controls_edges:
            assert edge.relation_type == "CONTROLS"

    def test_empty_filter_returns_all(
        self, populated_world: GraphWorldModel
    ) -> None:
        result = populated_world.query_subgraph({})
        assert len(result["nodes"]) == 6  # all nodes
        assert len(result["edges"]) == 3  # all edges

    def test_no_match_returns_empty(
        self, populated_world: GraphWorldModel
    ) -> None:
        result = populated_world.query_subgraph(
            {"node_ids": ["does-not-exist"]}
        )
        assert len(result["nodes"]) == 0
        assert len(result["edges"]) == 0


# ══════════════════════════════════════════════════════════════════════
#  GraphWorldModel — State Snapshot
# ══════════════════════════════════════════════════════════════════════

class TestGraphWorldModelState:
    """Test get_state() snapshot."""

    def test_empty_state(self, world: GraphWorldModel) -> None:
        state = world.get_state()
        assert state["node_count"] == 0
        assert state["edge_count"] == 0

    def test_populated_state(
        self, populated_world: GraphWorldModel
    ) -> None:
        state = populated_world.get_state()
        assert state["node_count"] == 6
        assert state["edge_count"] == 3
        assert "captured_at" in state

    def test_state_is_serialisable(
        self, populated_world: GraphWorldModel
    ) -> None:
        import json

        state = populated_world.get_state()
        # Must not raise
        serialised = json.dumps(state)
        assert isinstance(serialised, str)


# ══════════════════════════════════════════════════════════════════════
#  GraphWorldModel — Observation Ingestion
# ══════════════════════════════════════════════════════════════════════

class TestObservationIngestion:
    """Test ingest_observation() callback hook."""

    def _make_observation_message(
        self,
        sensors: list[str] | None = None,
        zone: str = "",
        measurements: dict[str, float] | None = None,
    ) -> ACAMessage:
        """Helper to construct a valid OBSERVATION ACAMessage."""
        return create_message(
            source="test-harness",
            destination="world_model",
            message_type=MessageType.OBSERVATION,
            payload=ObservationPayload(
                observation_id="obs-001",
                source_sensors=sensors or ["sensor-01"],
                target_zone=zone,
                observation_time="2026-07-14T12:00:00Z",
                measurements=measurements or {"soil_moisture": 0.42},
            ),
        )

    def test_ingest_updates_existing_sensor(
        self, world: GraphWorldModel
    ) -> None:
        world.update_node("sensor-01", {"type": "SENSOR"})
        msg = self._make_observation_message(
            measurements={"soil_moisture": 0.55}
        )
        world.ingest_observation(msg)
        node = world.get_node("sensor-01")
        assert node is not None
        props = node.properties_as_dict()
        assert props["soil_moisture"] == 0.55
        assert props["last_observation_id"] == "obs-001"

    def test_ingest_auto_creates_sensor(
        self, world: GraphWorldModel
    ) -> None:
        msg = self._make_observation_message(
            sensors=["new-sensor"], measurements={"temp_c": 28.3}
        )
        world.ingest_observation(msg)
        node = world.get_node("new-sensor")
        assert node is not None
        assert node.type == NodeType.SENSOR
        assert node.properties_as_dict()["temp_c"] == 28.3

    def test_ingest_upserts_zone(self, world: GraphWorldModel) -> None:
        msg = self._make_observation_message(
            sensors=["s1"], zone="field-A"
        )
        world.ingest_observation(msg)
        zone = world.get_node("field-A")
        assert zone is not None
        assert zone.type == NodeType.ZONE

    def test_ingest_creates_sensor_zone_edge(
        self, world: GraphWorldModel
    ) -> None:
        msg = self._make_observation_message(
            sensors=["s1"], zone="field-A"
        )
        world.ingest_observation(msg)
        state = world.get_state()
        edges = state["edges"]
        assert any(
            e["source_id"] == "field-A"
            and e["target_id"] == "s1"
            and e["relation_type"] == "CONTAINS"
            for e in edges
        )

    def test_ingest_wrong_message_type_raises(
        self, world: GraphWorldModel
    ) -> None:
        # Construct the message directly, bypassing create_message()
        # validation, so we can test the world model's own type guard.
        msg = ACAMessage(
            uuid="test-uuid",
            timestamp="2026-07-14T12:00:00Z",
            source="test",
            destination="world_model",
            message_type=MessageType.DECISION,
            confidence=1.0,
            priority=3,
            payload=ObservationPayload(observation_id="x"),
        )
        with pytest.raises(TypeError, match="OBSERVATION"):
            world.ingest_observation(msg)


# ══════════════════════════════════════════════════════════════════════
#  Thread Safety
# ══════════════════════════════════════════════════════════════════════

class TestThreadSafety:
    """Verify concurrent access does not corrupt internal state."""

    def test_concurrent_node_creation(self, world: GraphWorldModel) -> None:
        """Spawn many threads, each creating a unique node."""
        errors: list[Exception] = []
        num_threads = 50

        def create_node(idx: int) -> None:
            try:
                world.update_node(
                    f"node-{idx}",
                    {"type": "SENSOR", "index": idx},
                )
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=create_node, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent creation: {errors}"
        state = world.get_state()
        assert state["node_count"] == num_threads

    def test_concurrent_read_write(
        self, populated_world: GraphWorldModel
    ) -> None:
        """Mix reads and writes across threads."""
        errors: list[Exception] = []

        def writer(idx: int) -> None:
            try:
                populated_world.update_node(
                    f"writer-{idx}", {"type": "ASSET", "v": idx}
                )
            except Exception as exc:
                errors.append(exc)

        def reader() -> None:
            try:
                populated_world.get_state()
                populated_world.query_subgraph({"node_type": "SENSOR"})
            except Exception as exc:
                errors.append(exc)

        threads = []
        for i in range(20):
            threads.append(threading.Thread(target=writer, args=(i,)))
            threads.append(threading.Thread(target=reader))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent R/W: {errors}"
