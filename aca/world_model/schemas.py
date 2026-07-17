"""
ACA World Model — Data Schemas
==============================

Immutable, typed data structures for the graph elements that make up
the Agricultural Cognitive Architecture's spatial / relational world
model.

Structures
----------
- ``NodeType``    – Enum classifying the kind of entity a node
                    represents (zone, sensor, actuator, asset).
- ``EntityNode``  – Frozen dataclass for a single graph node.
- ``SpatialEdge`` – Frozen dataclass for a directed, weighted edge.
- ``GraphSnapshot`` – Frozen dataclass capturing a full point-in-time
                      snapshot of the graph (used by ``get_state()``).

Design Decisions
~~~~~~~~~~~~~~~~
- **Frozen dataclasses** guarantee that graph state cannot be mutated
  in-place after construction, which simplifies reasoning about
  concurrent access.
- ``EntityNode.properties`` is stored as a *tuple of key-value pairs*
  rather than a ``dict`` so that the dataclass remains truly hashable /
  frozen.  Helper methods convert to / from plain dicts.
- Timestamps use ISO-8601 strings for JSON-serialisability.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, unique
from typing import Any, Dict, List, Tuple


# ── Node-type enumeration ────────────────────────────────────────────────

@unique
class NodeType(Enum):
    """
    Classification of entities representable in the world model.

    Members
    -------
    ZONE
        A physical or logical zone within the farm (e.g. field, plot,
        greenhouse bay).
    SENSOR
        A data-producing device (e.g. soil moisture probe, weather
        station, NDVI camera).
    ACTUATOR
        A controllable device that alters the environment (e.g.
        irrigation valve, fertiliser injector, drone sprayer).
    ASSET
        A passive physical asset tracked for inventory or maintenance
        purposes (e.g. tractor, reservoir, silo).
    """

    ZONE = "ZONE"
    SENSOR = "SENSOR"
    ACTUATOR = "ACTUATOR"
    ASSET = "ASSET"


# ── Frozen node dataclass ────────────────────────────────────────────────

@dataclass(frozen=True)
class EntityNode:
    """
    Immutable representation of a single entity in the world graph.

    Because the dataclass is frozen, ``properties`` is stored as a tuple
    of ``(key, value)`` pairs.  Use the helper class-method
    :meth:`from_dict` for convenient construction and the instance
    method :meth:`properties_as_dict` for convenient reading.

    Attributes:
        id: Globally unique node identifier.
        type: The ``NodeType`` classification of this entity.
        properties: Immutable tuple of ``(key, value)`` property pairs.
        last_updated: ISO-8601 timestamp of the most recent mutation.
    """

    id: str
    type: NodeType
    properties: Tuple[Tuple[str, Any], ...] = ()
    last_updated: str = ""

    def __post_init__(self) -> None:
        """Validate invariants immediately after construction."""
        if not self.id:
            raise ValueError("EntityNode.id must be a non-empty string.")
        if not isinstance(self.type, NodeType):
            raise TypeError(
                f"EntityNode.type must be a NodeType member, got "
                f"{type(self.type).__name__}."
            )

    # ── Convenience helpers ───────────────────────────────────────────

    def properties_as_dict(self) -> Dict[str, Any]:
        """Return the frozen properties tuple as a mutable dict copy."""
        return dict(self.properties)

    @classmethod
    def from_dict(
        cls,
        node_id: str,
        node_type: NodeType,
        properties: Dict[str, Any] | None = None,
        last_updated: str | None = None,
    ) -> "EntityNode":
        """
        Construct an ``EntityNode`` from a plain dict of properties.

        Args:
            node_id: Unique identifier.
            node_type: Entity classification.
            properties: Mutable dict that will be frozen into a tuple.
            last_updated: Timestamp string; defaults to *now* (UTC).

        Returns:
            A new, frozen ``EntityNode``.
        """
        props = properties or {}
        # Deep-copy values so that the caller cannot mutate internals
        # through mutable sub-objects.
        frozen_props = tuple(
            (k, copy.deepcopy(v)) for k, v in sorted(props.items())
        )
        ts = last_updated or datetime.now(timezone.utc).isoformat()
        return cls(
            id=node_id,
            type=node_type,
            properties=frozen_props,
            last_updated=ts,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the node to a JSON-compatible dict."""
        return {
            "id": self.id,
            "type": self.type.value,
            "properties": self.properties_as_dict(),
            "last_updated": self.last_updated,
        }


# ── Frozen edge dataclass ────────────────────────────────────────────────

@dataclass(frozen=True)
class SpatialEdge:
    """
    Immutable directed edge between two ``EntityNode`` instances.

    Attributes:
        source_id: ID of the source node.
        target_id: ID of the target node.
        relation_type: Semantic label (e.g. ``"CONTAINS"``,
                       ``"MONITORS"``, ``"FEEDS"``).
        weight: Numeric weight / strength of the relationship.
    """

    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate invariants immediately after construction."""
        if not self.source_id:
            raise ValueError(
                "SpatialEdge.source_id must be a non-empty string."
            )
        if not self.target_id:
            raise ValueError(
                "SpatialEdge.target_id must be a non-empty string."
            )
        if not self.relation_type:
            raise ValueError(
                "SpatialEdge.relation_type must be a non-empty string."
            )
        if not isinstance(self.weight, (int, float)):
            raise TypeError(
                f"SpatialEdge.weight must be numeric, got "
                f"{type(self.weight).__name__}."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the edge to a JSON-compatible dict."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "weight": self.weight,
        }


# ── Full graph snapshot ──────────────────────────────────────────────────

@dataclass(frozen=True)
class GraphSnapshot:
    """
    Immutable point-in-time snapshot of the entire world graph.

    Returned by ``AbstractWorldModel.get_state()`` so that callers
    receive a safe, detached copy of the current world state.

    Attributes:
        nodes: Tuple of all ``EntityNode`` instances.
        edges: Tuple of all ``SpatialEdge`` instances.
        captured_at: ISO-8601 timestamp of when the snapshot was taken.
    """

    nodes: Tuple[EntityNode, ...] = ()
    edges: Tuple[SpatialEdge, ...] = ()
    captured_at: str = ""

    @property
    def node_count(self) -> int:
        """Number of nodes in this snapshot."""
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        """Number of edges in this snapshot."""
        return len(self.edges)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the full snapshot to a JSON-compatible dict."""
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "captured_at": self.captured_at,
        }
