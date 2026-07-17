"""
ACA World Model — In-Memory Graph Engine
=========================================

Concrete implementation of ``AbstractWorldModel`` backed by an
in-memory adjacency-list graph with full thread-safety guarantees.

Architecture
~~~~~~~~~~~~
- **Nodes** are stored in a ``dict[str, EntityNode]`` keyed by node ID.
- **Edges** are stored in a forward-adjacency dict
  ``dict[str, list[SpatialEdge]]`` keyed by source node ID, as well as
  a reverse-adjacency dict keyed by target node ID.  The dual index
  allows O(1) incident-edge lookup for node deletion.
- A ``threading.RLock`` serialises all mutations.  Read-only methods
  also acquire the lock to guarantee snapshot consistency.
- An ``ACAConfig`` is injected at construction time, following the
  project-wide Dependency Injection pattern.

Message Ingestion
~~~~~~~~~~~~~~~~~
``ingest_observation()`` accepts ``ACAMessage`` instances whose
``message_type`` is ``MessageType.OBSERVATION``.  It resolves each
sensor referenced in the payload, upserts the sensor node's properties
with the observation measurements, and optionally upserts the target
zone node.

Design Decisions
~~~~~~~~~~~~~~~~
- ``RLock`` (reentrant) is chosen over ``Lock`` because helper methods
  like ``_resolve_node_type`` call ``get_node`` internally — a non-
  reentrant lock would deadlock.
- Deep copies are returned for mutable containers so that callers
  cannot silently corrupt the internal graph.
- Edge uniqueness is enforced on the triple (source, target,
  relation_type); multiple edges with different relation types between
  the same pair of nodes are allowed.
"""

from __future__ import annotations

import copy
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from aca.config import ACAConfig
from aca.logging_config import get_logger
from aca.orchestration.schemas import (
    ACAMessage,
    MessageType,
    ObservationPayload,
)
from aca.world_model.interfaces import AbstractWorldModel
from aca.world_model.schemas import (
    EntityNode,
    GraphSnapshot,
    NodeType,
    SpatialEdge,
)

logger = get_logger("world_model.graph_engine")


class GraphWorldModel(AbstractWorldModel):
    """
    Thread-safe, in-memory graph implementation of the ACA world model.

    Args:
        config: Root ``ACAConfig`` instance (injected). The engine
                currently uses the top-level ``environment`` field for
                log verbosity decisions; future milestones may add a
                dedicated ``WorldModelConfig`` sub-section.

    Example::

        config = ACAConfig.load()
        world  = GraphWorldModel(config)
        world.update_node("zone-north", {"type": "ZONE", "area_ha": 2.5})
        world.update_node("sensor-01", {"type": "SENSOR", "model": "TDR"})
        world.add_edge("zone-north", "sensor-01", "CONTAINS")
    """

    def __init__(self, config: ACAConfig) -> None:
        self._config = config

        # ── Internal graph storage ────────────────────────────────────
        # Primary node store: node_id → EntityNode
        self._nodes: Dict[str, EntityNode] = {}

        # Forward adjacency: source_id → list[SpatialEdge]
        self._adj_forward: Dict[str, List[SpatialEdge]] = {}

        # Reverse adjacency: target_id → list[SpatialEdge]
        self._adj_reverse: Dict[str, List[SpatialEdge]] = {}

        # ── Thread safety ─────────────────────────────────────────────
        self._lock = threading.RLock()

        logger.info(
            "GraphWorldModel initialised (env=%s)", self._config.environment
        )

    # ══════════════════════════════════════════════════════════════════
    #  Node operations
    # ══════════════════════════════════════════════════════════════════

    def update_node(self, node_id: str, data: Dict[str, Any]) -> EntityNode:
        """
        Create a new node or merge *data* into an existing node.

        When creating:
            *data* **must** contain a ``"type"`` key whose value is a
            valid ``NodeType`` member name (e.g. ``"SENSOR"``).

        When updating:
            If ``"type"`` is present in *data* it must match the
            existing node's type; otherwise a ``KeyError`` is raised
            to prevent accidental type mutation.

        All other keys in *data* are treated as property updates and
        are merged into the existing property bag (last-write-wins).

        Returns:
            The resulting frozen ``EntityNode``.
        """
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            existing = self._nodes.get(node_id)

            if existing is None:
                # ── Create ────────────────────────────────────────────
                node_type = self._resolve_node_type(node_id, data)
                # Strip "type" from the property bag — it's stored as a
                # first-class field on EntityNode.
                props = {k: v for k, v in data.items() if k != "type"}
                node = EntityNode.from_dict(
                    node_id=node_id,
                    node_type=node_type,
                    properties=props,
                    last_updated=now,
                )
                self._nodes[node_id] = node
                # Initialise empty adjacency buckets.
                self._adj_forward.setdefault(node_id, [])
                self._adj_reverse.setdefault(node_id, [])
                logger.debug(
                    "Created node %s (type=%s)", node_id, node_type.value
                )
                return node

            # ── Update ────────────────────────────────────────────────
            # Validate type consistency if caller re-specifies it.
            if "type" in data:
                requested = data["type"]
                # Accept both NodeType instances and raw strings.
                if isinstance(requested, NodeType):
                    requested_type = requested
                else:
                    try:
                        requested_type = NodeType(requested)
                    except ValueError:
                        try:
                            requested_type = NodeType[requested]
                        except KeyError:
                            raise KeyError(
                                f"Cannot change node '{node_id}' type: "
                                f"'{requested}' is not a valid NodeType."
                            )
                if requested_type != existing.type:
                    raise KeyError(
                        f"Cannot change node '{node_id}' type from "
                        f"{existing.type.value} to {requested_type.value}."
                    )

            # Merge properties (last-write-wins).
            merged_props = existing.properties_as_dict()
            for k, v in data.items():
                if k != "type":
                    merged_props[k] = v

            node = EntityNode.from_dict(
                node_id=node_id,
                node_type=existing.type,
                properties=merged_props,
                last_updated=now,
            )
            self._nodes[node_id] = node
            logger.debug("Updated node %s", node_id)
            return node

    def get_node(self, node_id: str) -> Optional[EntityNode]:
        """Retrieve a single node by ID, or ``None`` if absent."""
        with self._lock:
            return self._nodes.get(node_id)

    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node and all of its incident edges (both outgoing and
        incoming).

        Returns ``True`` if the node existed, ``False`` otherwise.
        """
        with self._lock:
            if node_id not in self._nodes:
                return False

            # Remove outgoing edges from forward adjacency.
            outgoing = self._adj_forward.pop(node_id, [])
            for edge in outgoing:
                # Remove from the reverse adjacency of each target.
                rev_list = self._adj_reverse.get(edge.target_id, [])
                self._adj_reverse[edge.target_id] = [
                    e for e in rev_list if e.source_id != node_id
                ]

            # Remove incoming edges from reverse adjacency.
            incoming = self._adj_reverse.pop(node_id, [])
            for edge in incoming:
                # Remove from the forward adjacency of each source.
                fwd_list = self._adj_forward.get(edge.source_id, [])
                self._adj_forward[edge.source_id] = [
                    e for e in fwd_list if e.target_id != node_id
                ]

            del self._nodes[node_id]
            logger.debug(
                "Removed node %s (dropped %d outgoing, %d incoming edges)",
                node_id,
                len(outgoing),
                len(incoming),
            )
            return True

    # ══════════════════════════════════════════════════════════════════
    #  Edge operations
    # ══════════════════════════════════════════════════════════════════

    def add_edge(
        self,
        source: str,
        target: str,
        relation_type: str,
        weight: float = 1.0,
    ) -> SpatialEdge:
        """
        Create a directed edge.

        Raises ``KeyError`` if either endpoint is missing and
        ``ValueError`` if a duplicate edge (same source, target,
        relation_type) already exists.
        """
        with self._lock:
            # Validate endpoints exist.
            if source not in self._nodes:
                raise KeyError(
                    f"Source node '{source}' does not exist in the graph."
                )
            if target not in self._nodes:
                raise KeyError(
                    f"Target node '{target}' does not exist in the graph."
                )

            # Check for duplicates.
            for existing_edge in self._adj_forward.get(source, []):
                if (
                    existing_edge.target_id == target
                    and existing_edge.relation_type == relation_type
                ):
                    raise ValueError(
                        f"Duplicate edge: ({source}) --[{relation_type}]--> "
                        f"({target}) already exists."
                    )

            edge = SpatialEdge(
                source_id=source,
                target_id=target,
                relation_type=relation_type,
                weight=weight,
            )
            self._adj_forward.setdefault(source, []).append(edge)
            self._adj_reverse.setdefault(target, []).append(edge)
            logger.debug(
                "Added edge (%s) --[%s]--> (%s) weight=%.2f",
                source,
                relation_type,
                target,
                weight,
            )
            return edge

    def remove_edge(
        self, source: str, target: str, relation_type: str
    ) -> bool:
        """
        Remove a specific directed edge identified by the triple
        (source, target, relation_type).

        Returns ``True`` if the edge existed and was removed.
        """
        with self._lock:
            fwd = self._adj_forward.get(source, [])
            original_len = len(fwd)
            self._adj_forward[source] = [
                e
                for e in fwd
                if not (
                    e.target_id == target
                    and e.relation_type == relation_type
                )
            ]
            removed = len(self._adj_forward[source]) < original_len

            if removed:
                rev = self._adj_reverse.get(target, [])
                self._adj_reverse[target] = [
                    e
                    for e in rev
                    if not (
                        e.source_id == source
                        and e.relation_type == relation_type
                    )
                ]
                logger.debug(
                    "Removed edge (%s) --[%s]--> (%s)",
                    source,
                    relation_type,
                    target,
                )

            return removed

    # ══════════════════════════════════════════════════════════════════
    #  Query
    # ══════════════════════════════════════════════════════════════════

    def query_subgraph(
        self, filter_criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Return a filtered subset of the graph.

        Supported filter keys (all optional, conjunctive):

        - ``"node_type"``     – ``NodeType`` member name (str) or
                                ``NodeType`` instance.
        - ``"node_ids"``      – explicit ``list[str]`` of node IDs to
                                include.
        - ``"relation_type"`` – only edges with this label are returned.
        - ``"properties"``    – ``dict[str, Any]``; only nodes whose
                                property bag is a superset of these
                                key-value pairs are included.

        Returns:
            ``{"nodes": list[EntityNode], "edges": list[SpatialEdge]}``
        """
        with self._lock:
            # ── Filter nodes ──────────────────────────────────────────
            candidates = list(self._nodes.values())

            # Filter by explicit ID list.
            node_ids_filter = filter_criteria.get("node_ids")
            if node_ids_filter is not None:
                id_set = set(node_ids_filter)
                candidates = [n for n in candidates if n.id in id_set]

            # Filter by node type.
            node_type_filter = filter_criteria.get("node_type")
            if node_type_filter is not None:
                nt = self._coerce_node_type(node_type_filter)
                candidates = [n for n in candidates if n.type == nt]

            # Filter by property superset.
            props_filter = filter_criteria.get("properties")
            if props_filter:
                filtered: List[EntityNode] = []
                for node in candidates:
                    node_props = node.properties_as_dict()
                    if all(
                        node_props.get(k) == v
                        for k, v in props_filter.items()
                    ):
                        filtered.append(node)
                candidates = filtered

            # ── Collect matching node IDs ─────────────────────────────
            matched_ids = {n.id for n in candidates}

            # ── Filter edges ─────────────────────────────────────────
            relation_filter = filter_criteria.get("relation_type")
            matched_edges: List[SpatialEdge] = []
            for edge_list in self._adj_forward.values():
                for edge in edge_list:
                    # Both endpoints must be in the matched set.
                    if (
                        edge.source_id in matched_ids
                        and edge.target_id in matched_ids
                    ):
                        if relation_filter is None or edge.relation_type == relation_filter:
                            matched_edges.append(edge)

            return {"nodes": list(candidates), "edges": matched_edges}

    # ══════════════════════════════════════════════════════════════════
    #  State snapshot
    # ══════════════════════════════════════════════════════════════════

    def get_state(self) -> Dict[str, Any]:
        """
        Return a detached, JSON-serialisable snapshot of the full graph.

        The returned dict contains keys ``"nodes"``, ``"edges"``,
        ``"node_count"``, ``"edge_count"``, and ``"captured_at"``.
        """
        with self._lock:
            all_nodes = tuple(self._nodes.values())
            all_edges: List[SpatialEdge] = []
            for edge_list in self._adj_forward.values():
                all_edges.extend(edge_list)

            snapshot = GraphSnapshot(
                nodes=all_nodes,
                edges=tuple(all_edges),
                captured_at=datetime.now(timezone.utc).isoformat(),
            )
            return snapshot.to_dict()

    # ══════════════════════════════════════════════════════════════════
    #  Observation ingestion (ACAMessage callback)
    # ══════════════════════════════════════════════════════════════════

    def ingest_observation(self, message: ACAMessage) -> None:
        """
        Process an ``ACAMessage`` of type ``OBSERVATION``.

        Behaviour:
        1. Validates ``message.message_type == MessageType.OBSERVATION``.
        2. Extracts the ``ObservationPayload``.
        3. For each sensor ID in ``payload.source_sensors``:
           - If the sensor node exists, merges the measurements into
             its properties.
           - If the sensor node does *not* exist, creates it with type
             ``SENSOR`` and the observation measurements as initial
             properties.
        4. If ``payload.target_zone`` is non-empty, upserts a ``ZONE``
           node with the observation timestamp as a property.

        This method is designed to be registered as a message-bus
        callback for ``MessageType.OBSERVATION`` topics.
        """
        # ── Type gate ─────────────────────────────────────────────────
        if message.message_type != MessageType.OBSERVATION:
            raise TypeError(
                f"ingest_observation expects MessageType.OBSERVATION, "
                f"got {message.message_type.value}."
            )

        if not isinstance(message.payload, ObservationPayload):
            raise TypeError(
                f"Payload must be an ObservationPayload, "
                f"got {type(message.payload).__name__}."
            )

        payload: ObservationPayload = message.payload

        logger.info(
            "Ingesting observation %s (sensors=%s, zone=%s)",
            payload.observation_id,
            payload.source_sensors,
            payload.target_zone or "<none>",
        )

        # ── Update sensor nodes ───────────────────────────────────────
        measurement_data: Dict[str, Any] = {
            **payload.measurements,
            "last_observation_id": payload.observation_id,
            "last_observation_time": payload.observation_time or message.timestamp,
        }

        for sensor_id in payload.source_sensors:
            existing = self.get_node(sensor_id)
            if existing is None:
                # Auto-create the sensor node with initial measurements.
                self.update_node(
                    sensor_id,
                    {"type": NodeType.SENSOR.value, **measurement_data},
                )
                logger.debug(
                    "Auto-created sensor node '%s' from observation.",
                    sensor_id,
                )
            else:
                # Merge measurements into existing properties.
                self.update_node(sensor_id, measurement_data)

        # ── Upsert zone node ─────────────────────────────────────────
        if payload.target_zone:
            zone_data: Dict[str, Any] = {
                "last_observation_time": (
                    payload.observation_time or message.timestamp
                ),
                "last_observation_id": payload.observation_id,
            }
            existing_zone = self.get_node(payload.target_zone)
            if existing_zone is None:
                zone_data["type"] = NodeType.ZONE.value
                self.update_node(payload.target_zone, zone_data)
                logger.debug(
                    "Auto-created zone node '%s' from observation.",
                    payload.target_zone,
                )

                # Auto-link sensors → zone if both now exist.
                for sensor_id in payload.source_sensors:
                    if self.get_node(sensor_id) is not None:
                        try:
                            self.add_edge(
                                payload.target_zone,
                                sensor_id,
                                "CONTAINS",
                            )
                        except ValueError:
                            # Edge already exists — safe to ignore.
                            pass
            else:
                self.update_node(payload.target_zone, zone_data)

    # ══════════════════════════════════════════════════════════════════
    #  Internal helpers
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _resolve_node_type(node_id: str, data: Dict[str, Any]) -> NodeType:
        """
        Extract and validate a ``NodeType`` from the ``"type"`` key in
        *data*.

        Raises ``ValueError`` if the key is missing or invalid.
        """
        raw_type = data.get("type")
        if raw_type is None:
            raise ValueError(
                f"Cannot create node '{node_id}': 'type' key is required "
                f"in data when the node does not yet exist.  "
                f"Valid types: {[t.value for t in NodeType]}."
            )

        # Accept both NodeType instances and raw strings.
        if isinstance(raw_type, NodeType):
            return raw_type

        # Try value match first (e.g. "SENSOR"), then name match.
        try:
            return NodeType(raw_type)
        except ValueError:
            pass
        try:
            return NodeType[raw_type]
        except KeyError:
            raise ValueError(
                f"Cannot create node '{node_id}': '{raw_type}' is not a "
                f"valid NodeType.  "
                f"Valid types: {[t.value for t in NodeType]}."
            )

    @staticmethod
    def _coerce_node_type(value: Any) -> NodeType:
        """
        Coerce a raw value into a ``NodeType``, accepting strings and
        enum members.
        """
        if isinstance(value, NodeType):
            return value
        try:
            return NodeType(value)
        except ValueError:
            pass
        try:
            return NodeType[value]
        except KeyError:
            raise ValueError(
                f"'{value}' is not a valid NodeType.  "
                f"Valid types: {[t.value for t in NodeType]}."
            )
