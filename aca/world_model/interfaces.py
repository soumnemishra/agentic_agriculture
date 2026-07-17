"""
ACA World Model — Abstract Interface
=====================================

Defines the ``AbstractWorldModel`` contract that every concrete world-model
implementation must satisfy.  The interface is intentionally kept narrow so
that downstream consumers (agents, cognitive loop, planning engine) depend
only on the abstract surface, not on the in-memory graph details.

Methods
-------
- ``update_node``   – Create or mutate a node's property bag.
- ``get_node``      – Retrieve a single node by ID.
- ``remove_node``   – Delete a node (and its incident edges).
- ``add_edge``      – Establish a directed relationship between nodes.
- ``remove_edge``   – Remove a specific directed edge.
- ``query_subgraph``– Filter nodes / edges by caller-supplied criteria.
- ``get_state``     – Return an immutable snapshot of the full graph.
- ``ingest_observation`` – Accept an ``ACAMessage`` carrying an
                           ``ObservationPayload`` and update the graph.

Design Decisions
~~~~~~~~~~~~~~~~
- ``abc.ABCMeta`` is used (rather than ``typing.Protocol``) so that
  missing-method errors are raised at *class-definition* time, not at
  first call-site.
- Return types are kept generic (``dict``, ``list``) so that the contract
  does not leak implementation details such as ``networkx`` types.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Forward-reference only — the concrete import happens inside implementors.
# ---------------------------------------------------------------------------
from aca.orchestration.schemas import ACAMessage
from aca.world_model.schemas import EntityNode, SpatialEdge


class AbstractWorldModel(metaclass=ABCMeta):
    """
    Contract for a graph-backed world model that maintains the spatial /
    relational state of the agricultural environment.

    All implementations **must** be thread-safe: concurrent reads and
    writes from multiple agents are expected during normal operation.
    """

    # ── Node operations ───────────────────────────────────────────────

    @abstractmethod
    def update_node(self, node_id: str, data: Dict[str, Any]) -> EntityNode:
        """
        Create a new node or merge *data* into an existing node's
        property bag.

        Args:
            node_id: Unique identifier for the entity.
            data: Key-value properties to set / overwrite.  Must include
                  a ``"type"`` key (str matching a ``NodeType`` member
                  name) when the node does not yet exist.

        Returns:
            The resulting ``EntityNode`` after the update.

        Raises:
            ValueError: If the node does not exist and *data* is missing
                        a valid ``"type"`` key.
            KeyError:   If a ``"type"`` key is supplied for an existing
                        node and it conflicts with the current type.
        """

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[EntityNode]:
        """
        Retrieve a single node.

        Returns:
            The ``EntityNode``, or ``None`` if no node with *node_id*
            exists.
        """

    @abstractmethod
    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node and all of its incident edges.

        Returns:
            ``True`` if the node existed and was removed.
        """

    # ── Edge operations ───────────────────────────────────────────────

    @abstractmethod
    def add_edge(
        self,
        source: str,
        target: str,
        relation_type: str,
        weight: float = 1.0,
    ) -> SpatialEdge:
        """
        Create a directed edge between two existing nodes.

        Args:
            source: ID of the source node.
            target: ID of the target node.
            relation_type: Semantic label for the relationship
                           (e.g. ``"CONTAINS"``, ``"MONITORS"``).
            weight: Optional numeric weight (defaults to ``1.0``).

        Returns:
            The created ``SpatialEdge``.

        Raises:
            KeyError:   If *source* or *target* node does not exist.
            ValueError: If an identical edge already exists (same source,
                        target, and relation_type).
        """

    @abstractmethod
    def remove_edge(
        self, source: str, target: str, relation_type: str
    ) -> bool:
        """
        Remove a specific directed edge.

        Returns:
            ``True`` if the edge existed and was removed.
        """

    # ── Query ─────────────────────────────────────────────────────────

    @abstractmethod
    def query_subgraph(
        self, filter_criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Return the subset of the graph matching *filter_criteria*.

        Supported filter keys (all optional):

        - ``"node_type"``  – ``NodeType`` member name (str).
        - ``"node_ids"``   – explicit list of node IDs.
        - ``"relation_type"`` – edge label to include.
        - ``"properties"`` – dict of property-name → expected-value
                             pairs; only nodes whose properties are a
                             superset are included.

        Returns:
            A dict with keys ``"nodes"`` (list of ``EntityNode``) and
            ``"edges"`` (list of ``SpatialEdge``).
        """

    # ── State snapshot ────────────────────────────────────────────────

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Return an immutable, JSON-serialisable snapshot of the entire
        graph.

        Returns:
            A dict with keys ``"nodes"`` (list of node dicts),
            ``"edges"`` (list of edge dicts), and ``"node_count"`` /
            ``"edge_count"`` summary integers.
        """

    # ── Message ingestion ─────────────────────────────────────────────

    @abstractmethod
    def ingest_observation(self, message: ACAMessage) -> None:
        """
        Accept an ``ACAMessage`` whose ``message_type`` is
        ``MessageType.OBSERVATION`` and apply its measurements to the
        graph.

        The method must:
        1. Validate that the message type is ``OBSERVATION``.
        2. Resolve sensor nodes referenced in the payload.
        3. Upsert property values on those nodes.

        Args:
            message: A validated ``ACAMessage`` instance.

        Raises:
            TypeError:  If the message type is not ``OBSERVATION``.
            ValueError: If the payload is malformed or references
                        unknown sensors (implementation-dependent).
        """
