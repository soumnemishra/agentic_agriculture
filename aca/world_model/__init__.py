"""
ACA World Model Subsystem
=========================

Graph-backed spatial / relational model of the agricultural
environment.  This package exposes three layers:

- **interfaces** — Abstract contract (``AbstractWorldModel``).
- **schemas**    — Frozen data structures (``NodeType``, ``EntityNode``,
                   ``SpatialEdge``, ``GraphSnapshot``).
- **graph_engine** — Concrete in-memory implementation
                     (``GraphWorldModel``).

Downstream consumers should depend on ``AbstractWorldModel`` only.
"""

from aca.world_model.graph_engine import GraphWorldModel
from aca.world_model.interfaces import AbstractWorldModel
from aca.world_model.schemas import (
    EntityNode,
    GraphSnapshot,
    NodeType,
    SpatialEdge,
)

__all__ = [
    "AbstractWorldModel",
    "GraphWorldModel",
    "EntityNode",
    "GraphSnapshot",
    "NodeType",
    "SpatialEdge",
]
