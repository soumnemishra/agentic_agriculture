"""
ACA Digital Twin — Data Schemas
================================

Immutable, typed data structures for the Digital Twin simulation
subsystem.

Structures
----------
- ``ActionType``      – Enum classifying the kind of intervention the
                        Digital Twin can simulate (IRRIGATE, FERTILIZE).
- ``ProposedAction``  – Frozen dataclass describing a single planned
                        intervention to apply during simulation.
- ``SimulationResult``– Frozen dataclass capturing the full output of a
                        simulation trajectory: the original snapshot, the
                        predicted snapshot, a health delta, and risk
                        flags.

Design Decisions
~~~~~~~~~~~~~~~~
- **Frozen dataclasses** ensure simulation inputs and outputs are
  immutable, making it safe to pass them across threads or cache them
  without defensive copying.
- ``risk_flags`` is stored as a *tuple* of strings (not a list) to
  preserve true immutability within the frozen dataclass.
- The ``health_delta`` is signed: positive means health improved,
  negative means it degraded.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Dict, Tuple

from aca.world_model.schemas import GraphSnapshot


# ── Action-type enumeration ──────────────────────────────────────────────

@unique
class ActionType(Enum):
    """
    Classification of agricultural interventions that the Digital Twin
    can model.

    Members
    -------
    IRRIGATE
        Water application — increases ``soil_moisture`` on the target
        zone node.
    FERTILIZE
        Nitrogen fertiliser application — increases ``nitrogen_level``
        on the target zone node.
    """

    IRRIGATE = "IRRIGATE"
    FERTILIZE = "FERTILIZE"


# ── Proposed action dataclass ────────────────────────────────────────────

@dataclass(frozen=True)
class ProposedAction:
    """
    A single planned intervention to be evaluated by the Digital Twin.

    Attributes:
        action_type: The kind of intervention (``IRRIGATE`` or
                     ``FERTILIZE``).
        target_node_id: ID of the ``EntityNode`` (typically a ZONE) to
                        which the action will be applied.
        amount: Magnitude of the intervention.  Units depend on
                ``action_type``:
                - ``IRRIGATE``:  litres per square metre (L/m²).
                - ``FERTILIZE``: kilograms of nitrogen per hectare
                                 (kg N/ha).
    """

    action_type: ActionType
    target_node_id: str
    amount: float

    def __post_init__(self) -> None:
        """Validate invariants immediately after construction."""
        if not isinstance(self.action_type, ActionType):
            raise TypeError(
                f"ProposedAction.action_type must be an ActionType member, "
                f"got {type(self.action_type).__name__}."
            )
        if not self.target_node_id:
            raise ValueError(
                "ProposedAction.target_node_id must be a non-empty string."
            )
        if not isinstance(self.amount, (int, float)):
            raise TypeError(
                f"ProposedAction.amount must be numeric, "
                f"got {type(self.amount).__name__}."
            )
        if self.amount < 0.0:
            raise ValueError(
                f"ProposedAction.amount must be non-negative, "
                f"got {self.amount}."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the action to a JSON-compatible dict."""
        return {
            "action_type": self.action_type.value,
            "target_node_id": self.target_node_id,
            "amount": self.amount,
        }


# ── Simulation result dataclass ──────────────────────────────────────────

@dataclass(frozen=True)
class SimulationResult:
    """
    Output of a Digital Twin simulation trajectory.

    Attributes:
        original_snapshot: The ``GraphSnapshot`` that was used as the
                           starting point for the simulation.
        predicted_snapshot: The ``GraphSnapshot`` representing the
                           predicted world state after all actions and
                           time-steps have been applied.
        predicted_health_delta: Signed change in the aggregate health
                                index.  Positive ⇒ improvement,
                                negative ⇒ degradation.
        risk_flags: Tuple of human-readable risk warnings generated
                    during the simulation (e.g. ``"Risk of Root Rot"``).
    """

    original_snapshot: GraphSnapshot
    predicted_snapshot: GraphSnapshot
    predicted_health_delta: float
    risk_flags: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate invariants immediately after construction."""
        if not isinstance(self.original_snapshot, GraphSnapshot):
            raise TypeError(
                "SimulationResult.original_snapshot must be a "
                "GraphSnapshot instance."
            )
        if not isinstance(self.predicted_snapshot, GraphSnapshot):
            raise TypeError(
                "SimulationResult.predicted_snapshot must be a "
                "GraphSnapshot instance."
            )
        if not isinstance(self.predicted_health_delta, (int, float)):
            raise TypeError(
                f"SimulationResult.predicted_health_delta must be numeric, "
                f"got {type(self.predicted_health_delta).__name__}."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the result to a JSON-compatible dict."""
        return {
            "original_snapshot": self.original_snapshot.to_dict(),
            "predicted_snapshot": self.predicted_snapshot.to_dict(),
            "predicted_health_delta": self.predicted_health_delta,
            "risk_flags": list(self.risk_flags),
        }
