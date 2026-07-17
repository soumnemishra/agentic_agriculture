"""
ACA Digital Twin Subsystem
==========================

A side-effect-free simulation layer that predicts future farm state
without mutating the live world model.  This package exposes three
layers:

- **interfaces** — Abstract contract (``AbstractDigitalTwin``).
- **schemas**    — Frozen data structures (``ActionType``,
                   ``ProposedAction``, ``SimulationResult``).
- **engine**     — Concrete deterministic simulator
                   (``DeterministicCropSimulator``).

Downstream consumers should depend on ``AbstractDigitalTwin`` only.
"""

from aca.digital_twin.engine import DeterministicCropSimulator
from aca.digital_twin.interfaces import AbstractDigitalTwin
from aca.digital_twin.schemas import ActionType, ProposedAction, SimulationResult

__all__ = [
    "AbstractDigitalTwin",
    "ActionType",
    "DeterministicCropSimulator",
    "ProposedAction",
    "SimulationResult",
]
