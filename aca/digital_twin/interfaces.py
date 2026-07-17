"""
ACA Digital Twin — Abstract Interface
======================================

Defines the ``AbstractDigitalTwin`` contract that every concrete
simulation engine must satisfy.

The Digital Twin is a *pure prediction* component: it receives an
immutable ``GraphSnapshot`` and a sequence of ``ProposedAction``
objects, runs a forward simulation, and returns a new
``SimulationResult`` — **without** mutating the live
``GraphWorldModel``.

Design Decisions
~~~~~~~~~~~~~~~~
- ``abc.ABCMeta`` is used so that missing-method errors surface at
  class-definition time.
- The interface is deliberately minimal (one method) to keep the
  contract narrow and easy to swap implementations (deterministic,
  stochastic, ML-based, etc.).
- The ``hours_ahead`` parameter controls the simulation horizon,
  allowing callers to ask "what happens in 6 hours?" vs "what happens
  in 48 hours?".
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import List

from aca.digital_twin.schemas import ProposedAction, SimulationResult
from aca.world_model.schemas import GraphSnapshot


class AbstractDigitalTwin(metaclass=ABCMeta):
    """
    Contract for a Digital Twin simulation engine.

    Implementations receive a frozen world snapshot, apply a list of
    proposed agricultural interventions over a time horizon, and return
    a ``SimulationResult`` containing the predicted future state,
    health-index change, and any risk flags.

    All implementations **must** be side-effect-free: the live world
    model graph must never be modified.
    """

    @abstractmethod
    def simulate_trajectory(
        self,
        current_state: GraphSnapshot,
        actions: List[ProposedAction],
        hours_ahead: int,
    ) -> SimulationResult:
        """
        Run a forward simulation from *current_state*.

        Args:
            current_state: An immutable ``GraphSnapshot`` representing
                           the world as it exists right now.
            actions: Zero or more ``ProposedAction`` instances to apply
                     at the start of the simulation window.
            hours_ahead: Number of one-hour discrete time-steps to
                         simulate forward.  Must be ≥ 1.

        Returns:
            A ``SimulationResult`` containing the original snapshot,
            the predicted snapshot, the aggregate health delta, and
            any risk flags generated during the simulation.

        Raises:
            ValueError: If *hours_ahead* < 1 or if any action
                        references a node not present in the snapshot.
        """
