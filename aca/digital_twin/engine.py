"""
ACA Digital Twin — Deterministic Crop Simulator
=================================================

Concrete implementation of ``AbstractDigitalTwin`` that performs a
step-by-step deterministic forward simulation on a cloned
``GraphSnapshot``.

Physics Model
~~~~~~~~~~~~~
The simulator operates on ZONE nodes and updates three core properties
per discrete one-hour time-step:

1. **soil_moisture** ∈ [0.0, 1.0]
   - Decays each step via exponential evapotranspiration::

         moisture *= (1 - evaporation_rate)

   - IRRIGATE actions add an instant boost at step 0::

         moisture += irrigation_moisture_gain × amount

   - Clamped to [0.0, 1.0] after every step.

2. **nitrogen_level** ∈ [0.0, 1.0]
   - Decays each step via exponential leaching / plant uptake::

         nitrogen *= (1 - nitrogen_leaching_rate)

   - FERTILIZE actions add an instant boost at step 0::

         nitrogen += fertilizer_nitrogen_gain × amount

   - Clamped to [0.0, 1.0] after every step.

3. **health_index** ∈ [0.0, 1.0]
   - If moisture and nitrogen are both within the optimal band
     [0.30, 0.80], health recovers::

         health += base_health_recovery

   - Otherwise, penalties accumulate::

         if moisture < 0.20 or moisture > 0.90:
             health -= moisture_stress_penalty
         if nitrogen < 0.20 or nitrogen > 0.80:
             health -= nitrogen_stress_penalty

   - Clamped to [0.0, 1.0] after every step.

Risk Flags
~~~~~~~~~~
- ``"Risk of Root Rot"``  — emitted when ``soil_moisture > 0.90``
  at any step.
- ``"Risk of Wilting"``   — emitted when ``soil_moisture < 0.20``
  at any step.

Design Decisions
~~~~~~~~~~~~~~~~
- Physical constants are injected via ``DigitalTwinConfig`` through the
  ``ACAConfig`` DI root — no global singletons, no hardcoded magic
  numbers.
- The live ``GraphWorldModel`` is never touched.  All mutations happen
  on plain-dict copies of node properties that are frozen back into new
  ``EntityNode`` instances at the end.
- Non-ZONE nodes pass through the simulation unchanged.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Set, Tuple

from aca.config import ACAConfig, DigitalTwinConfig
from aca.digital_twin.interfaces import AbstractDigitalTwin
from aca.digital_twin.schemas import ActionType, ProposedAction, SimulationResult
from aca.logging_config import get_logger
from aca.world_model.schemas import EntityNode, GraphSnapshot, NodeType

logger = get_logger("digital_twin.engine")

# ── Optimal-range constants ──────────────────────────────────────────────
# These define the "comfort zone" for moisture and nitrogen within which
# the crop health index recovers rather than degrades.

_MOISTURE_OPTIMAL_LOW: float = 0.30# moisture low optimal threshold

_MOISTURE_OPTIMAL_HIGH: float = 0.80
_NITROGEN_OPTIMAL_LOW: float = 0.20
_NITROGEN_OPTIMAL_HIGH: float = 0.80

# Risk thresholds — hard boundaries that generate risk flags.
_MOISTURE_ROOT_ROT_THRESHOLD: float = 0.90
_MOISTURE_WILTING_THRESHOLD: float = 0.20


class DeterministicCropSimulator(AbstractDigitalTwin):
    """
    Step-function crop simulator operating on ``GraphSnapshot`` data.

    The simulator clones the snapshot's ZONE nodes into mutable dicts,
    applies actions at ``t=0``, then iterates ``hours_ahead`` one-hour
    steps of environmental decay and health-index updates.  At the end,
    it freezes everything back into a new ``GraphSnapshot`` and returns
    a ``SimulationResult``.

    Args:
        config: Root ``ACAConfig`` instance (injected).  The engine
                reads ``config.digital_twin`` for all physical
                constants.

    Example::

        config = ACAConfig()
        sim    = DeterministicCropSimulator(config)
        result = sim.simulate_trajectory(snapshot, [irrigate_action], 24)
    """

    def __init__(self, config: ACAConfig) -> None:
        self._config = config
        self._dt: DigitalTwinConfig = config.digital_twin
        logger.info(
            "DeterministicCropSimulator initialised "
            "(evap=%.3f, leach=%.3f, env=%s)",
            self._dt.evaporation_rate,
            self._dt.nitrogen_leaching_rate,
            self._config.environment,
        )

    # ══════════════════════════════════════════════════════════════════
    #  Public API
    # ══════════════════════════════════════════════════════════════════

    def simulate_trajectory(
        self,
        current_state: GraphSnapshot,
        actions: List[ProposedAction],
        hours_ahead: int,
    ) -> SimulationResult:
        """
        Run a deterministic forward simulation.

        Steps
        -----
        1. Validate inputs.
        2. Build a mutable working copy of ZONE-node properties.
        3. Apply IRRIGATE / FERTILIZE actions at ``t=0``.
        4. Iterate ``hours_ahead`` one-hour decay + health steps.
        5. Collect risk flags emitted during the simulation.
        6. Freeze the predicted state into a new ``GraphSnapshot``.
        7. Compute the aggregate health delta.

        Returns:
            A ``SimulationResult`` with original and predicted
            snapshots, health delta, and risk flags.
        """
        # ── 1. Validate ───────────────────────────────────────────────
        if hours_ahead < 1:
            raise ValueError(
                f"hours_ahead must be ≥ 1, got {hours_ahead}."
            )

        # Build an index of snapshot nodes for quick lookup.
        node_index: Dict[str, EntityNode] = {
            n.id: n for n in current_state.nodes
        }

        # Validate that every action targets an existing node.
        for action in actions:
            if action.target_node_id not in node_index:
                raise ValueError(
                    f"Action targets node '{action.target_node_id}' which "
                    f"does not exist in the snapshot."
                )

        # ── 2. Mutable working copies for ZONE nodes ─────────────────
        # Only ZONE nodes participate in the simulation; other node
        # types (SENSOR, ACTUATOR, ASSET) pass through unchanged.
        zone_states: Dict[str, Dict[str, float]] = {}
        for node in current_state.nodes:
            if node.type == NodeType.ZONE:
                props = node.properties_as_dict()
                zone_states[node.id] = {
                    "soil_moisture": float(props.get("soil_moisture", 0.50)),
                    "nitrogen_level": float(props.get("nitrogen_level", 0.50)),
                    "health_index": float(props.get("health_index", 0.70)),
                }

        # Record initial aggregate health for delta computation.
        initial_health = self._aggregate_health(zone_states)

        # ── 3. Apply actions at t=0 ───────────────────────────────────
        for action in actions:
            if action.target_node_id not in zone_states:
                # Action targets a non-ZONE node — skip silently.
                logger.debug(
                    "Skipping action %s on non-ZONE node '%s'.",
                    action.action_type.value,
                    action.target_node_id,
                )
                continue
            self._apply_action(zone_states[action.target_node_id], action)

        # ── 4. Step-function simulation ───────────────────────────────
        risk_flags: Set[str] = set()

        for hour in range(1, hours_ahead + 1):
            for zone_id, state in zone_states.items():
                self._step_decay(state)
                self._step_health(state)
                self._check_risks(state, risk_flags)

            logger.debug(
                "Simulation step %d/%d complete.", hour, hours_ahead
            )

        # ── 5. Compute final aggregate health & delta ─────────────────
        final_health = self._aggregate_health(zone_states)
        health_delta = final_health - initial_health

        # ── 6. Freeze predicted state ─────────────────────────────────
        predicted_nodes = self._build_predicted_nodes(
            current_state.nodes, zone_states
        )
        predicted_snapshot = GraphSnapshot(
            nodes=tuple(predicted_nodes),
            edges=current_state.edges,  # edges are unchanged
            captured_at=datetime.now(timezone.utc).isoformat(),
        )

        logger.info(
            "Simulation complete: %d hours, health_delta=%.4f, "
            "risk_flags=%s",
            hours_ahead,
            health_delta,
            sorted(risk_flags) or "(none)",
        )

        return SimulationResult(
            original_snapshot=current_state,
            predicted_snapshot=predicted_snapshot,
            predicted_health_delta=round(health_delta, 6),
            risk_flags=tuple(sorted(risk_flags)),
        )

    # ══════════════════════════════════════════════════════════════════
    #  Internal helpers
    # ══════════════════════════════════════════════════════════════════

    def _apply_action(
        self, state: Dict[str, float], action: ProposedAction
    ) -> None:
        """
        Apply a single ``ProposedAction`` to a zone's mutable state
        dict.

        IRRIGATE increases ``soil_moisture``.
        FERTILIZE increases ``nitrogen_level``.
        Both are clamped to [0.0, 1.0].
        """
        if action.action_type == ActionType.IRRIGATE:
            gain = self._dt.irrigation_moisture_gain * action.amount
            state["soil_moisture"] = min(
                1.0, state["soil_moisture"] + gain
            )
            logger.debug(
                "IRRIGATE %.2f → moisture now %.4f",
                action.amount,
                state["soil_moisture"],
            )

        elif action.action_type == ActionType.FERTILIZE:
            gain = self._dt.fertilizer_nitrogen_gain * action.amount
            state["nitrogen_level"] = min(
                1.0, state["nitrogen_level"] + gain
            )
            logger.debug(
                "FERTILIZE %.2f → nitrogen now %.4f",
                action.amount,
                state["nitrogen_level"],
            )

    def _step_decay(self, state: Dict[str, float]) -> None:
        """
        Apply one hour of environmental decay to moisture and nitrogen.

        Uses multiplicative (exponential) decay::

            value *= (1 - rate)

        This models the physical reality that evapotranspiration and
        leaching slow down as the resource depletes.
        """
        # Moisture evapotranspiration.
        state["soil_moisture"] = max(
            0.0,
            state["soil_moisture"] * (1.0 - self._dt.evaporation_rate),
        )
        # Nitrogen leaching / plant uptake.
        state["nitrogen_level"] = max(
            0.0,
            state["nitrogen_level"] * (1.0 - self._dt.nitrogen_leaching_rate),
        )

    def _step_health(self, state: Dict[str, float]) -> None:
        """
        Update the health index based on current moisture and nitrogen.

        If both values are in their respective optimal bands, health
        recovers.  Otherwise, each out-of-range dimension incurs a
        penalty.
        """
        moisture = state["soil_moisture"]
        nitrogen = state["nitrogen_level"]
        health = state["health_index"]

        moisture_ok = _MOISTURE_OPTIMAL_LOW <= moisture <= _MOISTURE_OPTIMAL_HIGH
        nitrogen_ok = _NITROGEN_OPTIMAL_LOW <= nitrogen <= _NITROGEN_OPTIMAL_HIGH

        if moisture_ok and nitrogen_ok:
            # Both in optimal range — health recovers.
            health += self._dt.base_health_recovery
        else:
            # Apply penalties for each out-of-range dimension.
            if not moisture_ok:
                health -= self._dt.moisture_stress_penalty
            if not nitrogen_ok:
                health -= self._dt.nitrogen_stress_penalty

        state["health_index"] = max(0.0, min(1.0, health))

    @staticmethod
    def _check_risks(
        state: Dict[str, float], risk_flags: Set[str]
    ) -> None:
        """
        Inspect the current zone state and append risk flags if
        thresholds are breached.

        Flags are accumulated in a set, so duplicates across time-steps
        are automatically deduplicated.
        """
        if state["soil_moisture"] > _MOISTURE_ROOT_ROT_THRESHOLD:
            risk_flags.add("Risk of Root Rot")
        if state["soil_moisture"] < _MOISTURE_WILTING_THRESHOLD:
            risk_flags.add("Risk of Wilting")

    @staticmethod
    def _aggregate_health(
        zone_states: Dict[str, Dict[str, float]]
    ) -> float:
        """
        Compute the mean health index across all ZONE nodes.

        Returns 0.0 if there are no zones (avoids division by zero).
        """
        if not zone_states:
            return 0.0
        total = sum(s["health_index"] for s in zone_states.values())
        return total / len(zone_states)

    @staticmethod
    def _build_predicted_nodes(
        original_nodes: Tuple[EntityNode, ...],
        zone_states: Dict[str, Dict[str, float]],
    ) -> List[EntityNode]:
        """
        Construct the list of predicted ``EntityNode`` instances.

        ZONE nodes have their properties updated with the simulated
        values.  All other nodes are carried forward unchanged.
        """
        now = datetime.now(timezone.utc).isoformat()
        predicted: List[EntityNode] = []

        for node in original_nodes:
            if node.id in zone_states:
                # Merge simulation results into the original property
                # bag so that non-simulated properties are preserved.
                merged_props = node.properties_as_dict()
                merged_props.update(zone_states[node.id])
                predicted.append(
                    EntityNode.from_dict(
                        node_id=node.id,
                        node_type=node.type,
                        properties=merged_props,
                        last_updated=now,
                    )
                )
            else:
                # Non-ZONE node — pass through unchanged.
                predicted.append(node)

        return predicted
