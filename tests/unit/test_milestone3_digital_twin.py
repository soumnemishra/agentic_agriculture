"""
Unit Tests for ACA Digital Twin — Milestone 3, Phase 2
=======================================================

Tests cover:
  - Schema validation (ActionType, ProposedAction, SimulationResult)
  - Abstract interface enforcement (ABCMeta contract)
  - DeterministicCropSimulator:
      · Mathematical decay of soil_moisture (exponential evaporation)
      · Mathematical decay of nitrogen_level (exponential leaching)
      · IRRIGATE action correctly boosts moisture
      · FERTILIZE action correctly boosts nitrogen
      · Health-index recovery in optimal range
      · Health-index penalty when out of range
      · Risk flag: "Risk of Root Rot" when moisture > 90%
      · Risk flag: "Risk of Wilting" when moisture < 20%
      · Multi-zone simulations
      · Non-ZONE nodes pass through unchanged
      · Original snapshot is never mutated
      · Edge cases: zero actions, max clamp, multiple actions same node
  - Error handling:
      · hours_ahead < 1
      · Action targeting missing node
  - DigitalTwinConfig injection and custom constants
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Tuple

import pytest

from aca.config import ACAConfig, DigitalTwinConfig
from aca.digital_twin.engine import DeterministicCropSimulator
from aca.digital_twin.interfaces import AbstractDigitalTwin
from aca.digital_twin.schemas import (
    ActionType,
    ProposedAction,
    SimulationResult,
)
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
def dt_config() -> DigitalTwinConfig:
    """Direct handle to the default DigitalTwinConfig."""
    return DigitalTwinConfig()


@pytest.fixture
def simulator(config: ACAConfig) -> DeterministicCropSimulator:
    """Fresh DeterministicCropSimulator with default config."""
    return DeterministicCropSimulator(config)


@pytest.fixture
def single_zone_snapshot() -> GraphSnapshot:
    """Snapshot containing one ZONE node with known initial values."""
    zone = EntityNode.from_dict(
        node_id="zone-A",
        node_type=NodeType.ZONE,
        properties={
            "soil_moisture": 0.50,
            "nitrogen_level": 0.50,
            "health_index": 0.70,
        },
        last_updated="2026-07-14T00:00:00Z",
    )
    return GraphSnapshot(
        nodes=(zone,),
        edges=(),
        captured_at="2026-07-14T00:00:00Z",
    )


@pytest.fixture
def multi_zone_snapshot() -> GraphSnapshot:
    """Snapshot with two ZONE nodes, one SENSOR, and an edge."""
    zone_a = EntityNode.from_dict(
        "zone-A", NodeType.ZONE,
        {"soil_moisture": 0.50, "nitrogen_level": 0.50, "health_index": 0.70},
    )
    zone_b = EntityNode.from_dict(
        "zone-B", NodeType.ZONE,
        {"soil_moisture": 0.85, "nitrogen_level": 0.30, "health_index": 0.60},
    )
    sensor = EntityNode.from_dict(
        "sensor-01", NodeType.SENSOR,
        {"model": "TDR-300"},
    )
    edge = SpatialEdge("zone-A", "sensor-01", "CONTAINS")
    return GraphSnapshot(
        nodes=(zone_a, zone_b, sensor),
        edges=(edge,),
        captured_at="2026-07-14T00:00:00Z",
    )


# ══════════════════════════════════════════════════════════════════════
#  Schema Tests
# ══════════════════════════════════════════════════════════════════════

class TestActionType:
    """Verify ActionType enum members."""

    def test_members(self) -> None:
        assert ActionType.IRRIGATE.value == "IRRIGATE"
        assert ActionType.FERTILIZE.value == "FERTILIZE"

    def test_member_count(self) -> None:
        assert len(ActionType) == 2


class TestProposedAction:
    """Verify ProposedAction frozen dataclass."""

    def test_creation(self) -> None:
        action = ProposedAction(ActionType.IRRIGATE, "zone-A", 2.0)
        assert action.action_type == ActionType.IRRIGATE
        assert action.target_node_id == "zone-A"
        assert action.amount == 2.0

    def test_frozen(self) -> None:
        action = ProposedAction(ActionType.IRRIGATE, "z", 1.0)
        with pytest.raises(AttributeError):
            action.amount = 5.0  # type: ignore[misc]

    def test_negative_amount_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            ProposedAction(ActionType.IRRIGATE, "z", -1.0)

    def test_empty_target_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ProposedAction(ActionType.IRRIGATE, "", 1.0)

    def test_to_dict(self) -> None:
        d = ProposedAction(ActionType.FERTILIZE, "z1", 3.5).to_dict()
        assert d == {
            "action_type": "FERTILIZE",
            "target_node_id": "z1",
            "amount": 3.5,
        }


class TestSimulationResult:
    """Verify SimulationResult frozen dataclass."""

    def test_creation(self) -> None:
        snap = GraphSnapshot()
        result = SimulationResult(
            original_snapshot=snap,
            predicted_snapshot=snap,
            predicted_health_delta=-0.05,
            risk_flags=("Risk of Wilting",),
        )
        assert result.predicted_health_delta == -0.05
        assert result.risk_flags == ("Risk of Wilting",)

    def test_frozen(self) -> None:
        snap = GraphSnapshot()
        result = SimulationResult(snap, snap, 0.0)
        with pytest.raises(AttributeError):
            result.predicted_health_delta = 1.0  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════════════
#  Interface Contract Tests
# ══════════════════════════════════════════════════════════════════════

class TestAbstractDigitalTwin:
    """Ensure ABCMeta prevents incomplete implementations."""

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            AbstractDigitalTwin()  # type: ignore[abstract]

    def test_missing_method_raises(self) -> None:
        class Incomplete(AbstractDigitalTwin):
            pass  # missing simulate_trajectory

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_simulator_is_subclass(self) -> None:
        assert issubclass(DeterministicCropSimulator, AbstractDigitalTwin)


# ══════════════════════════════════════════════════════════════════════
#  Mathematical Decay Tests
# ══════════════════════════════════════════════════════════════════════

class TestMoistureDecay:
    """Verify exponential moisture decay matches the formula:
        moisture_t = moisture_0 × (1 - evaporation_rate)^t
    """

    def test_one_hour_decay(
        self,
        simulator: DeterministicCropSimulator,
        single_zone_snapshot: GraphSnapshot,
        dt_config: DigitalTwinConfig,
    ) -> None:
        result = simulator.simulate_trajectory(
            single_zone_snapshot, [], hours_ahead=1
        )
        predicted_node = self._get_zone(result, "zone-A")
        expected = 0.50 * (1.0 - dt_config.evaporation_rate)
        assert predicted_node["soil_moisture"] == pytest.approx(
            expected, abs=1e-9
        )

    def test_multi_hour_decay(
        self,
        simulator: DeterministicCropSimulator,
        single_zone_snapshot: GraphSnapshot,
        dt_config: DigitalTwinConfig,
    ) -> None:
        hours = 10
        result = simulator.simulate_trajectory(
            single_zone_snapshot, [], hours_ahead=hours
        )
        predicted_node = self._get_zone(result, "zone-A")
        expected = 0.50 * (1.0 - dt_config.evaporation_rate) ** hours
        assert predicted_node["soil_moisture"] == pytest.approx(
            expected, abs=1e-6
        )

    def test_24_hour_decay(
        self,
        simulator: DeterministicCropSimulator,
        single_zone_snapshot: GraphSnapshot,
        dt_config: DigitalTwinConfig,
    ) -> None:
        hours = 24
        result = simulator.simulate_trajectory(
            single_zone_snapshot, [], hours_ahead=hours
        )
        predicted_node = self._get_zone(result, "zone-A")
        expected = 0.50 * (1.0 - dt_config.evaporation_rate) ** hours
        assert predicted_node["soil_moisture"] == pytest.approx(
            expected, abs=1e-6
        )

    @staticmethod
    def _get_zone(result: SimulationResult, zone_id: str) -> Dict:
        for n in result.predicted_snapshot.nodes:
            if n.id == zone_id:
                return n.properties_as_dict()
        raise AssertionError(f"Zone {zone_id} not found in predicted snapshot")


class TestNitrogenDecay:
    """Verify exponential nitrogen decay matches the formula:
        nitrogen_t = nitrogen_0 × (1 - nitrogen_leaching_rate)^t
    """

    def test_one_hour_decay(
        self,
        simulator: DeterministicCropSimulator,
        single_zone_snapshot: GraphSnapshot,
        dt_config: DigitalTwinConfig,
    ) -> None:
        result = simulator.simulate_trajectory(
            single_zone_snapshot, [], hours_ahead=1
        )
        node = _get_zone_props(result, "zone-A")
        expected = 0.50 * (1.0 - dt_config.nitrogen_leaching_rate)
        assert node["nitrogen_level"] == pytest.approx(expected, abs=1e-9)

    def test_multi_hour_decay(
        self,
        simulator: DeterministicCropSimulator,
        single_zone_snapshot: GraphSnapshot,
        dt_config: DigitalTwinConfig,
    ) -> None:
        hours = 48
        result = simulator.simulate_trajectory(
            single_zone_snapshot, [], hours_ahead=hours
        )
        node = _get_zone_props(result, "zone-A")
        expected = 0.50 * (1.0 - dt_config.nitrogen_leaching_rate) ** hours
        assert node["nitrogen_level"] == pytest.approx(expected, abs=1e-6)


# ══════════════════════════════════════════════════════════════════════
#  Action Application Tests
# ══════════════════════════════════════════════════════════════════════

class TestIrrigateAction:
    """Verify IRRIGATE boosts soil_moisture correctly."""

    def test_irrigate_boosts_moisture(
        self,
        simulator: DeterministicCropSimulator,
        single_zone_snapshot: GraphSnapshot,
        dt_config: DigitalTwinConfig,
    ) -> None:
        action = ProposedAction(ActionType.IRRIGATE, "zone-A", 2.0)
        result = simulator.simulate_trajectory(
            single_zone_snapshot, [action], hours_ahead=1
        )
        node = _get_zone_props(result, "zone-A")

        # After action: moisture = 0.50 + 0.10 * 2.0 = 0.70
        # After 1 hour decay: 0.70 * (1 - 0.02) = 0.686
        boosted = 0.50 + dt_config.irrigation_moisture_gain * 2.0
        expected = boosted * (1.0 - dt_config.evaporation_rate)
        assert node["soil_moisture"] == pytest.approx(expected, abs=1e-9)

    def test_irrigate_clamps_at_one(
        self,
        simulator: DeterministicCropSimulator,
        single_zone_snapshot: GraphSnapshot,
    ) -> None:
        """Even a massive irrigation should not push moisture above 1.0."""
        action = ProposedAction(ActionType.IRRIGATE, "zone-A", 100.0)
        result = simulator.simulate_trajectory(
            single_zone_snapshot, [action], hours_ahead=1
        )
        node = _get_zone_props(result, "zone-A")
        assert node["soil_moisture"] <= 1.0


class TestFertilizeAction:
    """Verify FERTILIZE boosts nitrogen_level correctly."""

    def test_fertilize_boosts_nitrogen(
        self,
        simulator: DeterministicCropSimulator,
        single_zone_snapshot: GraphSnapshot,
        dt_config: DigitalTwinConfig,
    ) -> None:
        action = ProposedAction(ActionType.FERTILIZE, "zone-A", 3.0)
        result = simulator.simulate_trajectory(
            single_zone_snapshot, [action], hours_ahead=1
        )
        node = _get_zone_props(result, "zone-A")

        # After action: nitrogen = 0.50 + 0.08 * 3.0 = 0.74
        # After 1 hour decay: 0.74 * (1 - 0.01) = 0.7326
        boosted = 0.50 + dt_config.fertilizer_nitrogen_gain * 3.0
        expected = boosted * (1.0 - dt_config.nitrogen_leaching_rate)
        assert node["nitrogen_level"] == pytest.approx(expected, abs=1e-9)

    def test_fertilize_clamps_at_one(
        self,
        simulator: DeterministicCropSimulator,
        single_zone_snapshot: GraphSnapshot,
    ) -> None:
        action = ProposedAction(ActionType.FERTILIZE, "zone-A", 100.0)
        result = simulator.simulate_trajectory(
            single_zone_snapshot, [action], hours_ahead=1
        )
        node = _get_zone_props(result, "zone-A")
        assert node["nitrogen_level"] <= 1.0


class TestMultipleActionsOnSameNode:
    """Verify multiple actions on the same zone are cumulative."""

    def test_double_irrigate(
        self,
        simulator: DeterministicCropSimulator,
        single_zone_snapshot: GraphSnapshot,
        dt_config: DigitalTwinConfig,
    ) -> None:
        actions = [
            ProposedAction(ActionType.IRRIGATE, "zone-A", 1.0),
            ProposedAction(ActionType.IRRIGATE, "zone-A", 1.0),
        ]
        result = simulator.simulate_trajectory(
            single_zone_snapshot, actions, hours_ahead=1
        )
        node = _get_zone_props(result, "zone-A")

        # 0.50 + 0.10 * 1.0 + 0.10 * 1.0 = 0.70, then decay
        boosted = 0.50 + 2 * dt_config.irrigation_moisture_gain * 1.0
        expected = boosted * (1.0 - dt_config.evaporation_rate)
        assert node["soil_moisture"] == pytest.approx(expected, abs=1e-9)


# ══════════════════════════════════════════════════════════════════════
#  Health Index Tests
# ══════════════════════════════════════════════════════════════════════

class TestHealthIndex:
    """Verify health-index recovery and penalty mechanics."""

    def test_health_recovers_in_optimal_range(
        self,
        simulator: DeterministicCropSimulator,
        dt_config: DigitalTwinConfig,
    ) -> None:
        """When moisture and nitrogen are both in optimal range, health
        should increase."""
        zone = EntityNode.from_dict(
            "zone-A", NodeType.ZONE,
            {"soil_moisture": 0.50, "nitrogen_level": 0.50, "health_index": 0.60},
        )
        snap = GraphSnapshot(nodes=(zone,), captured_at="t0")
        result = simulator.simulate_trajectory(snap, [], hours_ahead=1)
        node = _get_zone_props(result, "zone-A")
        # After 1 hour of recovery the health should exceed initial.
        assert node["health_index"] > 0.60

    def test_health_degrades_with_low_moisture(
        self,
        simulator: DeterministicCropSimulator,
    ) -> None:
        """When moisture is very low, health should decrease."""
        zone = EntityNode.from_dict(
            "zone-A", NodeType.ZONE,
            {"soil_moisture": 0.10, "nitrogen_level": 0.50, "health_index": 0.70},
        )
        snap = GraphSnapshot(nodes=(zone,), captured_at="t0")
        result = simulator.simulate_trajectory(snap, [], hours_ahead=5)
        node = _get_zone_props(result, "zone-A")
        assert node["health_index"] < 0.70

    def test_health_degrades_with_high_moisture(
        self,
        simulator: DeterministicCropSimulator,
    ) -> None:
        """When moisture is very high (>0.90), health should decrease."""
        zone = EntityNode.from_dict(
            "zone-A", NodeType.ZONE,
            {"soil_moisture": 0.95, "nitrogen_level": 0.50, "health_index": 0.70},
        )
        snap = GraphSnapshot(nodes=(zone,), captured_at="t0")
        result = simulator.simulate_trajectory(snap, [], hours_ahead=1)
        node = _get_zone_props(result, "zone-A")
        # At moisture 0.95 (> 0.80 optimal high), penalty applies.
        assert node["health_index"] < 0.70

    def test_health_clamped_at_zero(
        self,
        simulator: DeterministicCropSimulator,
    ) -> None:
        """Health index must not go below 0.0."""
        zone = EntityNode.from_dict(
            "zone-A", NodeType.ZONE,
            {"soil_moisture": 0.01, "nitrogen_level": 0.01, "health_index": 0.01},
        )
        snap = GraphSnapshot(nodes=(zone,), captured_at="t0")
        result = simulator.simulate_trajectory(snap, [], hours_ahead=100)
        node = _get_zone_props(result, "zone-A")
        assert node["health_index"] >= 0.0

    def test_health_clamped_at_one(
        self,
        simulator: DeterministicCropSimulator,
    ) -> None:
        """Health index must not exceed 1.0."""
        zone = EntityNode.from_dict(
            "zone-A", NodeType.ZONE,
            {"soil_moisture": 0.50, "nitrogen_level": 0.50, "health_index": 0.99},
        )
        snap = GraphSnapshot(nodes=(zone,), captured_at="t0")
        # Many recovery steps — health should cap at 1.0.
        result = simulator.simulate_trajectory(snap, [], hours_ahead=100)
        node = _get_zone_props(result, "zone-A")
        assert node["health_index"] <= 1.0


# ══════════════════════════════════════════════════════════════════════
#  Risk Flag Tests
# ══════════════════════════════════════════════════════════════════════

class TestRiskFlags:
    """Verify correct triggering of risk flags."""

    def test_root_rot_flag_when_moisture_exceeds_90(
        self,
        simulator: DeterministicCropSimulator,
    ) -> None:
        zone = EntityNode.from_dict(
            "zone-A", NodeType.ZONE,
            {"soil_moisture": 0.80, "nitrogen_level": 0.50, "health_index": 0.70},
        )
        snap = GraphSnapshot(nodes=(zone,), captured_at="t0")
        # Irrigate enough to push moisture above 0.90.
        action = ProposedAction(ActionType.IRRIGATE, "zone-A", 2.0)
        result = simulator.simulate_trajectory(snap, [action], hours_ahead=1)
        assert "Risk of Root Rot" in result.risk_flags

    def test_wilting_flag_when_moisture_below_20(
        self,
        simulator: DeterministicCropSimulator,
    ) -> None:
        zone = EntityNode.from_dict(
            "zone-A", NodeType.ZONE,
            {"soil_moisture": 0.15, "nitrogen_level": 0.50, "health_index": 0.70},
        )
        snap = GraphSnapshot(nodes=(zone,), captured_at="t0")
        result = simulator.simulate_trajectory(snap, [], hours_ahead=1)
        assert "Risk of Wilting" in result.risk_flags

    def test_no_flags_in_optimal_range(
        self,
        simulator: DeterministicCropSimulator,
        single_zone_snapshot: GraphSnapshot,
    ) -> None:
        """Moisture at 0.50 should never trigger any risk flags."""
        result = simulator.simulate_trajectory(
            single_zone_snapshot, [], hours_ahead=5
        )
        assert "Risk of Root Rot" not in result.risk_flags
        assert "Risk of Wilting" not in result.risk_flags

    def test_wilting_flag_after_prolonged_no_irrigation(
        self,
        simulator: DeterministicCropSimulator,
        single_zone_snapshot: GraphSnapshot,
    ) -> None:
        """After enough hours of evaporation, moisture should drop below
        0.20 and trigger the wilting flag."""
        # 0.50 * (0.98)^n < 0.20  →  n > log(0.40)/log(0.98) ≈ 45.4
        result = simulator.simulate_trajectory(
            single_zone_snapshot, [], hours_ahead=50
        )
        assert "Risk of Wilting" in result.risk_flags


# ══════════════════════════════════════════════════════════════════════
#  Snapshot Immutability Tests
# ══════════════════════════════════════════════════════════════════════

class TestSnapshotImmutability:
    """Ensure the original GraphSnapshot is never mutated."""

    def test_original_snapshot_unchanged(
        self,
        simulator: DeterministicCropSimulator,
        single_zone_snapshot: GraphSnapshot,
    ) -> None:
        original_dict = single_zone_snapshot.to_dict()
        action = ProposedAction(ActionType.IRRIGATE, "zone-A", 5.0)
        result = simulator.simulate_trajectory(
            single_zone_snapshot, [action], hours_ahead=10
        )
        # The original must be byte-for-byte identical.
        assert single_zone_snapshot.to_dict() == original_dict
        # And it must be the same object stored in the result.
        assert result.original_snapshot is single_zone_snapshot

    def test_predicted_is_different_object(
        self,
        simulator: DeterministicCropSimulator,
        single_zone_snapshot: GraphSnapshot,
    ) -> None:
        result = simulator.simulate_trajectory(
            single_zone_snapshot, [], hours_ahead=1
        )
        assert result.predicted_snapshot is not result.original_snapshot


# ══════════════════════════════════════════════════════════════════════
#  Multi-Zone & Non-ZONE Node Tests
# ══════════════════════════════════════════════════════════════════════

class TestMultiZoneSimulation:
    """Verify correct behaviour with multiple zones."""

    def test_actions_target_correct_zone(
        self,
        simulator: DeterministicCropSimulator,
        multi_zone_snapshot: GraphSnapshot,
        dt_config: DigitalTwinConfig,
    ) -> None:
        """Irrigating zone-A should not affect zone-B."""
        action = ProposedAction(ActionType.IRRIGATE, "zone-A", 2.0)
        result = simulator.simulate_trajectory(
            multi_zone_snapshot, [action], hours_ahead=1
        )
        node_a = _get_zone_props(result, "zone-A")
        node_b = _get_zone_props(result, "zone-B")

        # zone-A: boosted then decayed.
        boosted_a = 0.50 + dt_config.irrigation_moisture_gain * 2.0
        expected_a = boosted_a * (1.0 - dt_config.evaporation_rate)
        assert node_a["soil_moisture"] == pytest.approx(expected_a, abs=1e-9)

        # zone-B: only decayed (no action applied).
        expected_b = 0.85 * (1.0 - dt_config.evaporation_rate)
        assert node_b["soil_moisture"] == pytest.approx(expected_b, abs=1e-9)

    def test_non_zone_nodes_unchanged(
        self,
        simulator: DeterministicCropSimulator,
        multi_zone_snapshot: GraphSnapshot,
    ) -> None:
        result = simulator.simulate_trajectory(
            multi_zone_snapshot, [], hours_ahead=5
        )
        for node in result.predicted_snapshot.nodes:
            if node.id == "sensor-01":
                assert node.type == NodeType.SENSOR
                assert node.properties_as_dict()["model"] == "TDR-300"
                return
        pytest.fail("sensor-01 not found in predicted snapshot")

    def test_edges_preserved(
        self,
        simulator: DeterministicCropSimulator,
        multi_zone_snapshot: GraphSnapshot,
    ) -> None:
        result = simulator.simulate_trajectory(
            multi_zone_snapshot, [], hours_ahead=1
        )
        assert result.predicted_snapshot.edges == multi_zone_snapshot.edges


# ══════════════════════════════════════════════════════════════════════
#  Health Delta Tests
# ══════════════════════════════════════════════════════════════════════

class TestHealthDelta:
    """Verify predicted_health_delta is computed correctly."""

    def test_positive_delta_on_recovery(
        self,
        simulator: DeterministicCropSimulator,
    ) -> None:
        """In optimal conditions health recovers → positive delta."""
        zone = EntityNode.from_dict(
            "zone-A", NodeType.ZONE,
            {"soil_moisture": 0.50, "nitrogen_level": 0.50, "health_index": 0.50},
        )
        snap = GraphSnapshot(nodes=(zone,), captured_at="t0")
        result = simulator.simulate_trajectory(snap, [], hours_ahead=5)
        assert result.predicted_health_delta > 0.0

    def test_negative_delta_on_stress(
        self,
        simulator: DeterministicCropSimulator,
    ) -> None:
        """Under stress, health degrades → negative delta."""
        zone = EntityNode.from_dict(
            "zone-A", NodeType.ZONE,
            {"soil_moisture": 0.05, "nitrogen_level": 0.05, "health_index": 0.50},
        )
        snap = GraphSnapshot(nodes=(zone,), captured_at="t0")
        result = simulator.simulate_trajectory(snap, [], hours_ahead=10)
        assert result.predicted_health_delta < 0.0


# ══════════════════════════════════════════════════════════════════════
#  Error Handling Tests
# ══════════════════════════════════════════════════════════════════════

class TestErrorHandling:
    """Verify proper error handling for invalid inputs."""

    def test_hours_ahead_zero_raises(
        self,
        simulator: DeterministicCropSimulator,
        single_zone_snapshot: GraphSnapshot,
    ) -> None:
        with pytest.raises(ValueError, match="hours_ahead"):
            simulator.simulate_trajectory(single_zone_snapshot, [], 0)

    def test_hours_ahead_negative_raises(
        self,
        simulator: DeterministicCropSimulator,
        single_zone_snapshot: GraphSnapshot,
    ) -> None:
        with pytest.raises(ValueError, match="hours_ahead"):
            simulator.simulate_trajectory(single_zone_snapshot, [], -5)

    def test_action_on_missing_node_raises(
        self,
        simulator: DeterministicCropSimulator,
        single_zone_snapshot: GraphSnapshot,
    ) -> None:
        action = ProposedAction(ActionType.IRRIGATE, "nonexistent", 1.0)
        with pytest.raises(ValueError, match="does not exist"):
            simulator.simulate_trajectory(
                single_zone_snapshot, [action], 1
            )

    def test_zero_actions_is_valid(
        self,
        simulator: DeterministicCropSimulator,
        single_zone_snapshot: GraphSnapshot,
    ) -> None:
        """Empty action list should produce a valid result (decay only)."""
        result = simulator.simulate_trajectory(
            single_zone_snapshot, [], hours_ahead=1
        )
        assert isinstance(result, SimulationResult)


# ══════════════════════════════════════════════════════════════════════
#  Custom Config Tests
# ══════════════════════════════════════════════════════════════════════

class TestCustomConfig:
    """Verify that injected physical constants are respected."""

    def test_high_evaporation_rate(self) -> None:
        """Doubling evaporation should produce faster moisture loss."""
        fast_config = ACAConfig(
            digital_twin=DigitalTwinConfig(evaporation_rate=0.10)
        )
        slow_config = ACAConfig(
            digital_twin=DigitalTwinConfig(evaporation_rate=0.01)
        )
        fast_sim = DeterministicCropSimulator(fast_config)
        slow_sim = DeterministicCropSimulator(slow_config)

        zone = EntityNode.from_dict(
            "zone-A", NodeType.ZONE,
            {"soil_moisture": 0.80, "nitrogen_level": 0.50, "health_index": 0.70},
        )
        snap = GraphSnapshot(nodes=(zone,), captured_at="t0")

        fast_result = fast_sim.simulate_trajectory(snap, [], 10)
        slow_result = slow_sim.simulate_trajectory(snap, [], 10)

        fast_moisture = _get_zone_props(fast_result, "zone-A")["soil_moisture"]
        slow_moisture = _get_zone_props(slow_result, "zone-A")["soil_moisture"]

        assert fast_moisture < slow_moisture

    def test_custom_irrigation_gain(self) -> None:
        """A higher irrigation_moisture_gain should produce more moisture."""
        high_gain = ACAConfig(
            digital_twin=DigitalTwinConfig(irrigation_moisture_gain=0.50)
        )
        low_gain = ACAConfig(
            digital_twin=DigitalTwinConfig(irrigation_moisture_gain=0.01)
        )
        sim_high = DeterministicCropSimulator(high_gain)
        sim_low = DeterministicCropSimulator(low_gain)

        zone = EntityNode.from_dict(
            "zone-A", NodeType.ZONE,
            {"soil_moisture": 0.30, "nitrogen_level": 0.50, "health_index": 0.70},
        )
        snap = GraphSnapshot(nodes=(zone,), captured_at="t0")
        action = ProposedAction(ActionType.IRRIGATE, "zone-A", 1.0)

        high_result = sim_high.simulate_trajectory(snap, [action], 1)
        low_result = sim_low.simulate_trajectory(snap, [action], 1)

        high_m = _get_zone_props(high_result, "zone-A")["soil_moisture"]
        low_m = _get_zone_props(low_result, "zone-A")["soil_moisture"]

        assert high_m > low_m


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════

def _get_zone_props(
    result: SimulationResult, zone_id: str
) -> Dict:
    """Extract the properties dict for a zone from a SimulationResult."""
    for node in result.predicted_snapshot.nodes:
        if node.id == zone_id:
            return node.properties_as_dict()
    raise AssertionError(f"Zone '{zone_id}' not found in predicted snapshot")
