"""
ACA Farm Memory
===============

Represents the spatial-temporal topology of the physical farm:
field coordinates, zone boundaries, soil-type polygons, sensor
placements, actuator locations, irrigation infrastructure, and
historical yield records.

Farm Memory is the architecture's persistent geographic and
infrastructural ground truth.

Design Decisions:
    - Entities (zones, sensors, actuators) stored as typed dicts
      keyed by unique IDs.
    - Supports spatial queries (zones by coordinates) and relational
      queries (sensors in a zone).
    - Loadable from JSON registry files.
    - Thread-safe.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, List, Optional, Tuple

from aca.config import MemoryConfig
from aca.logging_config import get_logger

logger = get_logger("memory.farm")


class FarmMemory:
    """
    Persistent spatial-temporal farm topology store.

    Manages zones, sensors, actuators, and historical yield data
    as structured records.

    Args:
        config: Memory configuration.

    Example::

        fm = FarmMemory(MemoryConfig())
        fm.register_zone("field_1_a", {...})
        fm.register_sensor("soil_m_01", zone="field_1_a", ...)
    """

    def __init__(self, config: MemoryConfig) -> None:
        self._zones: Dict[str, Dict[str, Any]] = {}
        self._sensors: Dict[str, Dict[str, Any]] = {}
        self._actuators: Dict[str, Dict[str, Any]] = {}
        self._yield_history: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.RLock()
        logger.info("FarmMemory initialised")

    # ── Zone Management ───────────────────────────────────────────────

    def register_zone(
        self,
        zone_id: str,
        properties: Dict[str, Any],
    ) -> None:
        """
        Register a farm zone.

        Args:
            zone_id: Unique zone identifier.
            properties: Zone metadata (area, soil type, coordinates, etc.).
        """
        with self._lock:
            self._zones[zone_id] = {**properties, "zone_id": zone_id}
            logger.debug("Zone registered: %s", zone_id)

    def get_zone(self, zone_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve zone properties by ID."""
        with self._lock:
            return self._zones.get(zone_id)

    def list_zones(self) -> List[str]:
        """List all registered zone IDs."""
        with self._lock:
            return list(self._zones.keys())

    # ── Sensor Management ─────────────────────────────────────────────

    def register_sensor(
        self,
        sensor_id: str,
        zone_id: str,
        sensor_type: str,
        coordinates: Optional[Tuple[float, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a sensor and associate it with a zone.

        Args:
            sensor_id: Unique sensor identifier.
            zone_id: Zone where the sensor is deployed.
            sensor_type: Type descriptor (e.g. ``moisture``, ``temperature``).
            coordinates: Optional (lat, lon) placement.
            metadata: Additional sensor metadata.
        """
        with self._lock:
            self._sensors[sensor_id] = {
                "sensor_id": sensor_id,
                "zone_id": zone_id,
                "sensor_type": sensor_type,
                "coordinates": coordinates,
                "metadata": metadata or {},
            }

    def get_sensors_in_zone(self, zone_id: str) -> List[Dict[str, Any]]:
        """Return all sensors deployed in a given zone."""
        with self._lock:
            return [
                s for s in self._sensors.values() if s["zone_id"] == zone_id
            ]

    def get_sensor(self, sensor_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve sensor record by ID."""
        with self._lock:
            return self._sensors.get(sensor_id)

    # ── Actuator Management ───────────────────────────────────────────

    def register_actuator(
        self,
        actuator_id: str,
        zone_id: str,
        actuator_type: str,
        coordinates: Optional[Tuple[float, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register an actuator and associate it with a zone.

        Args:
            actuator_id: Unique actuator identifier.
            zone_id: Zone where the actuator operates.
            actuator_type: Type descriptor (e.g. ``irrigation_valve``).
            coordinates: Optional (lat, lon) placement.
            metadata: Additional actuator metadata.
        """
        with self._lock:
            self._actuators[actuator_id] = {
                "actuator_id": actuator_id,
                "zone_id": zone_id,
                "actuator_type": actuator_type,
                "coordinates": coordinates,
                "metadata": metadata or {},
            }

    def get_actuators_in_zone(self, zone_id: str) -> List[Dict[str, Any]]:
        """Return all actuators deployed in a given zone."""
        with self._lock:
            return [
                a for a in self._actuators.values() if a["zone_id"] == zone_id
            ]

    # ── Yield History ─────────────────────────────────────────────────

    def record_yield(
        self,
        zone_id: str,
        season: str,
        yield_kg_per_hectare: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a seasonal yield observation for a zone.

        Args:
            zone_id: Zone identifier.
            season: Season label (e.g. ``2025_kharif``).
            yield_kg_per_hectare: Measured yield.
            metadata: Additional context.
        """
        with self._lock:
            self._yield_history.setdefault(zone_id, []).append({
                "season": season,
                "yield_kg_per_hectare": yield_kg_per_hectare,
                "metadata": metadata or {},
            })

    def get_yield_history(self, zone_id: str) -> List[Dict[str, Any]]:
        """Return all yield records for a zone, oldest first."""
        with self._lock:
            return list(self._yield_history.get(zone_id, []))

    # ── Bulk Loading ──────────────────────────────────────────────────

    def load_from_dict(self, registry: Dict[str, Any]) -> None:
        """
        Bulk-load farm topology from a dictionary.

        Expected structure::

            {
                "zones": {zone_id: {properties}},
                "sensors": {sensor_id: {zone_id, sensor_type, ...}},
                "actuators": {actuator_id: {zone_id, actuator_type, ...}},
            }
        """
        for zone_id, props in registry.get("zones", {}).items():
            self.register_zone(zone_id, props)
        for sensor_id, info in registry.get("sensors", {}).items():
            self.register_sensor(
                sensor_id=sensor_id,
                zone_id=info.get("zone_id", ""),
                sensor_type=info.get("sensor_type", ""),
                coordinates=info.get("coordinates"),
                metadata=info.get("metadata"),
            )
        for actuator_id, info in registry.get("actuators", {}).items():
            self.register_actuator(
                actuator_id=actuator_id,
                zone_id=info.get("zone_id", ""),
                actuator_type=info.get("actuator_type", ""),
                coordinates=info.get("coordinates"),
                metadata=info.get("metadata"),
            )
        logger.info(
            "Loaded farm topology: %d zones, %d sensors, %d actuators",
            len(self._zones),
            len(self._sensors),
            len(self._actuators),
        )

    def load_from_file(self, path: str) -> None:
        """Load farm topology from a JSON file."""
        with open(path, "r", encoding="utf-8") as fh:
            self.load_from_dict(json.load(fh))
