"""
ACA Perception Layer
====================

Transforms raw sensor telemetry into validated, normalised feature
objects that downstream cognitive layers can reason over.

Pipeline:  Raw Observation → Validate → Normalise → Extract Features → Publish

Components:
    - ``FeatureObject``: Immutable, typed representation of an extracted feature.
    - ``ObservationValidator``: Checks observations against registered sensor
      schemas (expected ranges, required fields, staleness).
    - ``ObservationNormalizer``: Maps heterogeneous sensor scales to a
      canonical [0, 1] range using configurable min/max bounds.
    - ``ObservationManager``: Orchestrates the full perception pipeline
      and publishes results to the ``MessageBus``.

Design Decisions:
    - Domain-agnostic: no crop/disease knowledge here. Validation rules
      and normalisation bounds are injected as configuration dicts.
    - Every stage operates on ACA message schemas.
    - Confidence degrades when sensor readings are near boundary limits
      or when readings are stale.
"""

from __future__ import annotations

import math
import uuid as _uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from aca.logging_config import get_logger
from aca.orchestration.message_bus import MessageBus
from aca.orchestration.schemas import (
    ACAMessage,
    EvidencePayload,
    MessageType,
    ObservationPayload,
    create_message,
)

logger = get_logger("cognition.perception")


# ── FeatureObject ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class FeatureObject:
    """
    An immutable, normalised feature extracted from raw observations.

    Attributes:
        feature_id: Unique identifier.
        name: Human-readable feature name (e.g. ``soil_moisture``).
        raw_value: Original sensor reading before normalisation.
        normalised_value: Value mapped to [0, 1] canonical range.
        unit: Original measurement unit.
        confidence: Perception confidence in this reading [0, 1].
        source_sensor: Sensor that produced the reading.
        source_zone: Farm zone of origin.
        timestamp: ISO-8601 observation time.
        metadata: Extensible context (calibration date, sensor model).
    """

    feature_id: str
    name: str
    raw_value: float
    normalised_value: float
    unit: str
    confidence: float
    source_sensor: str
    source_zone: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── ObservationValidator ──────────────────────────────────────────────

@dataclass(frozen=True)
class SensorSchema:
    """
    Validation schema for a single sensor type.

    Attributes:
        sensor_type: Type identifier (e.g. ``moisture``, ``temperature``).
        required_fields: Measurement keys that must be present.
        valid_ranges: ``{field: (min, max)}`` acceptable value bounds.
        max_staleness_seconds: Maximum age of a reading before it is
                               considered stale.
    """

    sensor_type: str
    required_fields: List[str] = field(default_factory=list)
    valid_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    max_staleness_seconds: float = 3600.0


@dataclass
class ValidationResult:
    """
    Outcome of validating a single observation.

    Attributes:
        is_valid: Whether the observation passed all checks.
        errors: List of validation error descriptions.
        warnings: List of non-fatal warnings.
        adjusted_confidence: Confidence penalty applied (1.0 = no penalty).
    """

    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    adjusted_confidence: float = 1.0


class ObservationValidator:
    """
    Validates observations against registered sensor schemas.

    Checks:
        - Required measurement fields are present.
        - Values fall within declared valid ranges.
        - Readings are not stale (configurable threshold).

    Args:
        sensor_schemas: ``{sensor_type: SensorSchema}`` mapping.

    Example::

        validator = ObservationValidator({
            "moisture": SensorSchema(
                sensor_type="moisture",
                required_fields=["volumetric_water_content"],
                valid_ranges={"volumetric_water_content": (0.0, 1.0)},
            ),
        })
        result = validator.validate(observation_payload, sensor_type="moisture")
    """

    def __init__(self, sensor_schemas: Optional[Dict[str, SensorSchema]] = None) -> None:
        self._schemas = sensor_schemas or {}
        logger.info(
            "ObservationValidator initialised with %d schemas",
            len(self._schemas),
        )

    def register_schema(self, schema: SensorSchema) -> None:
        """Register or update a sensor validation schema."""
        self._schemas[schema.sensor_type] = schema

    def validate(
        self,
        observation: ObservationPayload,
        sensor_type: str,
        current_time: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate an observation against its sensor schema.

        Args:
            observation: The observation payload to validate.
            sensor_type: Sensor type key to look up the schema.
            current_time: ISO-8601 current time for staleness checks.

        Returns:
            A ``ValidationResult`` with errors, warnings, and adjusted
            confidence.
        """
        result = ValidationResult()
        schema = self._schemas.get(sensor_type)

        if schema is None:
            # No schema registered — pass with a warning
            result.warnings.append(
                f"No schema registered for sensor type '{sensor_type}'"
            )
            result.adjusted_confidence = 0.8
            return result

        # Check required fields
        for req in schema.required_fields:
            if req not in observation.measurements:
                result.is_valid = False
                result.errors.append(f"Missing required field: {req}")

        # Check value ranges
        for field_name, (lo, hi) in schema.valid_ranges.items():
            value = observation.measurements.get(field_name)
            if value is not None:
                if value < lo or value > hi:
                    result.is_valid = False
                    result.errors.append(
                        f"{field_name}={value} outside range [{lo}, {hi}]"
                    )
                else:
                    # Penalise readings near boundaries
                    range_span = hi - lo
                    if range_span > 0:
                        dist_from_center = abs(value - (lo + hi) / 2)
                        boundary_ratio = dist_from_center / (range_span / 2)
                        if boundary_ratio > 0.9:
                            penalty = 0.1 * (boundary_ratio - 0.9) / 0.1
                            result.adjusted_confidence -= penalty * 0.15
                            result.warnings.append(
                                f"{field_name}={value} near boundary"
                            )

        # Check staleness
        if observation.observation_time and current_time:
            try:
                obs_dt = datetime.fromisoformat(observation.observation_time)
                cur_dt = datetime.fromisoformat(current_time)
                age = (cur_dt - obs_dt).total_seconds()
                if age > schema.max_staleness_seconds:
                    result.warnings.append(
                        f"Observation is {age:.0f}s old "
                        f"(max {schema.max_staleness_seconds:.0f}s)"
                    )
                    result.adjusted_confidence -= 0.2
            except (ValueError, TypeError):
                pass

        result.adjusted_confidence = max(0.0, min(1.0, result.adjusted_confidence))
        return result


# ── ObservationNormalizer ─────────────────────────────────────────────

class ObservationNormalizer:
    """
    Maps heterogeneous sensor readings to a canonical [0, 1] range.

    Normalisation bounds are configured per measurement field.

    Args:
        bounds: ``{field_name: (min_physical, max_physical)}``
                defining the expected physical range for each field.

    Example::

        normalizer = ObservationNormalizer({
            "volumetric_water_content": (0.0, 0.60),
            "soil_temp_celsius": (-10.0, 60.0),
        })
        norm = normalizer.normalise("soil_temp_celsius", 25.0)
        # norm ≈ 0.5
    """

    def __init__(self, bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        self._bounds = bounds or {}
        logger.info(
            "ObservationNormalizer initialised with %d bounds",
            len(self._bounds),
        )

    def register_bounds(self, field_name: str, lo: float, hi: float) -> None:
        """Register normalisation bounds for a measurement field."""
        self._bounds[field_name] = (lo, hi)

    def normalise(self, field_name: str, value: float) -> float:
        """
        Normalise a single value to [0, 1].

        If no bounds are registered for the field, returns the value
        clamped to [0, 1].

        Args:
            field_name: The measurement field name.
            value: Raw sensor value.

        Returns:
            Normalised value in [0.0, 1.0].
        """
        bounds = self._bounds.get(field_name)
        if bounds is None:
            return max(0.0, min(1.0, value))
        lo, hi = bounds
        if hi <= lo:
            return 0.5
        normalised = (value - lo) / (hi - lo)
        return max(0.0, min(1.0, normalised))

    def normalise_observation(
        self, measurements: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Normalise all measurements in a dictionary.

        Args:
            measurements: ``{field: raw_value}`` mapping.

        Returns:
            ``{field: normalised_value}`` mapping.
        """
        return {
            name: self.normalise(name, val)
            for name, val in measurements.items()
        }


# ── ObservationManager ────────────────────────────────────────────────

class ObservationManager:
    """
    Orchestrates the perception pipeline: validate → normalise → extract
    features → publish evidence.

    Consumes ``OBSERVATION`` messages from the ``MessageBus`` and
    publishes ``EVIDENCE`` messages containing extracted ``FeatureObject``
    data.

    Args:
        message_bus: System-wide ``MessageBus``.
        validator: Observation validator instance.
        normalizer: Observation normalizer instance.
        sensor_type_map: ``{sensor_id: sensor_type}`` mapping for
                         looking up validation schemas.

    Example::

        manager = ObservationManager(bus, validator, normalizer,
                                      sensor_type_map={"s1": "moisture"})
        manager.start()
        # Now publish OBSERVATION messages to the bus
    """

    def __init__(
        self,
        message_bus: MessageBus,
        validator: ObservationValidator,
        normalizer: ObservationNormalizer,
        sensor_type_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self._bus = message_bus
        self._validator = validator
        self._normalizer = normalizer
        self._sensor_type_map = sensor_type_map or {}
        self._features: List[FeatureObject] = []
        self._active = False
        logger.info("ObservationManager initialised")

    def start(self) -> None:
        """Subscribe to OBSERVATION messages and begin processing."""
        self._bus.subscribe(MessageType.OBSERVATION, self._on_observation)
        self._active = True
        logger.info("ObservationManager started")

    def stop(self) -> None:
        """Stop processing observations."""
        self._active = False

    def process_observation(
        self,
        observation: ObservationPayload,
        source_confidence: float = 1.0,
    ) -> List[FeatureObject]:
        """
        Run the full perception pipeline on a single observation.

        Args:
            observation: Raw observation payload.
            source_confidence: Confidence from the originating message.

        Returns:
            List of extracted ``FeatureObject`` instances.
        """
        features: List[FeatureObject] = []

        # Determine sensor type for each source sensor
        sensor_types = set()
        for sid in observation.source_sensors:
            st = self._sensor_type_map.get(sid, "unknown")
            sensor_types.add(st)

        # Validate
        overall_confidence = source_confidence
        for st in sensor_types:
            vr = self._validator.validate(observation, st)
            if not vr.is_valid:
                logger.warning(
                    "Observation %s failed validation: %s",
                    observation.observation_id,
                    vr.errors,
                )
                return []
            overall_confidence *= vr.adjusted_confidence

        # Normalise and extract features
        normalised = self._normalizer.normalise_observation(
            observation.measurements
        )

        for field_name, raw_value in observation.measurements.items():
            feat = FeatureObject(
                feature_id=_uuid.uuid4().hex[:12],
                name=field_name,
                raw_value=raw_value,
                normalised_value=normalised.get(field_name, raw_value),
                unit="normalised",
                confidence=max(0.0, min(1.0, overall_confidence)),
                source_sensor=(
                    observation.source_sensors[0]
                    if observation.source_sensors
                    else "unknown"
                ),
                source_zone=observation.target_zone,
                timestamp=observation.observation_time
                or datetime.now(timezone.utc).isoformat(),
            )
            features.append(feat)

        self._features.extend(features)
        return features

    def get_recent_features(self, limit: int = 50) -> List[FeatureObject]:
        """Return most recent extracted features."""
        return self._features[-limit:]

    def clear_features(self) -> None:
        """Clear the feature buffer."""
        self._features.clear()

    def _on_observation(self, message: ACAMessage) -> None:
        """MessageBus handler for OBSERVATION messages."""
        if not self._active:
            return
        if not isinstance(message.payload, ObservationPayload):
            return

        features = self.process_observation(
            message.payload, message.confidence
        )

        if features:
            # Publish evidence summarising the extracted features
            evidence_payload = EvidencePayload(
                evidence_id=_uuid.uuid4().hex[:12],
                source_observation_ids=[message.payload.observation_id],
                indicator=",".join(f.name for f in features),
                magnitude=sum(f.normalised_value for f in features) / len(features),
                fused_signals={
                    f.name: f.normalised_value for f in features
                },
            )
            avg_confidence = sum(f.confidence for f in features) / len(features)
            ev_msg = create_message(
                source="perception_layer",
                destination="reasoning_layer",
                message_type=MessageType.EVIDENCE,
                payload=evidence_payload,
                confidence=avg_confidence,
                metadata={
                    "feature_ids": [f.feature_id for f in features],
                    "source_zone": features[0].source_zone,
                },
            )
            self._bus.publish(ev_msg)
