"""
ACA Configuration System
========================

Provides a centralized, immutable configuration object for the entire
Agricultural Cognitive Architecture. All subsystems receive their
configuration via dependency injection from this module.

Design Decisions:
    - Dataclass-based for type safety and IDE discoverability.
    - Frozen to prevent mutation after construction.
    - Factory method ``load()`` supports env-var overrides and file loading.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class MessageBusConfig:
    """Configuration for the publish-subscribe message bus."""

    max_queue_size: int = 10_000
    enable_tracing: bool = True
    default_priority: int = 3


@dataclass(frozen=True)
class MemoryConfig:
    """Configuration for the memory subsystem."""

    working_memory_capacity: int = 500
    episodic_retention_days: int = 365
    semantic_readonly: bool = True
    farm_memory_path: Optional[str] = None


@dataclass(frozen=True)
class SchedulerConfig:
    """Configuration for the task scheduler."""

    max_concurrent_tasks: int = 8
    default_timeout_seconds: float = 300.0
    prefer_edge: bool = True


@dataclass(frozen=True)
class DigitalTwinConfig:
    """Configuration for the Digital Twin crop-simulation engine.

    All rates are expressed *per simulated hour* and represent the
    fractional change applied to the relevant property during each
    discrete time-step.

    Attributes:
        evaporation_rate:           Fraction of ``soil_moisture`` lost
                                    per hour due to evapotranspiration.
        nitrogen_leaching_rate:     Fraction of ``nitrogen_level`` lost
                                    per hour due to leaching / uptake.
        irrigation_moisture_gain:   Absolute ``soil_moisture`` increase
                                    per unit of irrigation ``amount``.
        fertilizer_nitrogen_gain:   Absolute ``nitrogen_level`` increase
                                    per unit of fertiliser ``amount``.
        base_health_recovery:       Per-hour health-index recovery when
                                    moisture and nitrogen are within
                                    optimal ranges.
        moisture_stress_penalty:    Per-hour health-index penalty when
                                    moisture is outside optimal range.
        nitrogen_stress_penalty:    Per-hour health-index penalty when
                                    nitrogen is outside optimal range.
    """

    evaporation_rate: float = 0.02
    nitrogen_leaching_rate: float = 0.01
    irrigation_moisture_gain: float = 0.10
    fertilizer_nitrogen_gain: float = 0.08
    base_health_recovery: float = 0.005
    moisture_stress_penalty: float = 0.01
    nitrogen_stress_penalty: float = 0.008


@dataclass(frozen=True)
class LoggingConfig:
    """Configuration for the logging and tracing subsystem."""

    level: str = "INFO"
    format: str = "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s"
    log_file: Optional[str] = None
    enable_trace_ids: bool = True


@dataclass(frozen=True)
class ACAConfig:
    """
    Root configuration object for the Agricultural Cognitive Architecture.

    All subsystem configurations are nested here. Components receive this
    object (or a relevant sub-config) via constructor injection.

    Example::

        config = ACAConfig.load("config.json")
        bus = MessageBus(config.message_bus)
    """

    message_bus: MessageBusConfig = field(default_factory=MessageBusConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    digital_twin: DigitalTwinConfig = field(default_factory=DigitalTwinConfig)
    environment: str = "development"

    @staticmethod
    def load(path: Optional[str] = None) -> "ACAConfig":
        """
        Load configuration from a JSON file, with environment-variable
        overrides.

        Args:
            path: Optional filesystem path to a JSON config file.

        Returns:
            A fully constructed, frozen ``ACAConfig`` instance.
        """
        raw: Dict[str, Any] = {}
        if path and Path(path).exists():
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)

        env = os.environ.get("ACA_ENV", raw.get("environment", "development"))

        return ACAConfig(
            message_bus=MessageBusConfig(**raw.get("message_bus", {})),
            memory=MemoryConfig(**raw.get("memory", {})),
            scheduler=SchedulerConfig(**raw.get("scheduler", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
            digital_twin=DigitalTwinConfig(**raw.get("digital_twin", {})),
            environment=env,
        )
