"""
ACA Simulation Module
=====================

Provides deterministic simulators and digital twin mock environments:
    - ``IoTStreamer``: Real-world IoT microclimate telemetry streamer.
"""

from simulation.telemetry_streamer import IoTStreamer, TelemetryRecord

__all__ = ["IoTStreamer", "TelemetryRecord"]
