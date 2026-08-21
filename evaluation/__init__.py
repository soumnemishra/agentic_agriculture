"""
ACA Evaluation Module
=====================

Provides experiment simulation and cognitive telemetry logging tools:
    - ``CognitiveMetricsLogger``
    - ``CognitiveCycleRecord``
    - ``run_simulation``
"""

from evaluation.metrics_logger import (
    CognitiveCycleRecord,
    CognitiveMetricsLogger,
)
from evaluation.run_experiment import run_simulation

__all__ = [
    "CognitiveCycleRecord",
    "CognitiveMetricsLogger",
    "run_simulation",
]
