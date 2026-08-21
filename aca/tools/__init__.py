"""
ACA Tools Module
================

Provides the central ToolRegistry and concrete agricultural tools:
    - ``BaseTool``, ``ToolSchema``, ``ToolParameter``, ``ToolResult``
    - ``ToolRegistry``
    - ``AgronomyKnowledgeTool``: Vector search over agricultural knowledge base.
    - ``IrrigationControlTool``: Actuator control for zone irrigation valves.
    - ``TreatmentAlertTool``: Alert dispatch for phytosanitary treatments.
"""

from aca.tools.actuator_tools import (
    IrrigationControlTool,
    TreatmentAlertTool,
)
from aca.tools.agronomy_tool import AgronomyKnowledgeTool
from aca.tools.base_tool import (
    BaseTool,
    ToolParameter,
    ToolResult,
    ToolSchema,
)
from aca.tools.registry import ToolRegistry

__all__ = [
    "BaseTool",
    "ToolParameter",
    "ToolResult",
    "ToolSchema",
    "ToolRegistry",
    "AgronomyKnowledgeTool",
    "IrrigationControlTool",
    "TreatmentAlertTool",
]
