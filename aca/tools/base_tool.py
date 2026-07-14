"""
ACA Base Tool
=============

Defines the abstract interface that all tools in the Agricultural
Cognitive Architecture must implement. Tools are the *only* mechanism
through which agents interact with the external environment (sensors,
actuators, APIs, models).

Design Decisions:
    - Tools are stateless adapters — they translate abstract requests
      into concrete interactions and return structured results.
    - Every tool declares a schema (name, description, parameter spec)
      enabling dynamic discovery and validation by agents.
    - Tools do NOT make decisions; they execute and report.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class ToolParameter:
    """
    Describes a single parameter accepted by a tool.

    Attributes:
        name: Parameter name.
        description: Human-readable description.
        param_type: Expected Python type name (``str``, ``float``, etc.).
        required: Whether the parameter must be provided.
        default: Default value if not required.
    """

    name: str
    description: str
    param_type: str = "str"
    required: bool = True
    default: Any = None


@dataclass(frozen=True)
class ToolSchema:
    """
    Declarative schema describing a tool's interface.

    Attributes:
        name: Unique tool identifier.
        description: What the tool does.
        parameters: List of accepted parameters.
        returns: Description of the return value.
    """

    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    returns: str = ""


@dataclass
class ToolResult:
    """
    Standardized result returned by every tool invocation.

    Attributes:
        success: Whether the invocation succeeded.
        data: The returned payload.
        error: Error message if success is ``False``.
        metadata: Optional additional context.
    """

    success: bool
    data: Any = None
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """
    Abstract base class for all ACA tools.

    Subclasses must implement ``schema`` and ``execute``.

    Example::

        class SensorTool(BaseTool):
            @property
            def schema(self) -> ToolSchema:
                return ToolSchema(name="sensor_read", ...)

            def execute(self, **kwargs) -> ToolResult:
                reading = read_sensor(kwargs["sensor_id"])
                return ToolResult(success=True, data=reading)
    """

    @property
    @abstractmethod
    def schema(self) -> ToolSchema:
        """Return the declarative schema for this tool."""
        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool with the given parameters.

        Args:
            **kwargs: Parameters matching the tool schema.

        Returns:
            A ``ToolResult`` indicating success/failure and payload.
        """
        ...

    @property
    def name(self) -> str:
        """Convenience accessor for the tool's registered name."""
        return self.schema.name

    def validate_params(self, kwargs: Dict[str, Any]) -> List[str]:
        """
        Validate provided parameters against the schema.

        Returns:
            List of validation error strings (empty if valid).
        """
        errors: List[str] = []
        schema = self.schema
        for param in schema.parameters:
            if param.required and param.name not in kwargs:
                errors.append(f"Missing required parameter: {param.name}")
        for key in kwargs:
            known = {p.name for p in schema.parameters}
            if key not in known:
                errors.append(f"Unknown parameter: {key}")
        return errors
