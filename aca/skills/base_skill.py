"""
ACA Base Skill
==============

Defines the abstract interface for reusable agricultural competencies.
Skills encapsulate multi-step workflows (e.g., disease detection,
irrigation planning) that compose tool invocations into coherent
operational procedures.

Skills are the "how" of the architecture — Cognition decides *what* to
do, and Skills know *how* to do it using Tools.

Design Decisions:
    - Skills are stateless; all context arrives via parameters.
    - Skills interact with the environment exclusively through the
      ``ToolRegistry`` (never directly with hardware).
    - Each skill declares a schema for discoverability and validation.
    - Skills return structured ``SkillResult`` with success/failure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from aca.tools.registry import ToolRegistry


@dataclass(frozen=True)
class SkillParameter:
    """
    Describes a parameter accepted by a skill.

    Attributes:
        name: Parameter name.
        description: Human-readable description.
        param_type: Expected Python type name.
        required: Whether the parameter must be provided.
        default: Default value if not required.
    """

    name: str
    description: str
    param_type: str = "str"
    required: bool = True
    default: Any = None


@dataclass(frozen=True)
class SkillSchema:
    """
    Declarative schema describing a skill's interface.

    Attributes:
        name: Unique skill identifier.
        description: What the skill accomplishes.
        parameters: Accepted parameters.
        tools_required: Names of tools this skill depends on.
        estimated_duration_seconds: Rough runtime estimate.
    """

    name: str
    description: str
    parameters: List[SkillParameter] = field(default_factory=list)
    tools_required: List[str] = field(default_factory=list)
    estimated_duration_seconds: float = 0.0


@dataclass
class SkillResult:
    """
    Standardized result from a skill execution.

    Attributes:
        success: Whether the skill completed successfully.
        data: Output payload.
        error: Error message if failed.
        tools_invoked: List of tool names that were actually called.
        metadata: Extensible context.
    """

    success: bool
    data: Any = None
    error: str = ""
    tools_invoked: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseSkill(ABC):
    """
    Abstract base class for all ACA skills.

    Subclasses receive a ``ToolRegistry`` reference and must implement
    ``schema`` and ``execute``.

    Example::

        class DiseaseDetection(BaseSkill):
            @property
            def schema(self) -> SkillSchema:
                return SkillSchema(
                    name="disease_detection",
                    tools_required=["vision_model", "sensor_read"],
                )

            def execute(self, tool_registry, **kwargs) -> SkillResult:
                img = tool_registry.invoke("vision_model", ...)
                return SkillResult(success=True, data=img.data)
    """

    @property
    @abstractmethod
    def schema(self) -> SkillSchema:
        """Return the declarative schema for this skill."""
        ...

    @abstractmethod
    def execute(
        self,
        tool_registry: ToolRegistry,
        **kwargs: Any,
    ) -> SkillResult:
        """
        Execute the skill using tools from the registry.

        Args:
            tool_registry: The system's tool registry for interactions.
            **kwargs: Skill-specific parameters.

        Returns:
            A ``SkillResult`` indicating success/failure and payload.
        """
        ...

    @property
    def name(self) -> str:
        """Convenience accessor for the skill's registered name."""
        return self.schema.name

    def validate_params(self, kwargs: Dict[str, Any]) -> List[str]:
        """
        Validate provided parameters against the schema.

        Returns:
            List of validation error strings (empty if valid).
        """
        errors: List[str] = []
        for param in self.schema.parameters:
            if param.required and param.name not in kwargs:
                errors.append(f"Missing required parameter: {param.name}")
        return errors

    def check_tools_available(self, tool_registry: ToolRegistry) -> List[str]:
        """
        Verify that all required tools are registered.

        Returns:
            List of missing tool names (empty if all present).
        """
        available = set(tool_registry.list_tools())
        required = set(self.schema.tools_required)
        return sorted(required - available)
