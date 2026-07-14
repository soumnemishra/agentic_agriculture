"""
ACA Skill Registry
==================

Centralised registry for agricultural skill discovery, validation,
and invocation. The Planning Engine composes skills (rather than
issuing primitive tool calls) to accomplish complex tasks.

Design Decisions:
    - Skills registered at startup, immutable thereafter.
    - ``invoke()`` validates parameters and tool availability before
      delegating to the skill's ``execute()`` method.
    - Provides introspection for the planner to discover capabilities.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from aca.logging_config import get_logger
from aca.skills.base_skill import BaseSkill, SkillResult, SkillSchema
from aca.tools.registry import ToolRegistry

logger = get_logger("skills.registry")


class SkillRegistry:
    """
    Registry for ACA skills (reusable agricultural competencies).

    The Planning Engine queries this registry to discover available
    skills, validate their tool dependencies, and invoke them.

    Args:
        tool_registry: The system-wide ``ToolRegistry`` instance
                       (injected so skills can access tools).

    Example::

        skill_reg = SkillRegistry(tool_registry)
        skill_reg.register(DiseaseDetection())
        result = skill_reg.invoke("disease_detection", zone="field_1_a")
    """

    def __init__(self, tool_registry: ToolRegistry) -> None:
        self._tool_registry = tool_registry
        self._skills: Dict[str, BaseSkill] = {}
        self._lock = threading.RLock()
        logger.info("SkillRegistry initialised")

    # ── Registration ──────────────────────────────────────────────────

    def register(self, skill: BaseSkill) -> None:
        """
        Register a skill instance.

        Args:
            skill: A ``BaseSkill`` subclass instance.

        Raises:
            ValueError: If a skill with the same name is already registered.
        """
        with self._lock:
            name = skill.name
            if name in self._skills:
                raise ValueError(f"Skill '{name}' is already registered")
            self._skills[name] = skill
            logger.info("Registered skill: %s", name)

    def unregister(self, name: str) -> bool:
        """Remove a skill by name. Returns True if removed."""
        with self._lock:
            if name in self._skills:
                del self._skills[name]
                return True
            return False

    # ── Invocation ────────────────────────────────────────────────────

    def invoke(self, skill_name: str, **kwargs: Any) -> SkillResult:
        """
        Invoke a registered skill by name.

        Validates parameters and tool availability before execution.

        Args:
            skill_name: Registered skill name.
            **kwargs: Skill-specific parameters.

        Returns:
            A ``SkillResult`` with success/failure and payload.
        """
        skill = self.get(skill_name)
        if skill is None:
            return SkillResult(
                success=False,
                error=f"Skill '{skill_name}' not found in registry",
            )

        # Validate parameters
        param_errors = skill.validate_params(kwargs)
        if param_errors:
            return SkillResult(
                success=False,
                error=f"Parameter validation failed: {'; '.join(param_errors)}",
            )

        # Validate tool availability
        missing = skill.check_tools_available(self._tool_registry)
        if missing:
            return SkillResult(
                success=False,
                error=f"Missing required tools: {', '.join(missing)}",
            )

        try:
            return skill.execute(self._tool_registry, **kwargs)
        except Exception as exc:
            logger.exception("Skill '%s' raised an exception", skill_name)
            return SkillResult(success=False, error=str(exc))

    # ── Discovery ─────────────────────────────────────────────────────

    def get(self, name: str) -> Optional[BaseSkill]:
        """Retrieve a skill instance by name."""
        with self._lock:
            return self._skills.get(name)

    def list_skills(self) -> List[str]:
        """Return names of all registered skills."""
        with self._lock:
            return list(self._skills.keys())

    def get_schemas(self) -> Dict[str, SkillSchema]:
        """Return schemas for all registered skills."""
        with self._lock:
            return {name: s.schema for name, s in self._skills.items()}

    def get_skills_for_tool(self, tool_name: str) -> List[str]:
        """Find all skills that require a given tool."""
        with self._lock:
            return [
                name
                for name, skill in self._skills.items()
                if tool_name in skill.schema.tools_required
            ]

    @property
    def count(self) -> int:
        """Number of registered skills."""
        with self._lock:
            return len(self._skills)
