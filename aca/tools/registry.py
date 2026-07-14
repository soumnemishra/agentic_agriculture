"""
ACA Tool Registry
=================

Centralised registry for discovering, validating, and invoking tools.
Agents never instantiate tools directly — they request tools by name
from this registry, which enforces parameter validation before dispatch.

Design Decisions:
    - Singleton-safe (but typically injected via DI).
    - Tools are registered at startup and immutable thereafter.
    - ``invoke()`` validates parameters, executes, and returns a
      ``ToolResult``.
    - Provides introspection APIs for agent discovery.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from aca.logging_config import get_logger
from aca.tools.base_tool import BaseTool, ToolResult, ToolSchema

logger = get_logger("tools.registry")


class ToolRegistry:
    """
    Registry for ACA tools.

    All tool instances are registered here at system bootstrap. Agents
    and skills request tools by name; the registry validates inputs
    and delegates execution.

    Example::

        registry = ToolRegistry()
        registry.register(SensorTool())
        result = registry.invoke("sensor_read", sensor_id="soil_m_01")
    """

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}
        self._lock = threading.RLock()
        logger.info("ToolRegistry initialised")

    # ── Registration ──────────────────────────────────────────────────

    def register(self, tool: BaseTool) -> None:
        """
        Register a tool instance.

        Args:
            tool: A ``BaseTool`` subclass instance.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        with self._lock:
            name = tool.name
            if name in self._tools:
                raise ValueError(f"Tool '{name}' is already registered")
            self._tools[name] = tool
            logger.info("Registered tool: %s", name)

    def unregister(self, name: str) -> bool:
        """
        Remove a tool by name.

        Returns:
            ``True`` if the tool existed and was removed.
        """
        with self._lock:
            if name in self._tools:
                del self._tools[name]
                logger.info("Unregistered tool: %s", name)
                return True
            return False

    # ── Invocation ────────────────────────────────────────────────────

    def invoke(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """
        Invoke a registered tool by name.

        Validates parameters against the tool's schema before execution.

        Args:
            tool_name: Registered tool name.
            **kwargs: Execution parameters.

        Returns:
            A ``ToolResult`` with success/failure and payload.
        """
        tool = self.get(tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found in registry",
            )

        errors = tool.validate_params(kwargs)
        if errors:
            return ToolResult(
                success=False,
                error=f"Parameter validation failed: {'; '.join(errors)}",
            )

        try:
            return tool.execute(**kwargs)
        except Exception as exc:
            logger.exception("Tool '%s' raised an exception", tool_name)
            return ToolResult(success=False, error=str(exc))

    # ── Discovery ─────────────────────────────────────────────────────

    def get(self, name: str) -> Optional[BaseTool]:
        """Retrieve a tool instance by name."""
        with self._lock:
            return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """Return names of all registered tools."""
        with self._lock:
            return list(self._tools.keys())

    def get_schemas(self) -> Dict[str, ToolSchema]:
        """Return schemas for all registered tools."""
        with self._lock:
            return {name: t.schema for name, t in self._tools.items()}

    @property
    def count(self) -> int:
        """Number of registered tools."""
        with self._lock:
            return len(self._tools)
