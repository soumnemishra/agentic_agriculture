"""
ACA Base Agent — Agent Contracts
================================

Defines the abstract base agent and the formal ``AgentContract``
specification. Every agent in ACA declares a contract that specifies:

    - Purpose, inputs, outputs
    - Internal state shape
    - Memory access permissions (which memory modules it may read/write)
    - Tools it is allowed to invoke
    - Latency budget
    - Failure modes and recovery strategies
    - Confidence output range
    - Messages published and subscribed

The ``BaseAgent`` enforces these contracts at runtime: if an agent
attempts to access a memory module it has not declared, or invokes a
tool outside its allowlist, an ``AgentContractViolation`` is raised.

Design Decisions:
    - Contract-first design — coding new agents is nearly mechanical.
    - All environment interaction goes through the injected
      ``ToolRegistry``; agents never touch hardware directly.
    - Memory access is mediated by a ``MemoryGateway`` that checks
      the agent's declared permissions.
    - Agents communicate exclusively via the ``MessageBus``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from aca.logging_config import get_logger
from aca.orchestration.message_bus import MessageBus
from aca.orchestration.schemas import ACAMessage, MessageType
from aca.tools.registry import ToolRegistry

logger = get_logger("agents.base")


# ── Enumerations ──────────────────────────────────────────────────────

class MemoryAccess(Enum):
    """Allowed access mode for a memory module."""

    READ = "READ"
    WRITE = "WRITE"
    READ_WRITE = "READ_WRITE"


class CognitiveLayer(Enum):
    """The cognitive layer an agent plugs into."""

    PERCEPTION = "PERCEPTION"
    REASONING = "REASONING"
    PLANNING = "PLANNING"
    EXECUTION = "EXECUTION"
    LEARNING = "LEARNING"
    META_COGNITION = "META_COGNITION"


# ── Agent Contract ────────────────────────────────────────────────────

@dataclass(frozen=True)
class AgentContract:
    """
    Formal specification of an agent's capabilities and constraints.

    Attributes:
        agent_name: Unique identifier for this agent.
        purpose: Natural-language description of what the agent does.
        cognitive_layer: The layer this agent plugs into.
        inputs: List of input data descriptors.
        outputs: List of output data descriptors.
        memory_permissions: ``{memory_module: MemoryAccess}``.
        tools_allowed: Set of tool names the agent may invoke.
        latency_budget_ms: Maximum acceptable processing time.
        failure_modes: Known failure scenarios and recovery strategies.
        messages_published: ``MessageType`` values this agent emits.
        messages_subscribed: ``MessageType`` values this agent listens to.
        confidence_range: ``(min, max)`` for output confidence.
    """

    agent_name: str
    purpose: str
    cognitive_layer: CognitiveLayer
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    memory_permissions: Dict[str, MemoryAccess] = field(default_factory=dict)
    tools_allowed: Set[str] = field(default_factory=set)
    latency_budget_ms: float = 1000.0
    failure_modes: Dict[str, str] = field(default_factory=dict)
    messages_published: List[MessageType] = field(default_factory=list)
    messages_subscribed: List[MessageType] = field(default_factory=list)
    confidence_range: tuple = (0.0, 1.0)


# ── Contract Violation ────────────────────────────────────────────────

class AgentContractViolation(Exception):
    """Raised when an agent violates its declared contract."""

    pass


# ── Memory Gateway ────────────────────────────────────────────────────

class MemoryGateway:
    """
    Permission-checked proxy for memory access.

    Wraps the actual memory modules and checks the calling agent's
    ``memory_permissions`` before allowing reads or writes.

    Args:
        memories: ``{module_name: memory_instance}`` mapping.
        permissions: The agent's declared memory permissions.
    """

    def __init__(
        self,
        memories: Dict[str, Any],
        permissions: Dict[str, MemoryAccess],
    ) -> None:
        self._memories = memories
        self._permissions = permissions

    def get_module(self, name: str, mode: MemoryAccess) -> Any:
        """
        Retrieve a memory module if the agent has permission.

        Args:
            name: Memory module name (e.g. ``working``, ``episodic``).
            mode: Required access mode.

        Returns:
            The memory module instance.

        Raises:
            AgentContractViolation: If the agent lacks the required
                                    permission for this module.
        """
        perm = self._permissions.get(name)
        if perm is None:
            raise AgentContractViolation(
                f"Agent has no access to memory module '{name}'"
            )
        if mode == MemoryAccess.WRITE and perm == MemoryAccess.READ:
            raise AgentContractViolation(
                f"Agent has READ-only access to memory module '{name}'"
            )
        if mode == MemoryAccess.READ and perm == MemoryAccess.WRITE:
            raise AgentContractViolation(
                f"Agent has WRITE-only access to memory module '{name}'"
            )
        module = self._memories.get(name)
        if module is None:
            raise AgentContractViolation(
                f"Memory module '{name}' not found in system"
            )
        return module

    @property
    def available_modules(self) -> List[str]:
        """Return list of memory modules the agent can access."""
        return list(self._permissions.keys())


# ── Tool Gateway ──────────────────────────────────────────────────────

class ToolGateway:
    """
    Permission-checked proxy for tool invocation.

    Wraps the ``ToolRegistry`` and ensures the agent only invokes
    tools declared in its contract.

    Args:
        tool_registry: System-wide tool registry.
        allowed_tools: Set of tool names the agent may use.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        allowed_tools: Set[str],
    ) -> None:
        self._registry = tool_registry
        self._allowed = allowed_tools

    def invoke(self, tool_name: str, **kwargs: Any) -> Any:
        """
        Invoke a tool if permitted by the agent contract.

        Raises:
            AgentContractViolation: If the tool is not in the allowlist.
        """
        if tool_name not in self._allowed:
            raise AgentContractViolation(
                f"Agent is not permitted to invoke tool '{tool_name}'. "
                f"Allowed: {sorted(self._allowed)}"
            )
        return self._registry.invoke(tool_name, **kwargs)

    @property
    def available_tools(self) -> List[str]:
        """Return list of tools the agent is allowed to invoke."""
        return sorted(self._allowed)


# ── Base Agent ────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """
    Abstract base class for all ACA agents.

    Enforces the ``AgentContract`` at runtime via gated access to
    memory modules and tools. Agents communicate exclusively through
    the ``MessageBus``.

    Subclasses must implement:
        - ``contract``: Returns the agent's formal contract.
        - ``process(message)``: Handles incoming messages.

    Args:
        message_bus: The system-wide ``MessageBus``.
        memory_gateway: Permission-checked memory proxy.
        tool_gateway: Permission-checked tool proxy.

    Example::

        class SoilAgent(BaseAgent):
            @property
            def contract(self) -> AgentContract:
                return AgentContract(
                    agent_name="soil_agent",
                    purpose="Analyse soil moisture gradients",
                    cognitive_layer=CognitiveLayer.REASONING,
                    memory_permissions={"working": MemoryAccess.READ_WRITE},
                    tools_allowed={"sensor_read"},
                    messages_subscribed=[MessageType.OBSERVATION],
                    messages_published=[MessageType.EVIDENCE],
                )

            def process(self, message: ACAMessage) -> Optional[ACAMessage]:
                data = self.tool_gateway.invoke("sensor_read", ...)
                return self.create_message(MessageType.EVIDENCE, ...)
    """

    def __init__(
        self,
        message_bus: MessageBus,
        memory_gateway: MemoryGateway,
        tool_gateway: ToolGateway,
    ) -> None:
        self._bus = message_bus
        self._memory_gateway = memory_gateway
        self._tool_gateway = tool_gateway
        self._active = False

        # Auto-subscribe to declared message types
        for msg_type in self.contract.messages_subscribed:
            self._bus.subscribe(msg_type, self._on_message)

        logger.info(
            "Agent '%s' initialised (layer=%s)",
            self.contract.agent_name,
            self.contract.cognitive_layer.value,
        )

    # ── Abstract Interface ────────────────────────────────────────────

    @property
    @abstractmethod
    def contract(self) -> AgentContract:
        """Return the agent's formal contract specification."""
        ...

    @abstractmethod
    def process(self, message: ACAMessage) -> Optional[ACAMessage]:
        """
        Process an incoming message.

        Args:
            message: An ``ACAMessage`` matching one of the agent's
                     subscribed types.

        Returns:
            An optional response ``ACAMessage`` to publish, or ``None``.
        """
        ...

    # ── Gateways (protected access for subclasses) ────────────────────

    @property
    def memory(self) -> MemoryGateway:
        """Access to the permission-checked memory gateway."""
        return self._memory_gateway

    @property
    def tool_gateway(self) -> ToolGateway:
        """Access to the permission-checked tool gateway."""
        return self._tool_gateway

    # ── Message Helpers ───────────────────────────────────────────────

    def publish(self, message: ACAMessage) -> int:
        """
        Publish a message, validating it matches the contract.

        Raises:
            AgentContractViolation: If the message type is not in
                                    ``messages_published``.
        """
        if message.message_type not in self.contract.messages_published:
            raise AgentContractViolation(
                f"Agent '{self.contract.agent_name}' is not permitted to "
                f"publish {message.message_type.value}. "
                f"Allowed: {[m.value for m in self.contract.messages_published]}"
            )
        return self._bus.publish(message)

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Mark the agent as active."""
        self._active = True
        logger.info("Agent '%s' started", self.contract.agent_name)

    def stop(self) -> None:
        """Mark the agent as inactive. It will ignore incoming messages."""
        self._active = False
        logger.info("Agent '%s' stopped", self.contract.agent_name)

    @property
    def is_active(self) -> bool:
        """Whether the agent is currently processing messages."""
        return self._active

    # ── Internal Dispatch ─────────────────────────────────────────────

    def _on_message(self, message: ACAMessage) -> None:
        """Internal handler invoked by the MessageBus."""
        if not self._active:
            return
        try:
            response = self.process(message)
            if response is not None:
                self.publish(response)
        except AgentContractViolation:
            raise
        except Exception:
            logger.exception(
                "Agent '%s' error processing %s",
                self.contract.agent_name,
                message.uuid[:8],
            )
