"""
ACA Agents Module
=================

Provides the base agent abstraction and specialized cognitive agents:
    - ``BaseAgent``, ``AgentContract``, ``CognitiveLayer``, ``MemoryAccess``
    - ``MemoryGateway``, ``ToolGateway``, ``AgentContractViolation``
    - ``PerceptionAgent``: Ingests IoT telemetry & runs PyTorch vision diagnosis.
    - ``ReasoningAgent``: Performs multi-modal sensor fusion via Ollama LLM.
    - ``PlanningAgent``: Formulates physical & phytosanitary action decisions.
    - ``ExecutionAgent``: Dispatches actuator tools & produces closed-loop feedback.
"""

from aca.agents.base_agent import (
    AgentContract,
    AgentContractViolation,
    BaseAgent,
    CognitiveLayer,
    MemoryAccess,
    MemoryGateway,
    ToolGateway,
)
from aca.agents.execution_agent import ExecutionAgent
from aca.agents.perception_agent import PerceptionAgent
from aca.agents.planning_agent import PlanningAgent
from aca.agents.reasoning_agent import ReasoningAgent

__all__ = [
    "BaseAgent",
    "AgentContract",
    "AgentContractViolation",
    "CognitiveLayer",
    "MemoryAccess",
    "MemoryGateway",
    "ToolGateway",
    "PerceptionAgent",
    "ReasoningAgent",
    "PlanningAgent",
    "ExecutionAgent",
]
