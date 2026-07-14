"""
ACA Supervisor
==============

Top-level entry point that translates user-provided farm targets into
active missions and initialises the cognitive loop. The Supervisor
monitors overall system health and provides the human-facing interface
for mission management.

This is an interface stub for Milestone 1. Full implementation will
be added in a later milestone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from aca.orchestration.message_bus import MessageBus
from aca.orchestration.schemas import MessageType, MissionPayload, create_message
from aca.orchestration.workflow_engine import WorkflowEngine
from aca.logging_config import get_logger

logger = get_logger("orchestration.supervisor")


class SupervisorInterface(ABC):
    """
    Abstract interface for the ACA Supervisor.

    The Supervisor creates missions and monitors their status.
    Concrete implementation will integrate with the full cognitive
    loop in a later milestone.
    """

    @abstractmethod
    def submit_mission(
        self,
        objective: str,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Submit a new mission.

        Args:
            objective: Natural-language mission objective.
            constraints: Resource or temporal constraints.

        Returns:
            The generated mission ID.
        """
        ...

    @abstractmethod
    def get_mission_status(self, mission_id: str) -> Dict[str, Any]:
        """Return the current status of a mission."""
        ...

    @abstractmethod
    def list_missions(self) -> List[Dict[str, Any]]:
        """List all submitted missions and their statuses."""
        ...


class Supervisor(SupervisorInterface):
    """
    Default Supervisor implementation.

    Creates missions and publishes them to the ``MessageBus`` for
    downstream processing by the Cognition layers.

    Args:
        message_bus: System-wide ``MessageBus``.
        workflow_engine: The ``WorkflowEngine`` for mission orchestration.
    """

    def __init__(
        self,
        message_bus: MessageBus,
        workflow_engine: WorkflowEngine,
    ) -> None:
        self._bus = message_bus
        self._engine = workflow_engine
        self._missions: Dict[str, Dict[str, Any]] = {}
        logger.info("Supervisor initialised")

    def submit_mission(
        self,
        objective: str,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> str:
        import uuid as _uuid

        mission_id = _uuid.uuid4().hex[:16]
        self._missions[mission_id] = {
            "mission_id": mission_id,
            "objective": objective,
            "constraints": constraints or {},
            "status": "SUBMITTED",
        }

        msg = create_message(
            source="supervisor",
            destination="BROADCAST",
            message_type=MessageType.MISSION,
            payload=MissionPayload(
                mission_id=mission_id,
                objective=objective,
                constraints=constraints or {},
            ),
        )
        self._bus.publish(msg)
        logger.info("Mission submitted: %s — %s", mission_id, objective)
        return mission_id

    def get_mission_status(self, mission_id: str) -> Dict[str, Any]:
        return self._missions.get(mission_id, {"error": "not found"})

    def list_missions(self) -> List[Dict[str, Any]]:
        return list(self._missions.values())
