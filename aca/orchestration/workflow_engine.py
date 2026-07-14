"""
ACA Workflow Engine
===================

Responsible for decomposing missions into goal hierarchies and task
graphs, managing workflow lifecycle, and triggering replanning when
execution monitoring detects deviations.

The Workflow Engine separates *orchestration* from *reasoning* — it
does not decide what to do (that is Cognition's role), it coordinates
the sequence and status of tasks that Cognition has planned.

Design Decisions:
    - Workflow states: PENDING → RUNNING → COMPLETED | FAILED | REPLANNING.
    - Task dependency graph modelled as a DAG.
    - Communicates via ``MessageBus`` (publishes TASK, subscribes FEEDBACK).
    - Dependency-injected MessageBus and Scheduler.
"""

from __future__ import annotations

import threading
import uuid as _uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from aca.logging_config import get_logger
from aca.orchestration.message_bus import MessageBus
from aca.orchestration.schemas import (
    ACAMessage,
    FeedbackPayload,
    MessageType,
    TaskPayload,
    create_message,
)

logger = get_logger("orchestration.workflow_engine")


# ── State Enumerations ────────────────────────────────────────────────

class TaskStatus(Enum):
    """Lifecycle status of a single task."""

    PENDING = "PENDING"
    SCHEDULED = "SCHEDULED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class WorkflowStatus(Enum):
    """Lifecycle status of an entire workflow."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REPLANNING = "REPLANNING"


# ── Data Structures ───────────────────────────────────────────────────

@dataclass
class TaskNode:
    """
    A single node in the workflow task graph.

    Attributes:
        task_id: Unique task identifier.
        goal_id: The parent goal this task serves.
        skill_required: Skill to invoke.
        target_zone: Farm zone.
        parameters: Execution parameters.
        depends_on: Set of task IDs that must complete first.
        status: Current lifecycle status.
        result: Execution result (set after completion).
        created_at: ISO-8601 creation timestamp.
    """

    task_id: str
    goal_id: str
    skill_required: str
    target_zone: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    depends_on: Set[str] = field(default_factory=set)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class Workflow:
    """
    A collection of tasks forming a directed acyclic graph (DAG).

    Attributes:
        workflow_id: Unique workflow identifier.
        mission_id: The mission this workflow serves.
        tasks: Mapping of task_id → TaskNode.
        status: Current workflow lifecycle status.
        created_at: ISO-8601 creation timestamp.
    """

    workflow_id: str
    mission_id: str
    tasks: Dict[str, TaskNode] = field(default_factory=dict)
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class WorkflowEngine:
    """
    Manages workflow lifecycle: creation, task dependency resolution,
    status tracking, and replanning triggers.

    Args:
        message_bus: System-wide ``MessageBus`` for task dispatch and
                     feedback reception.

    Example::

        engine = WorkflowEngine(bus)
        wf = engine.create_workflow("mission_001")
        engine.add_task(wf.workflow_id, TaskNode(...))
        ready = engine.get_ready_tasks(wf.workflow_id)
    """

    def __init__(self, message_bus: MessageBus) -> None:
        self._bus = message_bus
        self._workflows: Dict[str, Workflow] = {}
        self._lock = threading.RLock()

        # Subscribe to feedback for task completion tracking
        self._bus.subscribe(MessageType.FEEDBACK, self._on_feedback)
        logger.info("WorkflowEngine initialised")

    # ── Workflow Lifecycle ────────────────────────────────────────────

    def create_workflow(self, mission_id: str) -> Workflow:
        """
        Create a new, empty workflow for a mission.

        Args:
            mission_id: The mission this workflow implements.

        Returns:
            The newly created ``Workflow`` instance.
        """
        wf = Workflow(
            workflow_id=_uuid.uuid4().hex[:16],
            mission_id=mission_id,
        )
        with self._lock:
            self._workflows[wf.workflow_id] = wf
        logger.info(
            "Created workflow %s for mission %s",
            wf.workflow_id,
            mission_id,
        )
        return wf

    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Retrieve a workflow by ID."""
        with self._lock:
            return self._workflows.get(workflow_id)

    def list_workflows(self) -> List[str]:
        """Return all workflow IDs."""
        with self._lock:
            return list(self._workflows.keys())

    # ── Task Management ───────────────────────────────────────────────

    def add_task(self, workflow_id: str, task: TaskNode) -> None:
        """
        Add a task node to a workflow.

        Args:
            workflow_id: Target workflow.
            task: The ``TaskNode`` to add.

        Raises:
            KeyError: If the workflow does not exist.
        """
        with self._lock:
            wf = self._workflows.get(workflow_id)
            if wf is None:
                raise KeyError(f"Workflow '{workflow_id}' not found")
            wf.tasks[task.task_id] = task
            logger.debug(
                "Added task %s to workflow %s", task.task_id, workflow_id
            )

    def get_ready_tasks(self, workflow_id: str) -> List[TaskNode]:
        """
        Return tasks whose dependencies are all COMPLETED and that
        are still PENDING.

        Args:
            workflow_id: Target workflow.

        Returns:
            List of ``TaskNode`` instances ready for scheduling.
        """
        with self._lock:
            wf = self._workflows.get(workflow_id)
            if wf is None:
                return []
            ready = []
            for task in wf.tasks.values():
                if task.status != TaskStatus.PENDING:
                    continue
                deps_met = all(
                    wf.tasks.get(dep, TaskNode(task_id=dep, goal_id="", skill_required="")).status
                    == TaskStatus.COMPLETED
                    for dep in task.depends_on
                )
                if deps_met:
                    ready.append(task)
            return ready

    def update_task_status(
        self,
        workflow_id: str,
        task_id: str,
        status: TaskStatus,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update a task's status.

        If all tasks are COMPLETED, marks the workflow as COMPLETED.
        If any task is FAILED, marks the workflow as FAILED.

        Args:
            workflow_id: Target workflow.
            task_id: Target task within the workflow.
            status: New status.
            result: Optional result data.
        """
        with self._lock:
            wf = self._workflows.get(workflow_id)
            if wf is None:
                return
            task = wf.tasks.get(task_id)
            if task is None:
                return
            task.status = status
            task.result = result

            # Check if workflow status needs updating
            statuses = {t.status for t in wf.tasks.values()}
            if all(s == TaskStatus.COMPLETED for s in statuses):
                wf.status = WorkflowStatus.COMPLETED
                logger.info("Workflow %s COMPLETED", workflow_id)
            elif TaskStatus.FAILED in statuses:
                wf.status = WorkflowStatus.FAILED
                logger.warning("Workflow %s FAILED", workflow_id)
            elif TaskStatus.RUNNING in statuses or TaskStatus.SCHEDULED in statuses:
                wf.status = WorkflowStatus.RUNNING

    def trigger_replan(self, workflow_id: str) -> None:
        """
        Mark a workflow for replanning.

        This pauses execution and signals the Cognition layer to
        re-evaluate the task graph.

        Args:
            workflow_id: Target workflow.
        """
        with self._lock:
            wf = self._workflows.get(workflow_id)
            if wf:
                wf.status = WorkflowStatus.REPLANNING
                logger.info("Workflow %s set to REPLANNING", workflow_id)

    # ── Dispatch ──────────────────────────────────────────────────────

    def dispatch_task(self, workflow_id: str, task: TaskNode) -> None:
        """
        Publish a TASK message to the bus for scheduling.

        Args:
            workflow_id: Owning workflow.
            task: The task to dispatch.
        """
        task.status = TaskStatus.SCHEDULED
        msg = create_message(
            source="workflow_engine",
            destination="scheduler",
            message_type=MessageType.TASK,
            payload=TaskPayload(
                task_id=task.task_id,
                goal_id=task.goal_id,
                skill_required=task.skill_required,
                target_zone=task.target_zone,
                parameters=task.parameters,
            ),
            metadata={"workflow_id": workflow_id},
        )
        self._bus.publish(msg)
        logger.debug("Dispatched task %s", task.task_id)

    # ── Feedback Handler ──────────────────────────────────────────────

    def _on_feedback(self, message: ACAMessage) -> None:
        """Handle FEEDBACK messages to update task statuses."""
        if not isinstance(message.payload, FeedbackPayload):
            return
        payload: FeedbackPayload = message.payload
        wf_id = message.metadata.get("workflow_id", "")
        if not wf_id:
            return

        assessment = payload.assessment.upper()
        if "SUCCESS" in assessment:
            self.update_task_status(
                wf_id, payload.action_id, TaskStatus.COMPLETED
            )
        elif "FAIL" in assessment:
            self.update_task_status(
                wf_id, payload.action_id, TaskStatus.FAILED
            )
