"""
ACA Scheduler
=============

Assigns decomposed tasks to execution runtimes (Edge or Cloud) based
on resource constraints, connectivity, priority, and latency budgets.

The Scheduler does not decide *what* to execute — that is the
Workflow Engine's role. The Scheduler decides *where* and *when*
tasks run.

Design Decisions:
    - Scheduling policy is injectable (strategy pattern).
    - Default policy: prefer edge for low-latency tasks, cloud for
      heavy compute.
    - Maintains a queue per runtime target.
    - Thread-safe.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from aca.config import SchedulerConfig
from aca.logging_config import get_logger
from aca.orchestration.schemas import ACAMessage, MessageType, TaskPayload

logger = get_logger("orchestration.scheduler")


# ── Runtime Target ────────────────────────────────────────────────────

class RuntimeTarget(Enum):
    """Deployment target for task execution."""

    EDGE = "EDGE"
    CLOUD = "CLOUD"


# ── Scheduled Task ────────────────────────────────────────────────────

@dataclass
class ScheduledTask:
    """
    A task that has been assigned to a runtime.

    Attributes:
        task_id: Unique task identifier.
        skill_required: Skill to invoke.
        target_zone: Farm zone.
        parameters: Execution parameters.
        runtime: Assigned runtime target.
        priority: Task priority [1…5].
        scheduled_at: ISO-8601 scheduling timestamp.
        metadata: Workflow-level metadata.
    """

    task_id: str
    skill_required: str
    target_zone: str
    parameters: Dict[str, Any]
    runtime: RuntimeTarget
    priority: int = 3
    scheduled_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Scheduling Policy (Strategy Pattern) ──────────────────────────────

class SchedulingPolicy(ABC):
    """
    Abstract scheduling policy.

    Determines which runtime a task should be assigned to.
    """

    @abstractmethod
    def assign_runtime(
        self,
        task_payload: TaskPayload,
        priority: int,
        config: SchedulerConfig,
    ) -> RuntimeTarget:
        """
        Determine the target runtime for a task.

        Args:
            task_payload: The task details.
            priority: Message priority.
            config: Scheduler configuration.

        Returns:
            The ``RuntimeTarget`` for this task.
        """
        ...


class DefaultSchedulingPolicy(SchedulingPolicy):
    """
    Default policy: prefer Edge for standard tasks, Cloud for
    compute-intensive tasks (identified by skill name heuristics).
    """

    CLOUD_SKILLS = frozenset({
        "yield_estimation",
        "anomaly_investigation",
        "mapping",
    })

    def assign_runtime(
        self,
        task_payload: TaskPayload,
        priority: int,
        config: SchedulerConfig,
    ) -> RuntimeTarget:
        """Assign Edge by default; Cloud for heavy-compute skills."""
        if task_payload.skill_required in self.CLOUD_SKILLS:
            return RuntimeTarget.CLOUD
        if config.prefer_edge:
            return RuntimeTarget.EDGE
        return RuntimeTarget.CLOUD


# ── Scheduler ─────────────────────────────────────────────────────────

class Scheduler:
    """
    Task scheduler that assigns tasks to Edge or Cloud runtimes.

    Args:
        config: Scheduler configuration.
        policy: Scheduling policy (defaults to ``DefaultSchedulingPolicy``).

    Example::

        scheduler = Scheduler(SchedulerConfig())
        scheduled = scheduler.schedule(task_payload, priority=4)
        edge_tasks = scheduler.get_queue(RuntimeTarget.EDGE)
    """

    def __init__(
        self,
        config: SchedulerConfig,
        policy: Optional[SchedulingPolicy] = None,
    ) -> None:
        self._config = config
        self._policy = policy or DefaultSchedulingPolicy()
        self._queues: Dict[RuntimeTarget, List[ScheduledTask]] = defaultdict(list)
        self._lock = threading.RLock()
        logger.info(
            "Scheduler initialised (max_concurrent=%d, prefer_edge=%s)",
            config.max_concurrent_tasks,
            config.prefer_edge,
        )

    # ── Scheduling ────────────────────────────────────────────────────

    def schedule(
        self,
        task_payload: TaskPayload,
        priority: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ScheduledTask:
        """
        Schedule a task by assigning it to a runtime.

        Args:
            task_payload: The task to schedule.
            priority: Task priority.
            metadata: Optional workflow metadata.

        Returns:
            The ``ScheduledTask`` with assigned runtime.
        """
        runtime = self._policy.assign_runtime(
            task_payload, priority, self._config
        )
        scheduled = ScheduledTask(
            task_id=task_payload.task_id,
            skill_required=task_payload.skill_required,
            target_zone=task_payload.target_zone,
            parameters=task_payload.parameters,
            runtime=runtime,
            priority=priority,
            metadata=metadata or {},
        )
        with self._lock:
            self._queues[runtime].append(scheduled)
        logger.info(
            "Scheduled task %s → %s", task_payload.task_id, runtime.value
        )
        return scheduled

    # ── Queue Management ──────────────────────────────────────────────

    def get_queue(self, runtime: RuntimeTarget) -> List[ScheduledTask]:
        """Return all tasks in a runtime's queue (priority-sorted)."""
        with self._lock:
            queue = list(self._queues.get(runtime, []))
        queue.sort(key=lambda t: -t.priority)
        return queue

    def pop_next(self, runtime: RuntimeTarget) -> Optional[ScheduledTask]:
        """
        Pop the highest-priority task from a runtime queue.

        Returns:
            The next ``ScheduledTask``, or ``None`` if empty.
        """
        with self._lock:
            queue = self._queues.get(runtime, [])
            if not queue:
                return None
            queue.sort(key=lambda t: -t.priority)
            return queue.pop(0)

    def remove_task(self, task_id: str) -> bool:
        """Remove a task from all queues. Returns True if found."""
        with self._lock:
            for queue in self._queues.values():
                for i, t in enumerate(queue):
                    if t.task_id == task_id:
                        queue.pop(i)
                        return True
        return False

    # ── Introspection ─────────────────────────────────────────────────

    @property
    def queue_sizes(self) -> Dict[str, int]:
        """Return queue lengths per runtime."""
        with self._lock:
            return {
                rt.value: len(q) for rt, q in self._queues.items()
            }

    @property
    def total_pending(self) -> int:
        """Total number of tasks across all queues."""
        with self._lock:
            return sum(len(q) for q in self._queues.values())
