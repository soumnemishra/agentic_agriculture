"""
ACA Planning Layer
==================

Translates decisions and goals from the Reasoning Layer into executable
task graphs composed of skills, then publishes them via the MessageBus
for scheduling and execution.

Components:
    - ``GoalPlanner``: Decomposes missions into prioritised, measurable goals.
    - ``TaskPlanner``: Breaks goals into ordered task sequences with dependencies.
    - ``SkillSelector``: Matches tasks to available skills from the SkillRegistry.
    - ``ExecutionPlanner``: Assembles a complete execution plan with timing,
      resource estimates, and rollback strategies.

Design Decisions:
    - Domain-agnostic: goal decomposition rules and skill mappings are
      injected, not hardcoded.
    - Every plan carries confidence propagated from the reasoning trace.
    - Plans publish TASK messages to the bus for the Scheduler.
    - The ExecutionPlanner can be extended with Digital Twin rollouts
      in a later milestone.
"""

from __future__ import annotations

import uuid as _uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from aca.logging_config import get_logger
from aca.orchestration.message_bus import MessageBus
from aca.orchestration.schemas import (
    GoalPayload,
    MessageType,
    TaskPayload,
    create_message,
)
from aca.skills.registry import SkillRegistry

logger = get_logger("cognition.planning")


# ── Data Structures ───────────────────────────────────────────────────

@dataclass
class Goal:
    """
    A concrete, measurable objective derived from a mission or decision.

    Attributes:
        goal_id: Unique identifier.
        mission_id: Parent mission.
        description: Human-readable goal statement.
        target_metric: Observable metric name.
        target_value: Desired metric value.
        operator: Comparison operator.
        priority: Goal priority [1..5].
        confidence: Confidence inherited from reasoning.
        status: Current lifecycle status.
    """

    goal_id: str
    mission_id: str
    description: str
    target_metric: str
    target_value: float
    operator: str = "GREATER_THAN_OR_EQUAL"
    priority: int = 3
    confidence: float = 1.0
    status: str = "PENDING"


@dataclass
class PlannedTask:
    """
    A single task within an execution plan.

    Attributes:
        task_id: Unique identifier.
        goal_id: Parent goal.
        skill_name: Skill to invoke.
        target_zone: Farm zone.
        parameters: Skill parameters.
        depends_on: Task IDs that must complete first.
        estimated_duration_s: Estimated execution time.
        confidence: Confidence inherited from the goal.
    """

    task_id: str
    goal_id: str
    skill_name: str
    target_zone: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    estimated_duration_s: float = 60.0
    confidence: float = 1.0


@dataclass
class ExecutionPlan:
    """
    A complete execution plan containing goals, tasks, and metadata.

    Attributes:
        plan_id: Unique identifier.
        mission_id: Parent mission.
        goals: Goals this plan addresses.
        tasks: Ordered task sequence.
        total_estimated_duration_s: Sum of task durations.
        overall_confidence: Minimum confidence across tasks.
        created_at: ISO-8601 creation timestamp.
        reasoning_trace_id: ID of the reasoning trace that produced this plan.
        rollback_strategy: What to do if execution fails.
    """

    plan_id: str
    mission_id: str
    goals: List[Goal] = field(default_factory=list)
    tasks: List[PlannedTask] = field(default_factory=list)
    total_estimated_duration_s: float = 0.0
    overall_confidence: float = 1.0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    reasoning_trace_id: str = ""
    rollback_strategy: str = "HALT_AND_REPORT"


# ── GoalPlanner ───────────────────────────────────────────────────────

# Injectable decomposition: (mission_objective, decision_action) → list of goals
GoalDecomposerFn = Callable[[str, str, float], List[Goal]]


def _default_goal_decomposer(
    mission_id: str, action: str, confidence: float
) -> List[Goal]:
    """Default: create one goal per action with inherited confidence."""
    return [
        Goal(
            goal_id=_uuid.uuid4().hex[:12],
            mission_id=mission_id,
            description=f"Execute action: {action}",
            target_metric="action_completion",
            target_value=1.0,
            confidence=confidence,
        )
    ]


class GoalPlanner:
    """
    Decomposes missions and decisions into prioritised goals.

    The decomposition logic is injectable via ``decomposer_fn``,
    allowing domain-specific goal hierarchies without hardcoding.

    Args:
        decomposer_fn: Callable that takes ``(mission_id, action, confidence)``
                       and returns a list of ``Goal`` instances.

    Example::

        planner = GoalPlanner()
        goals = planner.decompose("m1", "irrigate", confidence=0.85)
    """

    def __init__(
        self,
        decomposer_fn: Optional[GoalDecomposerFn] = None,
    ) -> None:
        self._fn = decomposer_fn or _default_goal_decomposer
        logger.info("GoalPlanner initialised")

    def decompose(
        self,
        mission_id: str,
        action: str,
        confidence: float = 1.0,
    ) -> List[Goal]:
        """
        Decompose an action into goals.

        Args:
            mission_id: Parent mission identifier.
            action: The action string from the reasoning decision.
            confidence: Confidence from the reasoning trace.

        Returns:
            List of ``Goal`` instances.
        """
        goals = self._fn(mission_id, action, confidence)
        logger.debug(
            "Decomposed action '%s' into %d goals", action, len(goals)
        )
        return goals


# ── TaskPlanner ───────────────────────────────────────────────────────

class TaskPlanner:
    """
    Breaks goals into ordered task sequences.

    For each goal, produces one or more ``PlannedTask`` instances.
    Task dependencies are set sequentially by default; custom
    dependency logic can be injected.

    Args:
        default_zone: Default farm zone if none specified.
    """

    def __init__(self, default_zone: str = "") -> None:
        self._default_zone = default_zone
        logger.info("TaskPlanner initialised")

    def plan_tasks(
        self,
        goal: Goal,
        skill_name: str,
        target_zone: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        subtask_count: int = 1,
    ) -> List[PlannedTask]:
        """
        Generate a task sequence for a goal.

        Args:
            goal: The goal to plan for.
            skill_name: Primary skill to use.
            target_zone: Farm zone target.
            parameters: Skill parameters.
            subtask_count: Number of subtasks to create (for multi-step skills).

        Returns:
            Ordered list of ``PlannedTask`` instances.
        """
        zone = target_zone or self._default_zone
        tasks: List[PlannedTask] = []
        prev_id: Optional[str] = None

        for i in range(subtask_count):
            tid = _uuid.uuid4().hex[:12]
            task = PlannedTask(
                task_id=tid,
                goal_id=goal.goal_id,
                skill_name=skill_name,
                target_zone=zone,
                parameters=parameters or {},
                depends_on=[prev_id] if prev_id else [],
                confidence=goal.confidence,
            )
            tasks.append(task)
            prev_id = tid

        logger.debug(
            "Planned %d tasks for goal %s", len(tasks), goal.goal_id
        )
        return tasks


# ── SkillSelector ─────────────────────────────────────────────────────

class SkillSelector:
    """
    Selects the most appropriate skill for a given task.

    Queries the ``SkillRegistry`` to find available skills, optionally
    filtering by name matching or custom scoring.

    Args:
        skill_registry: The system-wide ``SkillRegistry``.
        preference_map: ``{action_keyword: preferred_skill_name}``
                        for mapping actions to preferred skills.
    """

    def __init__(
        self,
        skill_registry: SkillRegistry,
        preference_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self._registry = skill_registry
        self._preference = preference_map or {}
        logger.info("SkillSelector initialised")

    def select(self, action: str, required_tools: Optional[List[str]] = None) -> Optional[str]:
        """
        Select a skill for the given action.

        Args:
            action: Action label from the decision.
            required_tools: Optional list of tools the skill must have.

        Returns:
            Skill name, or ``None`` if no match found.
        """
        # Check preference map first
        if action in self._preference:
            pref = self._preference[action]
            if self._registry.get(pref) is not None:
                return pref

        # Fallback: find by substring match in skill names
        available = self._registry.list_skills()
        for skill_name in available:
            if action.lower() in skill_name.lower():
                return skill_name

        # Return first available skill as last resort
        return available[0] if available else None

    def list_available(self) -> List[str]:
        """Return all available skill names."""
        return self._registry.list_skills()


# ── ExecutionPlanner ──────────────────────────────────────────────────

class ExecutionPlanner:
    """
    Assembles a complete ``ExecutionPlan`` from goals and tasks,
    and publishes task messages to the ``MessageBus``.

    The ExecutionPlanner is the integration point: it takes reasoning
    decisions, decomposes them into goals and tasks using the
    ``GoalPlanner``, ``TaskPlanner``, and ``SkillSelector``, then
    publishes the plan for scheduling.

    Args:
        message_bus: System-wide ``MessageBus``.
        goal_planner: Goal decomposition engine.
        task_planner: Task planning engine.
        skill_selector: Skill matching engine.

    Example::

        planner = ExecutionPlanner(bus, goal_planner, task_planner, selector)
        plan = planner.create_plan("m1", "irrigate", "field_1", confidence=0.8)
        planner.publish_plan(plan)
    """

    def __init__(
        self,
        message_bus: MessageBus,
        goal_planner: GoalPlanner,
        task_planner: TaskPlanner,
        skill_selector: SkillSelector,
    ) -> None:
        self._bus = message_bus
        self._goal_planner = goal_planner
        self._task_planner = task_planner
        self._skill_selector = skill_selector
        logger.info("ExecutionPlanner initialised")

    def create_plan(
        self,
        mission_id: str,
        action: str,
        target_zone: str = "",
        confidence: float = 1.0,
        reasoning_trace_id: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> ExecutionPlan:
        """
        Create a full execution plan for an action.

        Args:
            mission_id: Parent mission identifier.
            action: Action from the reasoning decision.
            target_zone: Target farm zone.
            confidence: Confidence from the reasoning trace.
            reasoning_trace_id: ID of the originating reasoning trace.
            parameters: Additional execution parameters.

        Returns:
            A complete ``ExecutionPlan``.
        """
        # Decompose into goals
        goals = self._goal_planner.decompose(mission_id, action, confidence)

        # Select skill
        skill_name = self._skill_selector.select(action) or "generic_intervention"

        # Plan tasks for each goal
        all_tasks: List[PlannedTask] = []
        for goal in goals:
            tasks = self._task_planner.plan_tasks(
                goal, skill_name, target_zone, parameters
            )
            all_tasks.extend(tasks)

        # Assemble plan
        total_duration = sum(t.estimated_duration_s for t in all_tasks)
        min_confidence = min(t.confidence for t in all_tasks) if all_tasks else 0.0

        plan = ExecutionPlan(
            plan_id=_uuid.uuid4().hex[:12],
            mission_id=mission_id,
            goals=goals,
            tasks=all_tasks,
            total_estimated_duration_s=total_duration,
            overall_confidence=min_confidence,
            reasoning_trace_id=reasoning_trace_id,
        )

        logger.info(
            "Created plan %s: %d goals, %d tasks, confidence=%.3f",
            plan.plan_id,
            len(goals),
            len(all_tasks),
            min_confidence,
        )
        return plan

    def publish_plan(self, plan: ExecutionPlan) -> int:
        """
        Publish all tasks in a plan to the MessageBus.

        Args:
            plan: The execution plan to publish.

        Returns:
            Number of task messages published.
        """
        published = 0
        for task in plan.tasks:
            msg = create_message(
                source="planning_layer",
                destination="scheduler",
                message_type=MessageType.TASK,
                payload=TaskPayload(
                    task_id=task.task_id,
                    goal_id=task.goal_id,
                    skill_required=task.skill_name,
                    target_zone=task.target_zone,
                    parameters=task.parameters,
                ),
                confidence=task.confidence,
                metadata={
                    "plan_id": plan.plan_id,
                    "reasoning_trace_id": plan.reasoning_trace_id,
                },
            )
            self._bus.publish(msg)
            published += 1

        logger.info(
            "Published %d tasks from plan %s", published, plan.plan_id
        )
        return published
