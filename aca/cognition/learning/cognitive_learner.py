"""
ACA Learning Layer
==================

Closes the cognitive loop by recording experiences, processing
execution feedback, and updating memory and knowledge stores.

Every intervention cycle produces a feedback loop:
    Plan → Execute → Monitor → Feedback → Learn → Update Memory

Components:
    - ``ExperienceRecorder``: Captures full intervention episodes and
      commits them to Episodic Memory.
    - ``MemoryUpdater``: Updates Working Memory with new observations,
      goals, and state changes.
    - ``KnowledgeUpdater``: Refines Semantic Memory rules and thresholds
      based on accumulated experience.
    - ``FeedbackProcessor``: Compares expected vs actual outcomes,
      computes prediction error, and publishes feedback messages.

Design Decisions:
    - Domain-agnostic: no crop models. Learning rules are statistical
      (mean tracking, exponential moving averages).
    - All updates publish events to the MessageBus for tracing.
    - Prediction error drives knowledge refinement.
"""

from __future__ import annotations

import uuid as _uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from aca.logging_config import get_logger
from aca.memory.episodic_memory import Episode, EpisodicMemory
from aca.memory.semantic_memory import SemanticMemory
from aca.memory.working_memory import WorkingMemory
from aca.orchestration.message_bus import MessageBus
from aca.orchestration.schemas import (
    FeedbackPayload,
    MessageType,
    create_message,
)

logger = get_logger("cognition.learning")


# ── Data Structures ───────────────────────────────────────────────────

@dataclass
class PredictionError:
    """
    Quantifies the deviation between expected and actual outcomes.

    Attributes:
        metric: The metric being compared.
        expected: Predicted value.
        actual: Observed value.
        absolute_error: |expected − actual|.
        relative_error: Normalised error (0..1 if possible).
        assessment: Qualitative verdict.
    """

    metric: str
    expected: float
    actual: float
    absolute_error: float = 0.0
    relative_error: float = 0.0
    assessment: str = ""

    def __post_init__(self) -> None:
        self.absolute_error = abs(self.expected - self.actual)
        if self.expected != 0:
            self.relative_error = self.absolute_error / abs(self.expected)
        if self.relative_error < 0.05:
            self.assessment = "EXCELLENT"
        elif self.relative_error < 0.15:
            self.assessment = "ACCEPTABLE"
        elif self.relative_error < 0.30:
            self.assessment = "MARGINAL"
        else:
            self.assessment = "POOR"


@dataclass
class LearningOutcome:
    """
    Summary of a learning cycle.

    Attributes:
        outcome_id: Unique identifier.
        prediction_errors: Errors for each tracked metric.
        memory_updates_applied: Number of memory writes.
        knowledge_updates_applied: Number of knowledge refinements.
        overall_assessment: Aggregate assessment.
    """

    outcome_id: str
    prediction_errors: List[PredictionError] = field(default_factory=list)
    memory_updates_applied: int = 0
    knowledge_updates_applied: int = 0
    overall_assessment: str = ""


# ── ExperienceRecorder ────────────────────────────────────────────────

class ExperienceRecorder:
    """
    Captures full intervention episodes and commits them to
    Episodic Memory.

    Each episode records the initial state, planned actions, executed
    actions, resulting state, and outcome assessment — providing the
    data foundation for retrospective learning.

    Args:
        episodic_memory: Episodic Memory instance.
    """

    def __init__(self, episodic_memory: EpisodicMemory) -> None:
        self._memory = episodic_memory
        logger.info("ExperienceRecorder initialised")

    def record(
        self,
        zone: str,
        initial_state: Dict[str, Any],
        planned_actions: List[Dict[str, Any]],
        executed_actions: List[Dict[str, Any]],
        resulting_state: Dict[str, Any],
        outcome_assessment: str = "",
        yield_impact: float = 0.0,
        tags: Optional[List[str]] = None,
    ) -> Episode:
        """
        Record an intervention episode.

        Args:
            zone: Farm zone where the intervention occurred.
            initial_state: State snapshot before intervention.
            planned_actions: What the planner intended.
            executed_actions: What was actually dispatched.
            resulting_state: State snapshot after execution.
            outcome_assessment: Qualitative verdict.
            yield_impact: Estimated yield impact.
            tags: Searchable metadata tags.

        Returns:
            The committed ``Episode`` instance.
        """
        episode = Episode(
            episode_id=_uuid.uuid4().hex[:12],
            timestamp=datetime.now(timezone.utc).isoformat(),
            zone=zone,
            initial_state=initial_state,
            planned_actions=planned_actions,
            executed_actions=executed_actions,
            resulting_state=resulting_state,
            outcome_assessment=outcome_assessment,
            yield_impact=yield_impact,
            tags=tuple(tags or []),
        )
        self._memory.commit(episode)
        logger.info(
            "Recorded episode %s in zone '%s' (%s)",
            episode.episode_id,
            zone,
            outcome_assessment,
        )
        return episode


# ── MemoryUpdater ─────────────────────────────────────────────────────

class MemoryUpdater:
    """
    Updates Working Memory with new observations, goal states, and
    intermediate results from the cognitive loop.

    Provides a structured API for cognitive components to write to
    Working Memory without direct coupling.

    Args:
        working_memory: Working Memory instance.
    """

    def __init__(self, working_memory: WorkingMemory) -> None:
        self._memory = working_memory
        logger.info("MemoryUpdater initialised")

    def update_observations(self, observations: Dict[str, Any]) -> int:
        """
        Store new observations in the ``observations`` namespace.

        Args:
            observations: ``{observation_id: data}`` mapping.

        Returns:
            Number of observations stored.
        """
        count = 0
        for oid, data in observations.items():
            self._memory.store("observations", oid, data)
            count += 1
        return count

    def update_goals(self, goals: Dict[str, Any]) -> int:
        """
        Store or update goal states in the ``goals`` namespace.

        Args:
            goals: ``{goal_id: goal_data}`` mapping.

        Returns:
            Number of goals stored.
        """
        count = 0
        for gid, data in goals.items():
            self._memory.store("goals", gid, data)
            count += 1
        return count

    def update_beliefs(self, beliefs: Dict[str, float]) -> None:
        """Store the current belief distribution."""
        self._memory.store("beliefs", "current", beliefs)

    def update_reasoning_trace(self, trace_id: str, trace_data: Dict[str, Any]) -> None:
        """Store a reasoning trace reference."""
        self._memory.store("reasoning_traces", trace_id, trace_data)

    def clear_stale(self, namespace: str) -> None:
        """Clear an entire namespace of working memory."""
        self._memory.clear_namespace(namespace)


# ── KnowledgeUpdater ──────────────────────────────────────────────────

class KnowledgeUpdater:
    """
    Refines Semantic Memory rules and thresholds based on
    accumulated experience and prediction errors.

    Uses exponential moving averages to smoothly update stored
    thresholds without overfitting to single events.

    Args:
        semantic_memory: Semantic Memory instance (must not be frozen).
        learning_rate: EMA smoothing factor (0 = no update, 1 = replace).
    """

    def __init__(
        self,
        semantic_memory: SemanticMemory,
        learning_rate: float = 0.1,
    ) -> None:
        self._memory = semantic_memory
        self._lr = learning_rate
        self._updates_applied = 0
        logger.info(
            "KnowledgeUpdater initialised (lr=%.3f)", learning_rate
        )

    def refine_threshold(
        self,
        domain: str,
        key: str,
        observed_value: float,
    ) -> Optional[float]:
        """
        Refine a stored threshold using an EMA update.

        new_value = (1 − lr) × old_value + lr × observed_value

        Args:
            domain: Semantic memory domain.
            key: Threshold key.
            observed_value: New observed value to incorporate.

        Returns:
            The updated threshold value, or ``None`` if the memory
            is frozen or the key does not exist.
        """
        try:
            current = self._memory.retrieve(domain, key)
            if current is None or not isinstance(current, (int, float)):
                return None
            updated = (1 - self._lr) * float(current) + self._lr * observed_value
            self._memory.store(domain, key, updated)
            self._updates_applied += 1
            logger.debug(
                "Refined %s.%s: %.4f → %.4f (observed=%.4f)",
                domain,
                key,
                current,
                updated,
                observed_value,
            )
            return updated
        except RuntimeError:
            logger.warning("Cannot refine threshold: SemanticMemory is frozen")
            return None

    def apply_prediction_errors(
        self,
        domain: str,
        errors: List[PredictionError],
    ) -> int:
        """
        Apply multiple prediction errors as threshold refinements.

        Each error's ``actual`` value is used to refine the
        corresponding key in semantic memory.

        Args:
            domain: Semantic memory domain.
            errors: List of prediction errors.

        Returns:
            Number of successful refinements.
        """
        count = 0
        for err in errors:
            result = self.refine_threshold(domain, err.metric, err.actual)
            if result is not None:
                count += 1
        return count

    @property
    def total_updates(self) -> int:
        """Total number of knowledge refinements applied."""
        return self._updates_applied


# ── FeedbackProcessor ─────────────────────────────────────────────────

class FeedbackProcessor:
    """
    Compares expected vs actual outcomes, computes prediction errors,
    and publishes feedback messages to the ``MessageBus``.

    Orchestrates the full learning cycle: compute errors → record
    experience → update memory → refine knowledge.

    Args:
        message_bus: System-wide ``MessageBus``.
        experience_recorder: Experience recording service.
        memory_updater: Working memory updater.
        knowledge_updater: Semantic memory refinement service.

    Example::

        processor = FeedbackProcessor(bus, recorder, mem_updater, know_updater)
        outcome = processor.process_feedback(
            action_id="a1",
            expected={"soil_moisture": 0.40},
            actual={"soil_moisture": 0.38},
            zone="field_1_a",
        )
    """

    def __init__(
        self,
        message_bus: MessageBus,
        experience_recorder: ExperienceRecorder,
        memory_updater: MemoryUpdater,
        knowledge_updater: KnowledgeUpdater,
    ) -> None:
        self._bus = message_bus
        self._recorder = experience_recorder
        self._mem_updater = memory_updater
        self._know_updater = knowledge_updater
        logger.info("FeedbackProcessor initialised")

    def process_feedback(
        self,
        action_id: str,
        expected: Dict[str, float],
        actual: Dict[str, float],
        zone: str = "",
        initial_state: Optional[Dict[str, Any]] = None,
        planned_actions: Optional[List[Dict[str, Any]]] = None,
        executed_actions: Optional[List[Dict[str, Any]]] = None,
        knowledge_domain: str = "thresholds",
    ) -> LearningOutcome:
        """
        Process execution feedback through the full learning cycle.

        Args:
            action_id: Identifier of the executed action.
            expected: ``{metric: expected_value}`` predictions.
            actual: ``{metric: actual_value}`` observations.
            zone: Farm zone where the intervention occurred.
            initial_state: State before intervention.
            planned_actions: What was planned.
            executed_actions: What was executed.
            knowledge_domain: Semantic memory domain for refinements.

        Returns:
            A ``LearningOutcome`` summarising the cycle.
        """
        outcome = LearningOutcome(outcome_id=_uuid.uuid4().hex[:12])

        # ── Compute prediction errors ────────────────────────────────
        errors: List[PredictionError] = []
        for metric in expected:
            if metric in actual:
                err = PredictionError(
                    metric=metric,
                    expected=expected[metric],
                    actual=actual[metric],
                )
                errors.append(err)
        outcome.prediction_errors = errors

        # ── Compute overall assessment ───────────────────────────────
        if errors:
            avg_rel = sum(e.relative_error for e in errors) / len(errors)
            if avg_rel < 0.05:
                outcome.overall_assessment = "PLAN_SUCCESSFUL"
            elif avg_rel < 0.15:
                outcome.overall_assessment = "PLAN_SUCCESSFUL_WITHIN_BOUNDS"
            elif avg_rel < 0.30:
                outcome.overall_assessment = "PLAN_PARTIAL_SUCCESS"
            else:
                outcome.overall_assessment = "PLAN_FAILED"
        else:
            outcome.overall_assessment = "NO_METRICS_TO_EVALUATE"

        # ── Record experience ────────────────────────────────────────
        self._recorder.record(
            zone=zone,
            initial_state=initial_state or {},
            planned_actions=planned_actions or [],
            executed_actions=executed_actions or [],
            resulting_state=dict(actual),
            outcome_assessment=outcome.overall_assessment,
            tags=["feedback_processed"],
        )

        # ── Update working memory ────────────────────────────────────
        self._mem_updater.update_observations(
            {action_id: {"actual": actual, "expected": expected}}
        )
        outcome.memory_updates_applied = 1

        # ── Refine knowledge ─────────────────────────────────────────
        kn_count = self._know_updater.apply_prediction_errors(
            knowledge_domain, errors
        )
        outcome.knowledge_updates_applied = kn_count

        # ── Publish feedback message ─────────────────────────────────
        primary_metric = errors[0].metric if errors else "unknown"
        fb_msg = create_message(
            source="learning_layer",
            destination="BROADCAST",
            message_type=MessageType.FEEDBACK,
            payload=FeedbackPayload(
                action_id=action_id,
                expected_outcome={
                    "target_metric": primary_metric,
                    "expected_value": expected.get(primary_metric, 0.0),
                },
                actual_outcome={
                    "value": actual.get(primary_metric, 0.0),
                },
                deviation=errors[0].absolute_error if errors else 0.0,
                assessment=outcome.overall_assessment,
            ),
            confidence=1.0 - (
                sum(e.relative_error for e in errors) / len(errors)
                if errors else 0.0
            ),
        )
        self._bus.publish(fb_msg)

        logger.info(
            "Feedback processed for action %s: %s (%d errors, %d updates)",
            action_id,
            outcome.overall_assessment,
            len(errors),
            kn_count,
        )
        return outcome
