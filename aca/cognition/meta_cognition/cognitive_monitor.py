"""
ACA Meta-Cognition Layer
========================

Implements self-monitoring of the cognitive system: evaluating whether
reasoning is confident, detecting conflicts between beliefs, reflecting
on reasoning quality, and escalating when the system recognises its
own uncertainty.

Meta-Cognition is "thinking about thinking" — it watches the reasoning
pipeline and intervenes when:
    - Confidence is too low to act.
    - Two hypotheses are too close to distinguish.
    - Evidence is contradictory.
    - A plan has failed and needs replanning.

Components:
    - ``ConfidenceMonitor``: Tracks confidence across pipeline stages.
    - ``ConflictDetector``: Identifies competing hypotheses and
      contradictory evidence.
    - ``ReflectionEngine``: Evaluates reasoning quality and suggests
      improvements.
    - ``EscalationManager``: Decides when to request more data or
      escalate to human review.
    - ``ReplanningManager``: Triggers and coordinates replanning cycles.

Design Decisions:
    - Thresholds are configurable (injected, not hardcoded).
    - All assessments are published to the MessageBus for auditability.
    - Domain-agnostic: works on belief distributions and confidence
      values without knowing crop/disease specifics.
"""

from __future__ import annotations

import math
import uuid as _uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from aca.logging_config import get_logger
from aca.orchestration.message_bus import MessageBus
from aca.orchestration.schemas import (
    ExplanationPayload,
    MessageType,
    create_message,
)

logger = get_logger("cognition.meta_cognition")


# ── Enumerations ──────────────────────────────────────────────────────

class EscalationType(Enum):
    """Types of escalation the meta-cognitive layer can trigger."""

    GATHER_MORE_DATA = "GATHER_MORE_DATA"
    REQUEST_HIGH_RES_SCAN = "REQUEST_HIGH_RES_SCAN"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    WAIT_AND_OBSERVE = "WAIT_AND_OBSERVE"
    REPLAN = "REPLAN"


class ConflictSeverity(Enum):
    """Severity levels for detected conflicts."""

    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ── Assessment Data Structures ────────────────────────────────────────

@dataclass
class ConfidenceAssessment:
    """
    Result of a confidence monitoring check.

    Attributes:
        assessment_id: Unique identifier.
        stage_confidences: ``{stage_name: confidence}`` across the pipeline.
        minimum_confidence: Lowest confidence in any stage.
        overall_adequate: Whether confidence is sufficient for action.
        bottleneck_stage: The stage with the lowest confidence.
        recommendations: Suggestions for improving confidence.
    """

    assessment_id: str
    stage_confidences: Dict[str, float] = field(default_factory=dict)
    minimum_confidence: float = 0.0
    overall_adequate: bool = True
    bottleneck_stage: str = ""
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ConflictReport:
    """
    Report of detected conflicts in the reasoning process.

    Attributes:
        report_id: Unique identifier.
        severity: Conflict severity.
        competing_hypotheses: Pairs of hypotheses that are too close.
        probability_gap: Difference between top two hypotheses.
        entropy: Shannon entropy of the belief distribution.
        description: Human-readable conflict description.
    """

    report_id: str
    severity: ConflictSeverity = ConflictSeverity.NONE
    competing_hypotheses: List[Tuple[str, float]] = field(default_factory=list)
    probability_gap: float = 1.0
    entropy: float = 0.0
    description: str = ""


@dataclass
class ReflectionResult:
    """
    Output of a reasoning quality reflection.

    Attributes:
        reflection_id: Unique identifier.
        reasoning_quality_score: [0, 1] score of reasoning quality.
        evidence_sufficiency: Whether enough evidence was gathered.
        belief_stability: Whether beliefs changed dramatically.
        issues: Detected quality issues.
        suggestions: Improvement suggestions.
    """

    reflection_id: str
    reasoning_quality_score: float = 1.0
    evidence_sufficiency: bool = True
    belief_stability: bool = True
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class EscalationDecision:
    """
    Decision from the escalation manager.

    Attributes:
        decision_id: Unique identifier.
        escalation_type: What kind of escalation.
        reason: Why escalation was triggered.
        priority: Urgency [1..5].
        metadata: Additional context.
    """

    decision_id: str
    escalation_type: EscalationType
    reason: str
    priority: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── ConfidenceMonitor ─────────────────────────────────────────────────

class ConfidenceMonitor:
    """
    Monitors confidence values across all reasoning pipeline stages.

    Flags stages where confidence drops below configurable thresholds,
    identifies the bottleneck, and recommends corrective actions.

    Args:
        min_stage_confidence: Minimum acceptable confidence per stage.
        min_overall_confidence: Minimum acceptable combined confidence.
    """

    def __init__(
        self,
        min_stage_confidence: float = 0.40,
        min_overall_confidence: float = 0.50,
    ) -> None:
        self._min_stage = min_stage_confidence
        self._min_overall = min_overall_confidence
        logger.info(
            "ConfidenceMonitor initialised (stage=%.2f, overall=%.2f)",
            min_stage_confidence,
            min_overall_confidence,
        )

    def assess(
        self,
        confidence_propagation: Dict[str, float],
    ) -> ConfidenceAssessment:
        """
        Assess confidence levels across pipeline stages.

        Args:
            confidence_propagation: ``{stage: confidence}`` from the
                                     reasoning trace.

        Returns:
            A ``ConfidenceAssessment`` with diagnostics.
        """
        assessment = ConfidenceAssessment(
            assessment_id=_uuid.uuid4().hex[:12],
            stage_confidences=dict(confidence_propagation),
        )

        if not confidence_propagation:
            assessment.overall_adequate = False
            assessment.recommendations.append("No confidence data available")
            return assessment

        min_conf = min(confidence_propagation.values())
        min_stage = min(
            confidence_propagation, key=confidence_propagation.get  # type: ignore
        )
        assessment.minimum_confidence = min_conf
        assessment.bottleneck_stage = min_stage

        # Check per-stage thresholds
        for stage, conf in confidence_propagation.items():
            if conf < self._min_stage:
                assessment.recommendations.append(
                    f"Stage '{stage}' confidence ({conf:.3f}) below threshold "
                    f"({self._min_stage:.3f})"
                )

        # Check overall
        overall = min_conf
        if overall < self._min_overall:
            assessment.overall_adequate = False
            assessment.recommendations.append(
                f"Overall confidence ({overall:.3f}) below threshold "
                f"({self._min_overall:.3f})"
            )

        return assessment


# ── ConflictDetector ──────────────────────────────────────────────────

class ConflictDetector:
    """
    Detects conflicts in belief distributions where competing
    hypotheses are too close to distinguish.

    Args:
        min_gap: Minimum probability gap between top two hypotheses
                 to consider the belief resolved. Below this → conflict.
        entropy_threshold: Shannon entropy above which beliefs are
                           considered too uncertain.
    """

    def __init__(
        self,
        min_gap: float = 0.15,
        entropy_threshold: float = 1.2,
    ) -> None:
        self._min_gap = min_gap
        self._entropy_threshold = entropy_threshold
        logger.info(
            "ConflictDetector initialised (gap=%.2f, entropy=%.2f)",
            min_gap,
            entropy_threshold,
        )

    def detect(
        self,
        beliefs: Dict[str, float],
        entropy: float,
    ) -> ConflictReport:
        """
        Detect conflicts in a belief distribution.

        Args:
            beliefs: ``{hypothesis: probability}`` distribution.
            entropy: Shannon entropy of the distribution.

        Returns:
            A ``ConflictReport`` describing any detected conflicts.
        """
        report = ConflictReport(
            report_id=_uuid.uuid4().hex[:12],
            entropy=entropy,
        )

        if not beliefs:
            report.severity = ConflictSeverity.NONE
            return report

        # Sort by probability descending
        sorted_beliefs = sorted(
            beliefs.items(), key=lambda x: x[1], reverse=True
        )
        report.competing_hypotheses = sorted_beliefs

        if len(sorted_beliefs) >= 2:
            top_prob = sorted_beliefs[0][1]
            second_prob = sorted_beliefs[1][1]
            gap = top_prob - second_prob
            report.probability_gap = gap

            if gap < self._min_gap:
                if gap < self._min_gap / 3:
                    report.severity = ConflictSeverity.CRITICAL
                    report.description = (
                        f"Near-tie between '{sorted_beliefs[0][0]}' "
                        f"({top_prob:.3f}) and '{sorted_beliefs[1][0]}' "
                        f"({second_prob:.3f})"
                    )
                elif gap < self._min_gap / 2:
                    report.severity = ConflictSeverity.HIGH
                    report.description = (
                        f"Strong conflict between top hypotheses "
                        f"(gap={gap:.3f})"
                    )
                else:
                    report.severity = ConflictSeverity.MEDIUM
                    report.description = (
                        f"Moderate conflict (gap={gap:.3f} < "
                        f"threshold {self._min_gap:.3f})"
                    )
            else:
                report.severity = ConflictSeverity.NONE

        # Entropy check
        if entropy > self._entropy_threshold:
            if report.severity == ConflictSeverity.NONE:
                report.severity = ConflictSeverity.LOW
            report.description += (
                f" | High entropy ({entropy:.3f} > {self._entropy_threshold:.3f})"
            )

        return report


# ── ReflectionEngine ──────────────────────────────────────────────────

class ReflectionEngine:
    """
    Evaluates the quality of a completed reasoning cycle.

    Checks:
        - Was enough evidence collected?
        - Did beliefs change dramatically from priors (instability)?
        - Was confidence adequate at each stage?

    Args:
        min_evidence_count: Minimum number of evidence items expected.
        max_prior_shift: Maximum acceptable shift from prior to posterior.
    """

    def __init__(
        self,
        min_evidence_count: int = 1,
        max_prior_shift: float = 0.8,
    ) -> None:
        self._min_evidence = min_evidence_count
        self._max_shift = max_prior_shift
        logger.info("ReflectionEngine initialised")

    def reflect(
        self,
        hypotheses_count: int,
        evidence_count: int,
        prior_distribution: Dict[str, float],
        posterior_distribution: Dict[str, float],
        confidence_assessment: ConfidenceAssessment,
    ) -> ReflectionResult:
        """
        Reflect on a completed reasoning cycle.

        Args:
            hypotheses_count: Number of hypotheses generated.
            evidence_count: Number of evidence items collected.
            prior_distribution: Prior probabilities.
            posterior_distribution: Posterior probabilities.
            confidence_assessment: Output from ``ConfidenceMonitor``.

        Returns:
            A ``ReflectionResult`` with quality score and suggestions.
        """
        result = ReflectionResult(
            reflection_id=_uuid.uuid4().hex[:12]
        )
        score = 1.0

        # Evidence sufficiency
        if evidence_count < self._min_evidence:
            result.evidence_sufficiency = False
            result.issues.append(
                f"Insufficient evidence: {evidence_count} < {self._min_evidence}"
            )
            result.suggestions.append("Gather additional sensor readings")
            score -= 0.3

        # Belief stability
        if prior_distribution and posterior_distribution:
            max_shift = 0.0
            for name in prior_distribution:
                prior = prior_distribution.get(name, 0.0)
                post = posterior_distribution.get(name, 0.0)
                shift = abs(post - prior)
                max_shift = max(max_shift, shift)

            if max_shift > self._max_shift:
                result.belief_stability = False
                result.issues.append(
                    f"Large belief shift ({max_shift:.3f} > {self._max_shift:.3f})"
                )
                result.suggestions.append(
                    "Consider collecting corroborating evidence"
                )
                score -= 0.2

        # Confidence adequacy
        if not confidence_assessment.overall_adequate:
            result.issues.append(
                f"Low confidence at '{confidence_assessment.bottleneck_stage}'"
            )
            result.suggestions.extend(confidence_assessment.recommendations)
            score -= 0.2

        # Hypothesis coverage
        if hypotheses_count < 2:
            result.issues.append("Fewer than 2 hypotheses considered")
            result.suggestions.append(
                "Expand hypothesis space for more robust reasoning"
            )
            score -= 0.1

        result.reasoning_quality_score = max(0.0, min(1.0, score))
        return result


# ── EscalationManager ────────────────────────────────────────────────

class EscalationManager:
    """
    Decides when to escalate beyond autonomous reasoning.

    Uses conflict reports, confidence assessments, and reflection
    results to determine the appropriate escalation action.

    Args:
        message_bus: System-wide ``MessageBus``.
        auto_escalation_confidence: Below this confidence, auto-escalate.
        human_review_confidence: Below this, require human review.
    """

    def __init__(
        self,
        message_bus: MessageBus,
        auto_escalation_confidence: float = 0.50,
        human_review_confidence: float = 0.30,
    ) -> None:
        self._bus = message_bus
        self._auto_threshold = auto_escalation_confidence
        self._human_threshold = human_review_confidence
        logger.info("EscalationManager initialised")

    def evaluate(
        self,
        confidence_assessment: ConfidenceAssessment,
        conflict_report: ConflictReport,
        reflection: ReflectionResult,
    ) -> Optional[EscalationDecision]:
        """
        Evaluate whether escalation is needed.

        Args:
            confidence_assessment: From ``ConfidenceMonitor``.
            conflict_report: From ``ConflictDetector``.
            reflection: From ``ReflectionEngine``.

        Returns:
            An ``EscalationDecision`` if escalation is needed, else ``None``.
        """
        min_conf = confidence_assessment.minimum_confidence

        # Critical conflict → human review
        if conflict_report.severity == ConflictSeverity.CRITICAL:
            decision = EscalationDecision(
                decision_id=_uuid.uuid4().hex[:12],
                escalation_type=EscalationType.HUMAN_REVIEW,
                reason=f"Critical conflict: {conflict_report.description}",
                priority=5,
            )
            self._publish_escalation(decision)
            return decision

        # Very low confidence → human review
        if min_conf < self._human_threshold:
            decision = EscalationDecision(
                decision_id=_uuid.uuid4().hex[:12],
                escalation_type=EscalationType.HUMAN_REVIEW,
                reason=f"Confidence too low ({min_conf:.3f}) for autonomous action",
                priority=4,
            )
            self._publish_escalation(decision)
            return decision

        # High conflict → gather more data
        if conflict_report.severity in (
            ConflictSeverity.HIGH,
            ConflictSeverity.MEDIUM,
        ):
            decision = EscalationDecision(
                decision_id=_uuid.uuid4().hex[:12],
                escalation_type=EscalationType.REQUEST_HIGH_RES_SCAN,
                reason=f"Conflict detected: {conflict_report.description}",
                priority=3,
            )
            self._publish_escalation(decision)
            return decision

        # Low confidence → gather more data
        if min_conf < self._auto_threshold:
            decision = EscalationDecision(
                decision_id=_uuid.uuid4().hex[:12],
                escalation_type=EscalationType.GATHER_MORE_DATA,
                reason=(
                    f"Confidence ({min_conf:.3f}) below auto threshold "
                    f"({self._auto_threshold:.3f})"
                ),
                priority=2,
            )
            self._publish_escalation(decision)
            return decision

        # Poor reasoning quality → wait and re-observe
        if reflection.reasoning_quality_score < 0.5:
            decision = EscalationDecision(
                decision_id=_uuid.uuid4().hex[:12],
                escalation_type=EscalationType.WAIT_AND_OBSERVE,
                reason=(
                    f"Reasoning quality score ({reflection.reasoning_quality_score:.3f}) "
                    f"below acceptable threshold"
                ),
                priority=2,
            )
            self._publish_escalation(decision)
            return decision

        return None

    def _publish_escalation(self, decision: EscalationDecision) -> None:
        """Publish escalation as an EXPLANATION message."""
        msg = create_message(
            source="meta_cognition",
            destination="BROADCAST",
            message_type=MessageType.EXPLANATION,
            payload=ExplanationPayload(
                explanation_id=decision.decision_id,
                decision_id=decision.decision_id,
                natural_language=f"Escalation: {decision.reason}",
                confidence_trace=f"type={decision.escalation_type.value}",
            ),
            confidence=0.0,
            priority=decision.priority,
            metadata={"escalation_type": decision.escalation_type.value},
        )
        self._bus.publish(msg)
        logger.warning(
            "Escalation triggered: %s — %s",
            decision.escalation_type.value,
            decision.reason,
        )


# ── ReplanningManager ────────────────────────────────────────────────

class ReplanningManager:
    """
    Triggers and coordinates replanning cycles when execution fails
    or when meta-cognitive assessment detects quality issues.

    Tracks replanning history to detect loops and limit retries.

    Args:
        max_replan_attempts: Maximum replanning attempts before
                             escalating to human review.
    """

    def __init__(self, max_replan_attempts: int = 3) -> None:
        self._max_attempts = max_replan_attempts
        self._history: Dict[str, int] = {}
        logger.info(
            "ReplanningManager initialised (max_attempts=%d)",
            max_replan_attempts,
        )

    def should_replan(
        self,
        plan_id: str,
        failure_reason: str,
    ) -> Tuple[bool, str]:
        """
        Determine whether replanning should be attempted.

        Args:
            plan_id: The failed plan's identifier.
            failure_reason: Description of why the plan failed.

        Returns:
            Tuple of ``(should_replan, reason)``.
        """
        count = self._history.get(plan_id, 0) + 1
        self._history[plan_id] = count

        if count > self._max_attempts:
            return False, (
                f"Max replan attempts ({self._max_attempts}) exceeded "
                f"for plan {plan_id}"
            )
        return True, f"Replan attempt {count}/{self._max_attempts}: {failure_reason}"

    def record_replan(self, plan_id: str) -> int:
        """Record a replanning attempt. Returns the attempt number."""
        count = self._history.get(plan_id, 0) + 1
        self._history[plan_id] = count
        return count

    def get_attempt_count(self, plan_id: str) -> int:
        """Return the number of replanning attempts for a plan."""
        return self._history.get(plan_id, 0)

    def reset(self, plan_id: Optional[str] = None) -> None:
        """Reset replanning history for a plan (or all plans)."""
        if plan_id:
            self._history.pop(plan_id, None)
        else:
            self._history.clear()
