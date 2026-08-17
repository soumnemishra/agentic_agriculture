from __future__ import annotations

"""
ACA Communication Protocol — Message Schemas
=============================================

Defines the formal message types exchanged across the Agricultural
Cognitive Architecture via the ``MessageBus``.

Every message carries a standard envelope (``ACAMessage``) containing:
    - uuid, timestamp, source, destination
    - confidence, priority
    - typed payload (one of the category dataclasses below)
    - metadata dict for tracing and extensibility

Payload categories map 1-to-1 to the ACA specification:

    Mission · Goal · Task · Observation · Evidence ·
    Hypothesis · Belief · Decision · Explanation · Feedback

Design Decisions:
    - Pure dataclasses with no business logic.
    - Factory helper ``create_message()`` auto-fills uuid / timestamp.
    - ``MessageType`` enum prevents stringly-typed dispatch.
"""

#config.py ──────────────
#Stores constants
#Stores settings
#Stores thresholds

#schemas.py───────────────
#Defines the language of the architecture
#Defines every message
#Defines the communication protocol
#shemas gives is the strucutre of the Data
# the shape that every message should follow 
# schema says what should be the structure of the data 
# It starts with purpose.
# A sensor only tells you what the world is.
# A mission tells you what the world should become.
# That single difference separates a passive monitoring system from an autonomous cognitive architecture.

# UUID IS USED TO GENERATE RANDOM NUMBERS 
import uuid as _uuid

# FROM DATA CLASS IMPORT THE DATA CLASS AND FIELD 
from dataclasses import dataclass, field
from datetime import datetime, timezone
# Enum prevents that by restricting values to a fixed set.
from enum import Enum

from typing import Any, Dict, List, Optional


# ── Message Type Enumeration ──────────────────────────────────────────────
# in python the class is genrally for the blueprint of the object 
# MessageType(Enum): — In Python, a class is usually a blueprint for an object.
# But by putting (Enum) in the parentheses, you are telling Python: 
#"This isn't a normal blueprint. This is a restricted list."
# this is used to avoid "stringly-typed dispatch" 
# this is basically helping to prevent typos 
# this also defines The 10 Steps of AI Thought (The Enum Values)
class MessageType(Enum):
    """Enumeration of all ACA message categories."""

    MISSION = "MISSION"
    GOAL = "GOAL"
    TASK = "TASK"
    OBSERVATION = "OBSERVATION"
    EVIDENCE = "EVIDENCE"
    HYPOTHESIS = "HYPOTHESIS"
    BELIEF = "BELIEF"
    DECISION = "DECISION"
    EXPLANATION = "EXPLANATION"
    FEEDBACK = "FEEDBACK"


# ── Payload Dataclasses ──────────────────────────────────────────────────
#@dataclass blocks. These are the literal 
#"blank paper forms" for the 10 steps above
@dataclass
class MissionPayload:
    # this descides one mission called mission id, objective and constraints 
    """
    Encapsulates a high-level farming mission.

    Attributes:
        mission_id: Unique identifier for this mission.
        objective: Natural-language description of the objective.
        constraints: Resource or temporal limits (e.g. max water litres).
    """
     # mission id is used to distinguish one mission from another mission 
    mission_id: str
    # the objective goal of the mission to be achieved 
    objective: str
    #This is a safety net. If a programmer creates a MissionPayload 
    #but forgets to fill out the constraints box, 
    #instead of crashing, Python will just silently insert an empty dictionary {}.
    # constraints can be any so it contains a dictionary of any type of data.
    constraints: Dict[str, Any] = field(default_factory=dict)

# goal payload exists because it preserve the intent between 
#mission payload and the task payload 
@dataclass
class GoalPayload:
    # 
    # goal payload is a blueprint for the goal message 
    """
    A concrete, measurable goal derived from a mission.

    Attributes:
        goal_id: Unique identifier.
        parent_mission_id: The mission this goal serves.
        target_metric: The observable metric being targeted.
        operator: Comparison operator (e.g. ``GREATER_THAN_OR_EQUAL``).
        value: Target threshold value.
        confidence_threshold: Minimum confidence required to accept.
    """

    goal_id: str
    parent_mission_id: str
    # target metric says what are we  measuring 
    target_metric: str
    # operator says how we compare the measured value to the goals 
    operator: str = "GREATER_THAN_OR_EQUAL"
    value: float = 0.0
    # it tells about how much the confident is 
    # different goal may have different confidence threshold 
    confidence_threshold: float = 0.80


@dataclass
class TaskPayload:
    # task payload is the contract between the planning and Execution 
    """
    A schedulable unit of work assigned to an agent or skill.

    Attributes:
        task_id: Unique identifier.
        goal_id: Parent goal this task supports.
        skill_required: Name of the skill to invoke.
        target_zone: Farm zone identifier.
        parameters: Skill-specific configuration.
    """

    task_id: str
    goal_id: str
    skill_required: str
    target_zone: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
# observe the task and send the obseervation to the next step 
class ObservationPayload:
    # OBSERVATION only writes the observation doesnot conclude anything 
    """
    A timestamped sensor reading or perceptual input.

    Attributes:
        observation_id: Unique identifier.
        source_sensors: List of sensor IDs that contributed.
        target_zone: Farm zone being observed.
        observation_time: When the reading was taken.
        measurements: Key-value sensor measurements.
    """

    observation_id: str
    source_sensors: List[str] = field(default_factory=list)
    target_zone: str = ""
    observation_time: str = ""
    measurements: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvidencePayload:
    """
    Processed evidence derived from one or more observations.

    Attributes:
        evidence_id: Unique identifier.
        source_observation_ids: Originating observations.
        indicator: What the evidence indicates (e.g. ``NDVI_drop``).
        magnitude: Strength of the signal.
        fused_signals: Noise and drift estimations.
    """

    evidence_id: str
    source_observation_ids: List[str] = field(default_factory=list)
    indicator: str = ""
    magnitude: float = 0.0
    fused_signals: Dict[str, float] = field(default_factory=dict)


@dataclass
class HypothesisPayload:
    """
    A candidate explanation for observed evidence.

    Attributes:
        hypothesis_id: Unique identifier.
        associated_evidence_ids: Supporting evidence.
        suspected_cause: The hypothesized root cause.
        prior_probability: Prior belief in this hypothesis.
        likelihood_ratio: How much the evidence updates the prior.
    """

    hypothesis_id: str
    associated_evidence_ids: List[str] = field(default_factory=list)
    suspected_cause: str = ""
    prior_probability: float = 0.0
    likelihood_ratio: float = 1.0


@dataclass
class BeliefPayload:
    """
    A posterior probability distribution over competing hypotheses.

    Attributes:
        belief_distribution: Mapping of hypothesis names to probabilities.
        entropy: Shannon entropy measuring overall uncertainty.
        verdict: System verdict (e.g. ``RESOLVED``, ``UNRESOLVED_UNDER_THRESHOLD``).
    """

    belief_distribution: Dict[str, float] = field(default_factory=dict)
    entropy: float = 0.0
    verdict: str = "UNRESOLVED"


@dataclass
class DecisionPayload:
    """
    A committed action decision with full justification trail.

    Attributes:
        decision_id: Unique identifier.
        justification_ids: Belief / evidence IDs supporting the decision.
        action_selected: High-level action type.
        skill_name: Specific skill to execute.
        parameters: Skill execution parameters.
    """

    decision_id: str
    justification_ids: List[str] = field(default_factory=list)
    action_selected: str = ""
    skill_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExplanationPayload:
    """
    A human-readable explanation of a decision.

    Attributes:
        explanation_id: Unique identifier.
        decision_id: The decision being explained.
        natural_language: Plain-text rationale.
        confidence_trace: Description of the confidence propagation path.
    """

    explanation_id: str
    decision_id: str
    natural_language: str = ""
    confidence_trace: str = ""


@dataclass
class FeedbackPayload:
    """
    Post-execution feedback comparing expected vs actual outcomes.

    Attributes:
        action_id: The executed action being evaluated.
        expected_outcome: What was predicted.
        actual_outcome: What was observed.
        deviation: Numeric difference.
        assessment: Qualitative verdict on plan success.
    """

    action_id: str
    expected_outcome: Dict[str, Any] = field(default_factory=dict)
    actual_outcome: Dict[str, Any] = field(default_factory=dict)
    deviation: float = 0.0
    assessment: str = ""


# ── Payload type union ────────────────────────────────────────────────────

PayloadType = (
    MissionPayload
    | GoalPayload
    | TaskPayload
    | ObservationPayload
    | EvidencePayload
    | HypothesisPayload
    | BeliefPayload
    | DecisionPayload
    | ExplanationPayload
    | FeedbackPayload
)

# Map from MessageType to expected payload class for runtime validation.
PAYLOAD_TYPE_MAP: Dict[MessageType, type] = {
    MessageType.MISSION: MissionPayload,
    MessageType.GOAL: GoalPayload,
    MessageType.TASK: TaskPayload,
    MessageType.OBSERVATION: ObservationPayload,
    MessageType.EVIDENCE: EvidencePayload,
    MessageType.HYPOTHESIS: HypothesisPayload,
    MessageType.BELIEF: BeliefPayload,
    MessageType.DECISION: DecisionPayload,
    MessageType.EXPLANATION: ExplanationPayload,
    MessageType.FEEDBACK: FeedbackPayload,
}


# ── Envelope ──────────────────────────────────────────────────────────────

@dataclass
class ACAMessage:
    """
    Universal message envelope for the ACA communication protocol.

    All inter-component communication flows through instances of this
    class, published and consumed via the ``MessageBus``.

    Attributes:
        uuid: Globally unique message identifier.
        timestamp: ISO-8601 creation timestamp.
        source: Originating agent or subsystem identifier.
        destination: Target agent/subsystem or ``BROADCAST``.
        message_type: Categorizes the payload.
        confidence: Sender's confidence in the payload [0.0, 1.0].
        priority: Urgency level [1 (low) … 5 (critical)].
        payload: The typed payload dataclass.
        metadata: Extensible key-value metadata (trace IDs, tags).
    """

    uuid: str
    timestamp: str
    source: str
    destination: str
    message_type: MessageType
    confidence: float
    priority: int
    payload: PayloadType
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Validate that the payload type matches the declared ``message_type``.

        Raises:
            TypeError: If the payload class does not match the expected type
                       for the declared ``message_type``.
            ValueError: If confidence or priority are out of range.
        """
        expected = PAYLOAD_TYPE_MAP.get(self.message_type)
        if expected and not isinstance(self.payload, expected):
            raise TypeError(
                f"Payload type {type(self.payload).__name__} does not match "
                f"message_type {self.message_type.value} "
                f"(expected {expected.__name__})"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Confidence must be in [0.0, 1.0], got {self.confidence}"
            )
        if not (1 <= self.priority <= 5):
            raise ValueError(
                f"Priority must be in [1, 5], got {self.priority}"
            )


# ── Factory Helper ────────────────────────────────────────────────────────

def create_message(
    source: str,
    destination: str,
    message_type: MessageType,
    payload: PayloadType,
    confidence: float = 1.0,
    priority: int = 3,
    metadata: Optional[Dict[str, Any]] = None,
) -> ACAMessage:
    """
    Convenience factory that auto-generates ``uuid`` and ``timestamp``.

    Args:
        source: Originating component identifier.
        destination: Target component or ``BROADCAST``.
        message_type: The message category.
        payload: A typed payload dataclass instance.
        confidence: Sender confidence [0.0, 1.0].
        priority: Message priority [1 … 5].
        metadata: Optional tracing / extensibility metadata.

    Returns:
        A fully populated ``ACAMessage`` ready for publishing.
    """
    msg = ACAMessage(
        uuid=_uuid.uuid4().hex,
        timestamp=datetime.now(timezone.utc).isoformat(),
        source=source,
        destination=destination,
        message_type=message_type,
        confidence=confidence,
        priority=priority,
        payload=payload,
        metadata=metadata or {},
    )
    msg.validate()
    return msg
