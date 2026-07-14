"""
ACA Reasoning Layer
===================

Implements the multi-stage reasoning pipeline that transforms evidence
into justified, confidence-weighted decisions with full reasoning traces.

Pipeline:
    Evidence → Hypothesis Generation → Evidence Collection →
    Evidence Fusion → Belief Management → Decision Selection

Every stage propagates confidence values, and every decision carries
a complete ``ReasoningTrace`` linking it back through beliefs, fused
evidence, hypotheses, and raw observations.

Components:
    - ``HypothesisGenerator``: Produces candidate explanations from evidence.
    - ``EvidenceCollector``: Gathers supporting/refuting evidence from memory.
    - ``EvidenceFusionEngine``: Fuses evidence via Bayesian-style updates.
    - ``BeliefManager``: Maintains and queries posterior belief distributions.
    - ``DecisionCandidate``: A scored, justified candidate action.
    - ``ReasoningTrace``: Full provenance chain for a decision.
    - ``ReasoningPipeline``: Orchestrates the end-to-end reasoning cycle.

Design Decisions:
    - Domain-agnostic: no crop/disease models embedded. Hypothesis
      priors and likelihood functions are injectable.
    - Bayesian evidence fusion uses configurable likelihood ratios.
    - Shannon entropy measures uncertainty for meta-cognition hooks.
    - All stages communicate via ACA message schemas on the MessageBus.
"""

from __future__ import annotations

import math
import uuid as _uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from aca.logging_config import get_logger
from aca.orchestration.message_bus import MessageBus
from aca.orchestration.schemas import (
    ACAMessage,
    BeliefPayload,
    DecisionPayload,
    EvidencePayload,
    ExplanationPayload,
    HypothesisPayload,
    MessageType,
    create_message,
)

logger = get_logger("cognition.reasoning")


# ── Data Structures ───────────────────────────────────────────────────

@dataclass
class Hypothesis:
    """
    A candidate explanation for observed phenomena.

    Attributes:
        hypothesis_id: Unique identifier.
        name: Short label (e.g. ``under_watering``, ``pathogen_stress``).
        prior: Prior probability [0, 1] before evidence.
        posterior: Updated probability after evidence fusion.
        supporting_evidence_ids: IDs of evidence supporting this hypothesis.
        refuting_evidence_ids: IDs of evidence refuting this hypothesis.
        metadata: Extensible tags.
    """

    hypothesis_id: str
    name: str
    prior: float
    posterior: float = 0.0
    supporting_evidence_ids: List[str] = field(default_factory=list)
    refuting_evidence_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceItem:
    """
    A piece of evidence associated with one or more hypotheses.

    Attributes:
        evidence_id: Unique identifier.
        indicator: What this evidence indicates.
        magnitude: Signal strength [0, 1].
        confidence: How much we trust this evidence [0, 1].
        likelihood_ratios: ``{hypothesis_name: ratio}`` — how much
            this evidence updates each hypothesis. >1 supports, <1 refutes.
        source_observation_ids: Originating observation IDs.
    """

    evidence_id: str
    indicator: str
    magnitude: float
    confidence: float
    likelihood_ratios: Dict[str, float] = field(default_factory=dict)
    source_observation_ids: List[str] = field(default_factory=list)


@dataclass
class DecisionCandidate:
    """
    A candidate action with a scored justification.

    Attributes:
        decision_id: Unique identifier.
        action: Proposed action label.
        skill_name: Skill to invoke.
        score: Combined score [0, 1] reflecting urgency and confidence.
        confidence: Confidence in this being the right action.
        justification: List of hypothesis names supporting this action.
        parameters: Execution parameters.
        reasoning_trace: Full provenance chain.
    """

    decision_id: str
    action: str
    skill_name: str
    score: float
    confidence: float
    justification: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    reasoning_trace: Optional["ReasoningTrace"] = None


@dataclass
class ReasoningTrace:
    """
    Complete provenance chain for a decision.

    Links the decision back through each reasoning stage, making
    the process fully explainable.

    Attributes:
        trace_id: Unique trace identifier.
        timestamp: When the reasoning cycle completed.
        hypotheses_generated: Hypotheses considered.
        evidence_collected: Evidence items used.
        belief_distribution: Final posterior distribution.
        entropy: Shannon entropy of the belief distribution.
        selected_decision: The chosen ``DecisionCandidate``.
        confidence_propagation: ``{stage: confidence}`` at each step.
        metadata: Additional context.
    """

    trace_id: str
    timestamp: str
    hypotheses_generated: List[Hypothesis] = field(default_factory=list)
    evidence_collected: List[EvidenceItem] = field(default_factory=list)
    belief_distribution: Dict[str, float] = field(default_factory=dict)
    entropy: float = 0.0
    selected_decision: Optional[DecisionCandidate] = None
    confidence_propagation: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── HypothesisGenerator ──────────────────────────────────────────────

# Type alias for injectable hypothesis generation functions.
# Receives evidence indicators and returns list of (name, prior) tuples.
HypothesisGeneratorFn = Callable[[List[str]], List[Tuple[str, float]]]


def _default_hypothesis_fn(indicators: List[str]) -> List[Tuple[str, float]]:
    """
    Default hypothesis generator: returns a generic set of competing
    hypotheses with uniform priors. Override with domain knowledge.
    """
    hypotheses = [
        ("environmental_stress", 0.25),
        ("resource_deficit", 0.25),
        ("biological_threat", 0.25),
        ("sensor_anomaly", 0.25),
    ]
    return hypotheses


class HypothesisGenerator:
    """
    Generates candidate hypotheses from observed evidence indicators.

    The generation function is injectable, allowing domain-specific
    hypothesis sets without embedding domain knowledge here.

    Args:
        generator_fn: A callable that takes a list of indicator strings
                      and returns ``[(name, prior), ...]``.

    Example::

        gen = HypothesisGenerator(my_domain_fn)
        hypotheses = gen.generate(["NDVI_drop", "moisture_low"])
    """

    def __init__(
        self,
        generator_fn: Optional[HypothesisGeneratorFn] = None,
    ) -> None:
        self._fn = generator_fn or _default_hypothesis_fn
        logger.info("HypothesisGenerator initialised")

    def generate(self, indicators: List[str]) -> List[Hypothesis]:
        """
        Generate hypotheses for the given evidence indicators.

        Normalises priors to sum to 1.0.

        Args:
            indicators: List of evidence indicator strings.

        Returns:
            List of ``Hypothesis`` instances with normalised priors.
        """
        raw = self._fn(indicators)
        total = sum(p for _, p in raw)
        if total <= 0:
            total = 1.0

        hypotheses = []
        for name, prior in raw:
            h = Hypothesis(
                hypothesis_id=_uuid.uuid4().hex[:12],
                name=name,
                prior=prior / total,
                posterior=prior / total,
            )
            hypotheses.append(h)
        logger.debug(
            "Generated %d hypotheses from %d indicators",
            len(hypotheses),
            len(indicators),
        )
        return hypotheses


# ── EvidenceCollector ─────────────────────────────────────────────────

class EvidenceCollector:
    """
    Collects and organises evidence from current observations and
    optionally from episodic memory.

    This collector converts ``EvidencePayload`` messages into internal
    ``EvidenceItem`` objects, enriching them with configurable
    likelihood ratios.

    Args:
        default_likelihood: Default likelihood ratio for unknown
                            indicator/hypothesis pairs.
        likelihood_table: ``{indicator: {hypothesis: ratio}}`` mapping
                          configuring how each indicator updates each
                          hypothesis. Injectable for domain customisation.
    """

    def __init__(
        self,
        default_likelihood: float = 1.0,
        likelihood_table: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self._default = default_likelihood
        self._table = likelihood_table or {}
        self._collected: List[EvidenceItem] = []
        logger.info("EvidenceCollector initialised")

    def register_likelihood(
        self,
        indicator: str,
        hypothesis: str,
        ratio: float,
    ) -> None:
        """Register a likelihood ratio for an indicator-hypothesis pair."""
        self._table.setdefault(indicator, {})[hypothesis] = ratio

    def collect_from_payload(
        self,
        payload: EvidencePayload,
        message_confidence: float = 1.0,
    ) -> EvidenceItem:
        """
        Convert an ``EvidencePayload`` into an ``EvidenceItem``.

        Assigns likelihood ratios based on the registered table.

        Args:
            payload: Evidence payload from the perception layer.
            message_confidence: Confidence from the carrying message.

        Returns:
            An ``EvidenceItem`` with assigned likelihood ratios.
        """
        indicators = [s.strip() for s in payload.indicator.split(",") if s.strip()]
        lr: Dict[str, float] = {}
        for ind in indicators:
            ind_ratios = self._table.get(ind, {})
            for hyp, ratio in ind_ratios.items():
                lr[hyp] = lr.get(hyp, 1.0) * ratio

        item = EvidenceItem(
            evidence_id=payload.evidence_id,
            indicator=payload.indicator,
            magnitude=payload.magnitude,
            confidence=message_confidence,
            likelihood_ratios=lr if lr else {
                k: self._default for k in []
            },
            source_observation_ids=payload.source_observation_ids,
        )
        self._collected.append(item)
        return item

    def get_collected(self) -> List[EvidenceItem]:
        """Return all collected evidence items."""
        return list(self._collected)

    def clear(self) -> None:
        """Clear the evidence buffer."""
        self._collected.clear()


# ── EvidenceFusionEngine ──────────────────────────────────────────────

class EvidenceFusionEngine:
    """
    Fuses multiple pieces of evidence into updated posterior beliefs
    using Bayesian-style updates.

    For each hypothesis h and evidence item e with likelihood ratio LR:
        posterior(h) ∝ prior(h) × ∏ LR(e, h)^confidence(e)

    The exponent weights the update by evidence confidence, so
    low-confidence evidence has less impact.

    This engine is domain-agnostic. Likelihood ratios are provided
    by the ``EvidenceCollector``.
    """

    def __init__(self) -> None:
        logger.info("EvidenceFusionEngine initialised")

    def fuse(
        self,
        hypotheses: List[Hypothesis],
        evidence: List[EvidenceItem],
    ) -> Dict[str, float]:
        """
        Compute posterior distribution over hypotheses given evidence.

        Args:
            hypotheses: Current hypothesis set with priors.
            evidence: Evidence items with likelihood ratios.

        Returns:
            Normalised posterior distribution ``{hypothesis_name: prob}``.
        """
        posteriors: Dict[str, float] = {}

        for h in hypotheses:
            log_posterior = math.log(max(h.prior, 1e-10))

            for ev in evidence:
                lr = ev.likelihood_ratios.get(h.name, 1.0)
                if lr > 0:
                    weighted_lr = lr ** ev.confidence
                    log_posterior += math.log(max(weighted_lr, 1e-10))

                    # Track evidence associations
                    if lr > 1.0:
                        h.supporting_evidence_ids.append(ev.evidence_id)
                    elif lr < 1.0:
                        h.refuting_evidence_ids.append(ev.evidence_id)

            posteriors[h.name] = math.exp(log_posterior)

        # Normalise
        total = sum(posteriors.values())
        if total > 0:
            posteriors = {k: v / total for k, v in posteriors.items()}
        else:
            n = len(posteriors)
            posteriors = {k: 1.0 / n for k in posteriors}

        # Update hypothesis posteriors
        for h in hypotheses:
            h.posterior = posteriors.get(h.name, 0.0)

        return posteriors


# ── BeliefManager ─────────────────────────────────────────────────────

class BeliefManager:
    """
    Maintains posterior belief distributions and provides query
    interfaces for downstream decision-making and meta-cognition.

    Tracks history of belief states for reasoning trace generation.
    """

    def __init__(self) -> None:
        self._current_beliefs: Dict[str, float] = {}
        self._history: List[Dict[str, float]] = []
        self._entropy: float = 0.0
        logger.info("BeliefManager initialised")

    def update(self, distribution: Dict[str, float]) -> None:
        """
        Set the current belief distribution.

        Computes Shannon entropy for meta-cognition monitoring.

        Args:
            distribution: ``{hypothesis_name: probability}`` — must
                          sum to ~1.0.
        """
        self._current_beliefs = dict(distribution)
        self._entropy = self._compute_entropy(distribution)
        self._history.append(dict(distribution))
        logger.debug(
            "Beliefs updated: %s (entropy=%.3f)",
            {k: f"{v:.3f}" for k, v in distribution.items()},
            self._entropy,
        )

    @property
    def current_beliefs(self) -> Dict[str, float]:
        """Current posterior distribution."""
        return dict(self._current_beliefs)

    @property
    def entropy(self) -> float:
        """Shannon entropy of the current distribution."""
        return self._entropy

    @property
    def history(self) -> List[Dict[str, float]]:
        """Historical belief distributions."""
        return list(self._history)

    def get_top_belief(self) -> Optional[Tuple[str, float]]:
        """Return the highest-probability hypothesis and its value."""
        if not self._current_beliefs:
            return None
        top = max(self._current_beliefs, key=self._current_beliefs.get)  # type: ignore
        return top, self._current_beliefs[top]

    def get_verdict(self, threshold: float = 0.70) -> str:
        """
        Produce a verdict based on the current belief state.

        Args:
            threshold: Minimum probability for a resolved verdict.

        Returns:
            ``RESOLVED`` if the top hypothesis exceeds threshold,
            ``UNRESOLVED_UNDER_THRESHOLD`` otherwise.
        """
        top = self.get_top_belief()
        if top is None:
            return "NO_BELIEFS"
        _, prob = top
        if prob >= threshold:
            return "RESOLVED"
        return "UNRESOLVED_UNDER_THRESHOLD"

    @staticmethod
    def _compute_entropy(dist: Dict[str, float]) -> float:
        """Compute Shannon entropy (base e) of a distribution."""
        ent = 0.0
        for p in dist.values():
            if p > 0:
                ent -= p * math.log(p)
        return ent


# ── ReasoningPipeline ─────────────────────────────────────────────────

class ReasoningPipeline:
    """
    Orchestrates the end-to-end reasoning cycle.

    Connects:
        HypothesisGenerator → EvidenceCollector → EvidenceFusionEngine
        → BeliefManager → DecisionCandidate selection

    Every completed reasoning cycle produces a ``ReasoningTrace`` and
    publishes ``HYPOTHESIS``, ``BELIEF``, and ``DECISION`` messages to
    the ``MessageBus``.

    Args:
        message_bus: System-wide ``MessageBus``.
        hypothesis_generator: Produces candidate hypotheses.
        evidence_collector: Gathers and annotates evidence.
        fusion_engine: Fuses evidence into posteriors.
        belief_manager: Manages belief state.
        decision_threshold: Minimum top-belief probability to issue
                            a decision (below this, verdict is unresolved).
        action_map: ``{hypothesis_name: (action, skill_name)}`` mapping
                    hypotheses to candidate actions. Injectable.

    Example::

        pipeline = ReasoningPipeline(bus, gen, collector, fuser, beliefs)
        trace = pipeline.reason(evidence_payloads)
    """

    def __init__(
        self,
        message_bus: MessageBus,
        hypothesis_generator: HypothesisGenerator,
        evidence_collector: EvidenceCollector,
        fusion_engine: EvidenceFusionEngine,
        belief_manager: BeliefManager,
        decision_threshold: float = 0.50,
        action_map: Optional[Dict[str, Tuple[str, str]]] = None,
    ) -> None:
        self._bus = message_bus
        self._hyp_gen = hypothesis_generator
        self._ev_collector = evidence_collector
        self._fusion = fusion_engine
        self._beliefs = belief_manager
        self._threshold = decision_threshold
        self._action_map = action_map or {}
        logger.info(
            "ReasoningPipeline initialised (threshold=%.2f)", decision_threshold
        )

    def reason(
        self,
        evidence_payloads: List[EvidencePayload],
        evidence_confidences: Optional[List[float]] = None,
    ) -> ReasoningTrace:
        """
        Execute a full reasoning cycle.

        Args:
            evidence_payloads: Evidence from the perception layer.
            evidence_confidences: Optional confidence values per payload.

        Returns:
            A complete ``ReasoningTrace`` with full provenance.
        """
        trace_id = _uuid.uuid4().hex[:12]
        confidence_propagation: Dict[str, float] = {}

        # ── Stage 1: Hypothesis Generation ────────────────────────────
        all_indicators: List[str] = []
        for ep in evidence_payloads:
            all_indicators.extend(
                s.strip() for s in ep.indicator.split(",") if s.strip()
            )

        hypotheses = self._hyp_gen.generate(all_indicators)
        confidence_propagation["hypothesis_generation"] = (
            max(h.prior for h in hypotheses) if hypotheses else 0.0
        )

        # Publish HYPOTHESIS messages
        for h in hypotheses:
            h_msg = create_message(
                source="reasoning_pipeline",
                destination="BROADCAST",
                message_type=MessageType.HYPOTHESIS,
                payload=HypothesisPayload(
                    hypothesis_id=h.hypothesis_id,
                    suspected_cause=h.name,
                    prior_probability=h.prior,
                ),
                confidence=h.prior,
                metadata={"trace_id": trace_id},
            )
            self._bus.publish(h_msg)

        # ── Stage 2: Evidence Collection ──────────────────────────────
        confs = evidence_confidences or [1.0] * len(evidence_payloads)
        evidence_items: List[EvidenceItem] = []
        for ep, conf in zip(evidence_payloads, confs):
            item = self._ev_collector.collect_from_payload(ep, conf)
            evidence_items.append(item)

        avg_evidence_conf = (
            sum(e.confidence for e in evidence_items) / len(evidence_items)
            if evidence_items
            else 0.0
        )
        confidence_propagation["evidence_collection"] = avg_evidence_conf

        # ── Stage 3: Evidence Fusion ──────────────────────────────────
        posterior = self._fusion.fuse(hypotheses, evidence_items)
        confidence_propagation["evidence_fusion"] = max(
            posterior.values()
        ) if posterior else 0.0

        # ── Stage 4: Belief Update ────────────────────────────────────
        self._beliefs.update(posterior)
        confidence_propagation["belief_update"] = 1.0 - self._beliefs.entropy

        # Publish BELIEF message
        belief_msg = create_message(
            source="reasoning_pipeline",
            destination="BROADCAST",
            message_type=MessageType.BELIEF,
            payload=BeliefPayload(
                belief_distribution=self._beliefs.current_beliefs,
                entropy=self._beliefs.entropy,
                verdict=self._beliefs.get_verdict(self._threshold),
            ),
            confidence=max(posterior.values()) if posterior else 0.0,
            metadata={"trace_id": trace_id},
        )
        self._bus.publish(belief_msg)

        # ── Stage 5: Decision Selection ───────────────────────────────
        selected_decision: Optional[DecisionCandidate] = None
        top = self._beliefs.get_top_belief()

        if top and top[1] >= self._threshold:
            hyp_name, prob = top
            action_info = self._action_map.get(hyp_name, ("intervene", "generic_intervention"))
            action, skill = action_info

            selected_decision = DecisionCandidate(
                decision_id=_uuid.uuid4().hex[:12],
                action=action,
                skill_name=skill,
                score=prob,
                confidence=prob * avg_evidence_conf,
                justification=[hyp_name],
                parameters={"target_hypothesis": hyp_name},
            )
            confidence_propagation["decision"] = selected_decision.confidence

            # Publish DECISION message
            dec_msg = create_message(
                source="reasoning_pipeline",
                destination="planning_layer",
                message_type=MessageType.DECISION,
                payload=DecisionPayload(
                    decision_id=selected_decision.decision_id,
                    justification_ids=[
                        h.hypothesis_id
                        for h in hypotheses
                        if h.name == hyp_name
                    ],
                    action_selected=action,
                    skill_name=skill,
                    parameters=selected_decision.parameters,
                ),
                confidence=selected_decision.confidence,
                metadata={"trace_id": trace_id},
            )
            self._bus.publish(dec_msg)

        # ── Build Trace ───────────────────────────────────────────────
        trace = ReasoningTrace(
            trace_id=trace_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            hypotheses_generated=hypotheses,
            evidence_collected=evidence_items,
            belief_distribution=self._beliefs.current_beliefs,
            entropy=self._beliefs.entropy,
            selected_decision=selected_decision,
            confidence_propagation=confidence_propagation,
        )

        logger.info(
            "Reasoning cycle %s complete: verdict=%s, entropy=%.3f",
            trace_id,
            self._beliefs.get_verdict(self._threshold),
            self._beliefs.entropy,
        )
        return trace
