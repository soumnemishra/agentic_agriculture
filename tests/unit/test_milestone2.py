"""
ACA Milestone 2 — Unit Tests
=============================

Comprehensive tests covering the cognitive core:
    - Perception Layer (Validator, Normalizer, Manager)
    - Reasoning Layer (Hypothesis, Evidence, Fusion, Beliefs, Pipeline)
    - Planning Layer (Goal, Task, Skill, Execution)
    - Meta-Cognition (Confidence, Conflicts, Reflection, Escalation, Replanning)
    - Learning Layer (Recorder, Updater, FeedbackProcessor)
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List, Tuple
from datetime import datetime, timezone, timedelta

# ── Core Infrastructure ────────────────────────────────────────────────
from aca.config import ACAConfig, MessageBusConfig, MemoryConfig
from aca.logging_config import setup_logging
from aca.orchestration.message_bus import MessageBus
from aca.orchestration.schemas import (
    ObservationPayload, EvidencePayload, DecisionPayload, FeedbackPayload,
    create_message, MessageType
)
from aca.memory.working_memory import WorkingMemory
from aca.memory.episodic_memory import EpisodicMemory
from aca.memory.semantic_memory import SemanticMemory
from aca.skills.registry import SkillRegistry
from aca.tools.registry import ToolRegistry

# ── Perception Layer ──────────────────────────────────────────────────
from aca.cognition.perception.perception_processor import (
    SensorSchema, ObservationValidator, ObservationNormalizer,
    ObservationManager, FeatureObject
)

# ── Reasoning Layer ───────────────────────────────────────────────────
from aca.cognition.reasoning.reasoning_engine import (
    HypothesisGenerator, EvidenceCollector, EvidenceFusionEngine,
    BeliefManager, ReasoningPipeline, EvidenceItem, Hypothesis
)

# ── Planning Layer ────────────────────────────────────────────────────
from aca.cognition.planning.planning_engine import (
    GoalPlanner, TaskPlanner, SkillSelector, ExecutionPlanner, Goal, PlannedTask
)

# ── Meta-Cognition Layer ──────────────────────────────────────────────
from aca.cognition.meta_cognition.cognitive_monitor import (
    ConfidenceMonitor, ConflictDetector, ReflectionEngine,
    EscalationManager, ReplanningManager, EscalationType, ConflictSeverity
)

# ── Learning Layer ────────────────────────────────────────────────────
from aca.cognition.learning.cognitive_learner import (
    ExperienceRecorder, MemoryUpdater, KnowledgeUpdater, FeedbackProcessor,
    PredictionError
)

# Initialise logging once
setup_logging(ACAConfig().logging)


# =====================================================================
# Test: Perception Layer
# =====================================================================

class TestPerceptionLayer(unittest.TestCase):
    def setUp(self):
        self.bus = MessageBus(MessageBusConfig(enable_tracing=False))
        self.validator = ObservationValidator()
        self.validator.register_schema(SensorSchema(
            sensor_type="moisture",
            required_fields=["volumetric_water_content"],
            valid_ranges={"volumetric_water_content": (0.0, 1.0)}
        ))
        self.normalizer = ObservationNormalizer({
            "volumetric_water_content": (0.0, 0.60)
        })
        self.manager = ObservationManager(
            self.bus, self.validator, self.normalizer,
            sensor_type_map={"sensor_1": "moisture"}
        )

    def test_validator_valid_reading(self):
        obs = ObservationPayload(observation_id="obs1", source_sensors=["sensor_1"], target_zone="zone1", measurements={"volumetric_water_content": 0.3})
        result = self.validator.validate(obs, "moisture")
        self.assertTrue(result.is_valid)
        self.assertEqual(result.adjusted_confidence, 1.0)

    def test_validator_missing_field(self):
        obs = ObservationPayload(observation_id="obs1", source_sensors=["sensor_1"], target_zone="zone1", measurements={"temperature": 25.0})
        result = self.validator.validate(obs, "moisture")
        self.assertFalse(result.is_valid)

    def test_validator_boundary_penalty(self):
        obs = ObservationPayload(observation_id="obs1", source_sensors=["sensor_1"], target_zone="zone1", measurements={"volumetric_water_content": 0.98})
        result = self.validator.validate(obs, "moisture")
        self.assertTrue(result.is_valid)
        self.assertLess(result.adjusted_confidence, 1.0)

    def test_normalizer(self):
        norm_val = self.normalizer.normalise("volumetric_water_content", 0.30)
        self.assertAlmostEqual(norm_val, 0.5)

    def test_manager_processing(self):
        obs = ObservationPayload(observation_id="obs1", source_sensors=["sensor_1"], target_zone="zone1", measurements={"volumetric_water_content": 0.30})
        features = self.manager.process_observation(obs)
        self.assertEqual(len(features), 1)
        self.assertEqual(features[0].name, "volumetric_water_content")
        self.assertAlmostEqual(features[0].normalised_value, 0.5)


# =====================================================================
# Test: Reasoning Layer
# =====================================================================

class TestReasoningLayer(unittest.TestCase):
    def setUp(self):
        self.bus = MessageBus(MessageBusConfig(enable_tracing=False))
        
        def mock_gen(inds):
            return [("disease_a", 0.5), ("disease_b", 0.5)]
            
        self.hyp_gen = HypothesisGenerator(mock_gen)
        self.ev_col = EvidenceCollector(default_likelihood=1.0)
        self.ev_col.register_likelihood("spotting", "disease_a", 3.0)
        self.ev_col.register_likelihood("spotting", "disease_b", 0.5)
        
        self.fusion = EvidenceFusionEngine()
        self.beliefs = BeliefManager()
        self.pipeline = ReasoningPipeline(
            self.bus, self.hyp_gen, self.ev_col, self.fusion, self.beliefs,
            action_map={"disease_a": ("treat_a", "spray_fungicide")}
        )

    def test_hypothesis_generation(self):
        hyps = self.hyp_gen.generate(["spotting"])
        self.assertEqual(len(hyps), 2)
        self.assertAlmostEqual(hyps[0].prior, 0.5)

    def test_evidence_collection(self):
        payload = EvidencePayload("ev1", ["obs1"], "spotting", 0.8)
        item = self.ev_col.collect_from_payload(payload)
        self.assertEqual(item.likelihood_ratios["disease_a"], 3.0)

    def test_fusion_and_beliefs(self):
        hyps = self.hyp_gen.generate(["spotting"])
        ev = self.ev_col.collect_from_payload(EvidencePayload("ev1", ["obs1"], "spotting", 0.8), message_confidence=1.0)
        posterior = self.fusion.fuse(hyps, [ev])
        self.assertGreater(posterior["disease_a"], posterior["disease_b"])
        self.beliefs.update(posterior)
        self.assertEqual(self.beliefs.get_verdict(0.6), "RESOLVED")

    def test_full_pipeline(self):
        payload = EvidencePayload("ev1", ["obs1"], "spotting", 0.8)
        trace = self.pipeline.reason([payload])
        self.assertIsNotNone(trace)
        self.assertIsNotNone(trace.selected_decision)
        if trace.selected_decision:
            self.assertEqual(trace.selected_decision.action, "treat_a")


# =====================================================================
# Test: Planning Layer
# =====================================================================

class TestPlanningLayer(unittest.TestCase):
    def setUp(self):
        self.bus = MessageBus(MessageBusConfig(enable_tracing=False))
        self.goal_planner = GoalPlanner()
        self.task_planner = TaskPlanner(default_zone="field_1")
        self.skill_reg = SkillRegistry(ToolRegistry())
        self.skill_selector = SkillSelector(self.skill_reg, {"irrigate": "irrigation_skill"})
        self.exec_planner = ExecutionPlanner(
            self.bus, self.goal_planner, self.task_planner, self.skill_selector
        )

    def test_goal_decomposition(self):
        goals = self.goal_planner.decompose("m1", "irrigate", 0.9)
        self.assertEqual(len(goals), 1)
        self.assertEqual(goals[0].confidence, 0.9)

    def test_task_planning(self):
        goal = Goal("g1", "m1", "desc", "metric", 1.0)
        tasks = self.task_planner.plan_tasks(goal, "skill1", subtask_count=2)
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[1].depends_on[0], tasks[0].task_id)

    def test_execution_planner(self):
        plan = self.exec_planner.create_plan("m1", "irrigate", confidence=0.8)
        self.assertEqual(len(plan.goals), 1)
        self.assertEqual(len(plan.tasks), 1)
        self.exec_planner.publish_plan(plan)


# =====================================================================
# Test: Meta-Cognition Layer
# =====================================================================

class TestMetaCognitionLayer(unittest.TestCase):
    def setUp(self):
        self.bus = MessageBus(MessageBusConfig(enable_tracing=False))
        self.conf_mon = ConfidenceMonitor(0.4, 0.5)
        self.conf_det = ConflictDetector(min_gap=0.15)
        self.ref_eng = ReflectionEngine()
        self.esc_man = EscalationManager(self.bus, auto_escalation_confidence=0.5)
        self.rep_man = ReplanningManager(max_replan_attempts=2)

    def test_confidence_monitor(self):
        assessment = self.conf_mon.assess({"stage1": 0.9, "stage2": 0.3})
        self.assertFalse(assessment.overall_adequate)
        self.assertEqual(assessment.bottleneck_stage, "stage2")

    def test_conflict_detector_critical(self):
        beliefs = {"h1": 0.50, "h2": 0.49}
        report = self.conf_det.detect(beliefs, entropy=0.5)
        self.assertEqual(report.severity, ConflictSeverity.CRITICAL)

    def test_reflection_engine(self):
        assessment = self.conf_mon.assess({"s1": 0.9})
        result = self.ref_eng.reflect(2, 0, {}, {}, assessment) # 0 evidence
        self.assertFalse(result.evidence_sufficiency)
        self.assertLess(result.reasoning_quality_score, 1.0)

    def test_escalation_manager(self):
        assessment = self.conf_mon.assess({"s1": 0.2})
        report = self.conf_det.detect({"h1": 0.9, "h2": 0.1}, 0.2)
        reflection = self.ref_eng.reflect(2, 2, {}, {}, assessment)
        
        decision = self.esc_man.evaluate(assessment, report, reflection)
        self.assertIsNotNone(decision)
        if decision:
            self.assertEqual(decision.escalation_type, EscalationType.HUMAN_REVIEW)

    def test_replanning_manager(self):
        should, _ = self.rep_man.should_replan("plan1", "failed")
        self.assertTrue(should)
        self.rep_man.should_replan("plan1", "failed")
        should3, _ = self.rep_man.should_replan("plan1", "failed")
        self.assertFalse(should3) # Exceeded max 2 attempts


# =====================================================================
# Test: Learning Layer
# =====================================================================

class TestLearningLayer(unittest.TestCase):
    def setUp(self):
        self.bus = MessageBus(MessageBusConfig(enable_tracing=False))
        self.em = EpisodicMemory(MemoryConfig())
        self.wm = WorkingMemory(MemoryConfig())
        self.sm = SemanticMemory(MemoryConfig(semantic_readonly=False))
        self.sm.store("thresholds", "metric1", 10.0)
        
        self.recorder = ExperienceRecorder(self.em)
        self.mem_upd = MemoryUpdater(self.wm)
        self.know_upd = KnowledgeUpdater(self.sm, learning_rate=0.5)
        self.processor = FeedbackProcessor(
            self.bus, self.recorder, self.mem_upd, self.know_upd
        )

    def test_prediction_error(self):
        pe = PredictionError("m1", 10.0, 11.0)
        self.assertEqual(pe.absolute_error, 1.0)
        self.assertAlmostEqual(pe.relative_error, 0.1)
        self.assertEqual(pe.assessment, "ACCEPTABLE")

    def test_knowledge_updater(self):
        self.know_upd.apply_prediction_errors("thresholds", [PredictionError("metric1", 10.0, 12.0)])
        new_val = self.sm.retrieve("thresholds", "metric1")
        # lr=0.5: 0.5*10 + 0.5*12 = 11
        self.assertAlmostEqual(new_val, 11.0)

    def test_feedback_processor(self):
        outcome = self.processor.process_feedback(
            "act1", {"metric1": 10.0}, {"metric1": 10.1}
        )
        self.assertEqual(outcome.overall_assessment, "PLAN_SUCCESSFUL")
        self.assertEqual(outcome.memory_updates_applied, 1)
        
        episodes = self.em.query()
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0].outcome_assessment, "PLAN_SUCCESSFUL")


# =====================================================================
# Run
# =====================================================================

if __name__ == "__main__":
    unittest.main()
