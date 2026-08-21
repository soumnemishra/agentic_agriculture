"""
ACA Milestone 7 — Unit & Integration Tests (Evaluation & Telemetry Logger)
==========================================================================

Comprehensive tests covering:
    - Task 1: CognitiveMetricsLogger lifecycle and causal chain tracing.
    - Task 2: CSV and JSONL dataset generation and format validity.
    - Task 3: Aggregate performance metric calculations.
    - Task 4: Multi-phase synthetic scenario frame generation.
"""

from __future__ import annotations

import csv
import os
import tempfile
import unittest
from datetime import datetime, timezone
from typing import Any, Dict, List

import torch

from aca.config import ACAConfig, MessageBusConfig
from aca.logging_config import setup_logging
from aca.orchestration.message_bus import MessageBus
from aca.orchestration.schemas import (
    ACAMessage,
    DecisionPayload,
    FeedbackPayload,
    HypothesisPayload,
    MessageType,
    ObservationPayload,
    create_message,
)
from evaluation.metrics_logger import CognitiveCycleRecord, CognitiveMetricsLogger
from evaluation.run_experiment import generate_scenario_frame

# Initialise logging once
setup_logging(ACAConfig().logging)


class TestCognitiveMetricsLogger(unittest.TestCase):
    def setUp(self) -> None:
        self.bus = MessageBus(MessageBusConfig(enable_tracing=True))
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.temp_dir.name, "test_results.csv")
        self.jsonl_path = os.path.join(self.temp_dir.name, "test_traces.jsonl")

        self.logger = CognitiveMetricsLogger(
            message_bus=self.bus,
            csv_output_path=self.csv_path,
            jsonl_output_path=self.jsonl_path,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_record_lifecycle_and_causal_tracing(self) -> None:
        """Verify full lifecycle from start_step through all 4 message types to end_step."""
        step_idx = 1
        self.logger.start_step(step_idx)

        # 1. Observation
        obs_payload = ObservationPayload(
            observation_id="obs_test_101",
            source_sensors=["sensor_temp", "camera"],
            target_zone="zone_a",
            observation_time=datetime.now(timezone.utc).isoformat(),
            measurements={
                "environment_temperature_c": 24.5,
                "environment_humidity": 88.0,
                "soil_moisture": 60.0,
                "vision_confidence": 0.92,
            },
        )
        obs_msg = create_message(
            source="perception_agent",
            destination="BROADCAST",
            message_type=MessageType.OBSERVATION,
            payload=obs_payload,
            metadata={"entry_id": 2245, "predicted_class": "Late_blight", "vision_diagnosis": {"inference_time_ms": 25.4}},
        )
        self.bus.publish(obs_msg)

        # 2. Hypothesis
        hyp_payload = HypothesisPayload(
            hypothesis_id="hyp_test_101",
            associated_evidence_ids=["obs_test_101"],
            suspected_cause="Late_blight",
            prior_probability=0.92,
            likelihood_ratio=1.85,
        )
        hyp_msg = create_message(
            source="reasoning_agent",
            destination="BROADCAST",
            message_type=MessageType.HYPOTHESIS,
            payload=hyp_payload,
            confidence=0.95,
            metadata={"etiology_supported": True, "model_engine": "gemma4:4b-q4_K_M", "reasoning_latency_ms": 12.3},
        )
        self.bus.publish(hyp_msg)

        # 3. Decision
        dec_payload = DecisionPayload(
            decision_id="dec_test_101",
            justification_ids=["hyp_test_101", "obs_test_101"],
            action_selected="EMERGENCY_FUNGAL_CONTAINMENT",
            skill_name="actuator_dispatch",
            parameters={"tool_calls": [{"tool_name": "irrigation_control"}, {"tool_name": "treatment_alert"}]},
        )
        dec_msg = create_message(
            source="planning_agent",
            destination="BROADCAST",
            message_type=MessageType.DECISION,
            payload=dec_payload,
            metadata={"planning_latency_ms": 8.7},
        )
        self.bus.publish(dec_msg)

        # 4. Feedback
        fb_payload = FeedbackPayload(
            action_id="dec_test_101",
            expected_outcome={"action": "EMERGENCY_FUNGAL_CONTAINMENT"},
            actual_outcome={"success_count": 2, "error_count": 0},
            deviation=0.0,
            assessment="SUCCESS",
        )
        fb_msg = create_message(
            source="execution_agent",
            destination="BROADCAST",
            message_type=MessageType.FEEDBACK,
            payload=fb_payload,
            metadata={"execution_latency_ms": 0.4},
        )
        self.bus.publish(fb_msg)

        # End Step
        rec = self.logger.end_step(step_idx)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.step_index, 1)
        self.assertEqual(rec.entry_id, 2245)
        self.assertEqual(rec.vision_predicted_class, "Late_blight")
        self.assertEqual(rec.vision_confidence, 0.92)
        self.assertEqual(rec.vision_latency_ms, 25.4)
        self.assertEqual(rec.reasoning_cause, "Late_blight")
        self.assertTrue(rec.reasoning_etiology_supported)
        self.assertEqual(rec.reasoning_latency_ms, 12.3)
        self.assertEqual(rec.planning_action, "EMERGENCY_FUNGAL_CONTAINMENT")
        self.assertEqual(rec.planning_latency_ms, 8.7)
        self.assertEqual(rec.execution_assessment, "SUCCESS")
        self.assertEqual(rec.execution_latency_ms, 0.4)
        self.assertTrue(rec.causal_chain_valid)

    def test_flush_and_dataset_output(self) -> None:
        """Verify CSV and JSONL files are flushed properly to disk."""
        self.test_record_lifecycle_and_causal_tracing()
        self.logger.flush_to_disk()

        self.assertTrue(os.path.exists(self.csv_path))
        self.assertTrue(os.path.exists(self.jsonl_path))

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["vision_predicted_class"], "Late_blight")
            self.assertEqual(rows[0]["causal_chain_valid"], "True")

        summary = self.logger.generate_summary()
        self.assertEqual(summary["total_cycles"], 1)
        self.assertEqual(summary["causal_chain_integrity_pct"], 100.0)
        self.assertEqual(summary["actuator_execution_success_pct"], 100.0)


class TestScenarioGenerator(unittest.TestCase):
    def test_frame_generation_shape_and_seed(self) -> None:
        """Verify tensor dimensions and determinism across steps."""
        frame1 = generate_scenario_frame(5, 100)
        frame2 = generate_scenario_frame(5, 100)
        frame3 = generate_scenario_frame(25, 100)

        self.assertEqual(frame1.shape, (1, 3, 224, 224))
        self.assertTrue(torch.equal(frame1, frame2))
        self.assertFalse(torch.equal(frame1, frame3))


if __name__ == "__main__":
    unittest.main()
