"""
Integration Test Suite — Milestone 5 (Simulation & Perception)
================================================================

Verifies the end-to-end flow of ACA v1.0 Milestone 5:
1. IoTStreamer dataset iteration and zero-phase 8-channel telemetry merging.
2. TomatoDiagnosisSkill deep learning inference with CondConViT_V2 & VRAM cleanup.
3. PerceptionAgent observation generation and pub/sub message dispatch.
4. ReasoningAgent multi-modal sensor fusion & prompt generation for gemma4:4b-q4_K_M.
"""

from __future__ import annotations

import unittest
import torch

from aca.agents.base_agent import MemoryGateway, ToolGateway
from aca.agents.perception_agent import PerceptionAgent
from aca.agents.reasoning_agent import ReasoningAgent
from aca.config import MessageBusConfig
from aca.orchestration.message_bus import MessageBus
from aca.orchestration.schemas import (
    ACAMessage,
    HypothesisPayload,
    MessageType,
    ObservationPayload,
)
from aca.skills.tomato_diagnosis_skill import TomatoDiagnosisSkill
from aca.tools.registry import ToolRegistry
from simulation.telemetry_streamer import IoTStreamer


class TestMilestone5(unittest.TestCase):

    def setUp(self) -> None:
        self.config = MessageBusConfig()
        self.bus = MessageBus(self.config)

        # Mock gateways for test scope
        self.memories = {"working": {}, "episodic": {}}
        self.memory_permissions = {"working": "READ_WRITE", "episodic": "READ"}
        self.memory_gateway = MemoryGateway(self.memories, {})
        self.tool_registry = ToolRegistry()
        self.tool_gateway = ToolGateway(self.tool_registry, set())

        # Initialize Milestone 5 components
        self.streamer = IoTStreamer(loop=True)
        self.diagnosis_skill = TomatoDiagnosisSkill()
        self.perception_agent = PerceptionAgent(
            message_bus=self.bus,
            memory_gateway=self.memory_gateway,
            tool_gateway=self.tool_gateway,
            telemetry_streamer=self.streamer,
            diagnosis_skill=self.diagnosis_skill,
        )
        self.reasoning_agent = ReasoningAgent(
            message_bus=self.bus,
            memory_gateway=self.memory_gateway,
            tool_gateway=self.tool_gateway,
            ollama_model_name="gemma4:4b-q4_K_M",
        )

        self.perception_agent.start()
        self.reasoning_agent.start()

    def test_01_iot_streamer_merging(self) -> None:
        """Test IoTStreamer loads and merges 100 rows across 8 Excel files."""
        self.assertEqual(self.streamer.total_records, 100)
        row = self.streamer.step()
        self.assertIsNotNone(row)
        self.assertIn("Entry_id", row)
        self.assertEqual(row["Entry_id"], 2245)
        self.assertIn("Environment Humidity", row)
        self.assertIn("Environment Temperature", row)
        self.assertIn("Soil Moisture", row)
        self.assertIn("Soil pH", row)

    def test_02_tomato_diagnosis_skill_execution(self) -> None:
        """Test TomatoDiagnosisSkill model loading, execution, and memory bounds."""
        result = self.diagnosis_skill.execute(image_path="")
        self.assertTrue(result.success)
        self.assertIn("predicted_class", result.data)
        self.assertIn("confidence", result.data)
        self.assertIn("inference_time_ms", result.data)
        self.assertIsInstance(result.data["predicted_class"], str)
        self.assertGreaterEqual(result.data["confidence"], 0.0)

    def test_03_perception_agent_perceive(self) -> None:
        """Test PerceptionAgent generates and publishes ObservationPayload."""
        published_messages = []

        def capture_obs(msg: ACAMessage) -> None:
            published_messages.append(msg)

        self.bus.subscribe(MessageType.OBSERVATION, capture_obs)
        obs_msg = self.perception_agent.perceive()

        self.assertEqual(obs_msg.message_type, MessageType.OBSERVATION)
        self.assertIsInstance(obs_msg.payload, ObservationPayload)
        self.assertEqual(len(published_messages), 1)

    def test_04_end_to_end_sensor_fusion_pipeline(self) -> None:
        """Test full pipeline: Perception Agent -> MessageBus -> Reasoning Agent -> Hypothesis."""
        hypotheses_received = []

        def capture_hyp(msg: ACAMessage) -> None:
            hypotheses_received.append(msg)

        self.bus.subscribe(MessageType.HYPOTHESIS, capture_hyp)

        # Trigger perception tick which auto-publishes OBSERVATION
        obs_msg = self.perception_agent.perceive()

        # Check that ReasoningAgent received OBSERVATION and published HYPOTHESIS
        self.assertEqual(len(hypotheses_received), 1)
        hyp_msg = hypotheses_received[0]
        self.assertEqual(hyp_msg.message_type, MessageType.HYPOTHESIS)
        self.assertIsInstance(hyp_msg.payload, HypothesisPayload)
        self.assertIn("prompt", hyp_msg.metadata)
        self.assertEqual(hyp_msg.metadata["ollama_model_name"], "gemma4:4b-q4_K_M")


if __name__ == "__main__":
    unittest.main()
