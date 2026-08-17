"""
ACA Milestone 5 — Unit & Integration Tests (Simulation & Perception)
====================================================================

Comprehensive tests covering:
    - Task 1: IoT Microclimate Telemetry Streamer (8 synchronized Excel files)
    - Task 2: Tomato Diagnosis BaseSkill (CondConViT_V2 PyTorch model & 11 classes)
    - Task 3: Perception Agent (Multi-modal sensor & vision observation publishing)
    - Task 4: Reasoning Agent (Sensor fusion & agronomic etiology reasoning)
    - Task 5: End-to-End Perception -> MessageBus -> Reasoning Pipeline
"""

from __future__ import annotations

import os
import unittest
from datetime import datetime, timezone
from typing import Any, Dict, List

import torch

from aca.agents.base_agent import (
    CognitiveLayer,
    MemoryAccess,
    MemoryGateway,
    ToolGateway,
)
from aca.agents.perception_agent import PerceptionAgent
from aca.agents.reasoning_agent import ETIOLOGY_PROFILES, ReasoningAgent
from aca.config import ACAConfig, MessageBusConfig
from aca.logging_config import setup_logging
from aca.orchestration.message_bus import MessageBus
from aca.orchestration.schemas import (
    ACAMessage,
    HypothesisPayload,
    MessageType,
    ObservationPayload,
    TaskPayload,
    create_message,
)
from aca.skills.tomato_diagnosis_skill import (
    TOMATO_CLASSES,
    CondConViT_V2,
    TomatoDiagnosisSkill,
)
from aca.tools.base_tool import BaseTool, ToolResult, ToolSchema
from aca.tools.registry import ToolRegistry
from simulation.telemetry_streamer import (
    CHANNEL_MAPPINGS,
    IoTStreamer,
    TelemetryRecord,
)

# Initialise logging once
setup_logging(ACAConfig().logging)


class MockTool(BaseTool):
    """Mock tool subclassing BaseTool for registry registration."""
    def __init__(self, tool_name: str) -> None:
        self._name = tool_name

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(name=self._name, description=f"Mock tool for {self._name}")

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, data={"status": "ok", "kwargs": kwargs})


# =====================================================================
# Test 1: IoT Telemetry Streamer
# =====================================================================

class TestIoTTelemetryStreamer(unittest.TestCase):
    def setUp(self) -> None:
        self.streamer = IoTStreamer(loop=True, auto_load=True)

    def test_dataset_merging_and_total_rows(self) -> None:
        """Verify that all 8 Excel files merge into 100 synchronized rows."""
        self.assertEqual(self.streamer.total_records, 100)
        self.assertEqual(len(CHANNEL_MAPPINGS), 8)

    def test_step_and_field_schema(self) -> None:
        """Verify that step() returns a complete, typed telemetry dictionary."""
        row = self.streamer.step()
        self.assertIsNotNone(row)
        self.assertIn("entry_id", row)
        self.assertIn("timestamp", row)
        self.assertIn("environment_humidity", row)
        self.assertIn("environment_light_lux", row)
        self.assertIn("environment_temperature_c", row)
        self.assertIn("soil_moisture", row)
        self.assertIn("soil_ph", row)
        self.assertIn("soil_temperature_c", row)
        self.assertIn("solar_battery_voltage", row)
        self.assertIn("water_tds", row)
        self.assertIn("units", row)

        self.assertEqual(row["entry_id"], 2245)
        self.assertGreater(row["environment_humidity"], 0.0)

    def test_iterator_and_reset(self) -> None:
        """Verify generator iteration and pointer reset."""
        self.streamer.reset(loop=False)
        self.assertEqual(self.streamer.current_index, 0)

        rows = list(self.streamer)
        self.assertEqual(len(rows), 100)

        # After exhausting without loop, next step returns None
        self.assertIsNone(self.streamer.step())

        # Reset
        self.streamer.reset(loop=True)
        self.assertIsNotNone(self.streamer.step())


# =====================================================================
# Test 2: Tomato Diagnosis BaseSkill & CondConViT_V2 Architecture
# =====================================================================

class TestTomatoDiagnosisSkill(unittest.TestCase):
    def setUp(self) -> None:
        self.skill = TomatoDiagnosisSkill(auto_load=True)

    def test_classes_count_and_labels(self) -> None:
        """Verify that 11 classes are mapped correctly."""
        self.assertEqual(len(TOMATO_CLASSES), 11)
        self.assertIn("healthy", TOMATO_CLASSES)
        self.assertIn("Early_blight", TOMATO_CLASSES)
        self.assertIn("Late_blight", TOMATO_CLASSES)
        self.assertIn("Septoria_leaf_spot", TOMATO_CLASSES)
        self.assertIn("powdery_mildew", TOMATO_CLASSES)

    def test_model_architecture_and_forward_pass(self) -> None:
        """Verify CondConViT_V2 forward pass shape and Softmax validity."""
        model = CondConViT_V2(num_classes=11)
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            logits = model(dummy_input)

        self.assertEqual(logits.shape, (1, 11))
        probs = torch.softmax(logits, dim=1)
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=4)

    def test_weights_loaded_without_mismatch(self) -> None:
        """Verify model loads best_model_v5.pth weights in evaluation mode."""
        self.assertIsNotNone(self.skill.model)
        self.assertTrue(self.skill._is_loaded)
        self.assertFalse(self.skill.model.training)

    def test_execute_with_tensor_input(self) -> None:
        """Verify execute() returns normalized SkillResult dictionary."""
        dummy_tensor = torch.randn(1, 3, 224, 224)
        result = self.skill.execute(image_path=dummy_tensor)

        self.assertTrue(result.success)
        data = result.data
        self.assertIn("predicted_class", data)
        self.assertIn("confidence", data)
        self.assertIn("inference_time_ms", data)
        self.assertIn("all_probabilities", data)

        self.assertIn(data["predicted_class"], TOMATO_CLASSES)
        self.assertGreaterEqual(data["confidence"], 0.0)
        self.assertLessEqual(data["confidence"], 1.0)
        self.assertEqual(len(data["all_probabilities"]), 11)

    def test_schema_metadata(self) -> None:
        """Verify declarative SkillSchema compliance."""
        schema = self.skill.schema
        self.assertEqual(schema.name, "tomato_diagnosis")
        self.assertEqual(len(schema.parameters), 1)
        self.assertEqual(schema.parameters[0].name, "image_path")


# =====================================================================
# Test 3: Perception Agent
# =====================================================================

class TestPerceptionAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.bus = MessageBus(MessageBusConfig(enable_tracing=True))
        self.tool_reg = ToolRegistry()
        self.tool_reg.register(MockTool("sensor_read"))
        self.tool_reg.register(MockTool("camera_capture"))

        self.mem_gateway = MemoryGateway({}, {"working": MemoryAccess.WRITE})
        self.tool_gateway = ToolGateway(self.tool_reg, {"sensor_read", "camera_capture"})

        self.streamer = IoTStreamer(loop=True)
        self.skill = TomatoDiagnosisSkill(auto_load=True)

        self.agent = PerceptionAgent(
            message_bus=self.bus,
            memory_gateway=self.mem_gateway,
            tool_gateway=self.tool_gateway,
            iot_streamer=self.streamer,
            diagnosis_skill=self.skill,
            target_zone="greenhouse_bay_3",
        )
        self.agent.start()

    def test_contract_declaration(self) -> None:
        """Verify strict AgentContract compliance."""
        contract = self.agent.contract
        self.assertEqual(contract.agent_name, "perception_agent")
        self.assertEqual(contract.cognitive_layer, CognitiveLayer.PERCEPTION)
        self.assertIn(MessageType.OBSERVATION, contract.messages_published)
        self.assertIn(MessageType.TASK, contract.messages_subscribed)

    def test_perceive_cycle_and_message_structure(self) -> None:
        """Verify that perceive() publishes a valid ACAMessage(MessageType.OBSERVATION)."""
        published_msgs: List[ACAMessage] = []
        self.bus.subscribe(MessageType.OBSERVATION, lambda m: published_msgs.append(m))

        dummy_img = torch.randn(1, 3, 224, 224)
        msg = self.agent.perceive(image_path=dummy_img)

        self.assertEqual(len(published_msgs), 1)
        received = published_msgs[0]

        self.assertEqual(received.message_type, MessageType.OBSERVATION)
        self.assertEqual(received.source, "perception_agent")
        self.assertIsInstance(received.payload, ObservationPayload)

        payload: ObservationPayload = received.payload
        self.assertEqual(payload.target_zone, "greenhouse_bay_3")
        self.assertIn("environment_temperature_c", payload.measurements)
        self.assertIn("environment_humidity", payload.measurements)
        self.assertIn("soil_moisture", payload.measurements)
        self.assertIn("vision_confidence", payload.measurements)

        meta = received.metadata
        self.assertIn("predicted_class", meta)
        self.assertIn("entry_id", meta)

    def test_task_trigger_processing(self) -> None:
        """Verify agent reacts to TASK messages."""
        task = TaskPayload(
            task_id="t_001",
            goal_id="g_001",
            skill_required="tomato_diagnosis",
            target_zone="zone_north",
            parameters={"image_path": torch.randn(1, 3, 224, 224)},
        )
        task_msg = create_message(
            source="supervisor",
            destination="perception_agent",
            message_type=MessageType.TASK,
            payload=task,
        )

        resp = self.agent.process(task_msg)
        self.assertIsNotNone(resp)
        self.assertEqual(resp.message_type, MessageType.OBSERVATION)


# =====================================================================
# Test 4: Reasoning Agent (Sensor Fusion)
# =====================================================================

class TestReasoningAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.bus = MessageBus(MessageBusConfig(enable_tracing=True))
        self.tool_reg = ToolRegistry()
        self.tool_reg.register(MockTool("llm_infer"))
        self.tool_reg.register(MockTool("agronomy_rules"))

        self.mem_gateway = MemoryGateway({}, {"working": MemoryAccess.READ_WRITE, "semantic": MemoryAccess.READ})
        self.tool_gateway = ToolGateway(self.tool_reg, {"llm_infer", "agronomy_rules"})

        self.agent = ReasoningAgent(
            message_bus=self.bus,
            memory_gateway=self.mem_gateway,
            tool_gateway=self.tool_gateway,
            ollama_model="gemma4:4b-q4_K_M",
            timeout_seconds=2.0,
        )
        self.agent.start()

    def test_contract_declaration(self) -> None:
        """Verify strict AgentContract compliance."""
        contract = self.agent.contract
        self.assertEqual(contract.agent_name, "reasoning_agent")
        self.assertEqual(contract.cognitive_layer, CognitiveLayer.REASONING)
        self.assertIn(MessageType.HYPOTHESIS, contract.messages_published)
        self.assertIn(MessageType.OBSERVATION, contract.messages_subscribed)

    def test_fusion_prompt_construction(self) -> None:
        """Verify agronomic prompt contains microclimate data."""
        prompt = self.agent.construct_fusion_prompt(
            predicted_class="Late_blight",
            confidence=0.88,
            temperature_c=21.5,
            humidity=92.0,
            soil_moisture=65.0,
            soil_ph=6.2,
            soil_temp_c=19.0,
            light_lux=45.0,
        )
        self.assertIn("Late_blight", prompt)
        self.assertIn("88.0%", prompt)
        self.assertIn("21.5°C", prompt)
        self.assertIn("92.0%", prompt)
        self.assertIn("65.0%", prompt)

    def test_reasoning_process_observation(self) -> None:
        """Verify agent receives OBSERVATION and emits HYPOTHESIS."""
        obs_payload = ObservationPayload(
            observation_id="obs_test_101",
            source_sensors=["humidity_sensor", "temp_sensor", "camera"],
            target_zone="zone_1",
            observation_time=datetime.now(timezone.utc).isoformat(),
            measurements={
                "environment_temperature_c": 22.0,
                "environment_humidity": 90.0,
                "soil_moisture": 62.0,
                "soil_ph": 6.1,
                "vision_confidence": 0.85,
            },
        )
        obs_msg = create_message(
            source="perception_agent",
            destination="BROADCAST",
            message_type=MessageType.OBSERVATION,
            payload=obs_payload,
            metadata={"predicted_class": "Late_blight", "confidence": 0.85},
        )

        resp = self.agent.process(obs_msg)
        self.assertIsNotNone(resp)
        self.assertEqual(resp.message_type, MessageType.HYPOTHESIS)
        self.assertIsInstance(resp.payload, HypothesisPayload)

        hyp: HypothesisPayload = resp.payload
        self.assertEqual(hyp.suspected_cause, "Late_blight")
        self.assertEqual(hyp.prior_probability, 0.85)
        self.assertGreater(hyp.likelihood_ratio, 1.0)  # High humidity (90%) favors Late Blight
        self.assertIn("obs_test_101", hyp.associated_evidence_ids)

        meta = resp.metadata
        self.assertTrue(meta["etiology_supported"])
        self.assertIn("recommendations", meta)
        self.assertGreater(len(meta["recommendations"]), 0)


# =====================================================================
# Test 5: End-to-End Multi-Agent Integration Pipeline
# =====================================================================

class TestEndToEndPerceptionReasoningPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.bus = MessageBus(MessageBusConfig(enable_tracing=True))
        self.tool_reg = ToolRegistry()
        self.tool_reg.register(MockTool("sensor_read"))
        self.tool_reg.register(MockTool("camera_capture"))
        self.tool_reg.register(MockTool("llm_infer"))
        self.tool_reg.register(MockTool("agronomy_rules"))

        self.p_mem_gw = MemoryGateway({}, {"working": MemoryAccess.WRITE})
        self.p_tool_gw = ToolGateway(self.tool_reg, {"sensor_read", "camera_capture"})

        self.r_mem_gw = MemoryGateway({}, {"working": MemoryAccess.READ_WRITE, "semantic": MemoryAccess.READ})
        self.r_tool_gw = ToolGateway(self.tool_reg, {"llm_infer", "agronomy_rules"})

        self.streamer = IoTStreamer(loop=True)
        self.skill = TomatoDiagnosisSkill(auto_load=True)

        self.perception_agent = PerceptionAgent(
            message_bus=self.bus,
            memory_gateway=self.p_mem_gw,
            tool_gateway=self.p_tool_gw,
            iot_streamer=self.streamer,
            diagnosis_skill=self.skill,
            target_zone="greenhouse_main",
        )

        self.reasoning_agent = ReasoningAgent(
            message_bus=self.bus,
            memory_gateway=self.r_mem_gw,
            tool_gateway=self.r_tool_gw,
            ollama_model="gemma4:4b-q4_K_M",
            timeout_seconds=2.0,
        )

        self.perception_agent.start()
        self.reasoning_agent.start()

    def test_full_pub_sub_loop(self) -> None:
        """
        Verify end-to-end flow:
        Perception Agent perceives -> Publishes OBSERVATION -> MessageBus triggers Reasoning Agent -> Reasoning Agent publishes HYPOTHESIS.
        """
        observations_received: List[ACAMessage] = []
        hypotheses_received: List[ACAMessage] = []

        self.bus.subscribe(MessageType.OBSERVATION, lambda m: observations_received.append(m))
        self.bus.subscribe(MessageType.HYPOTHESIS, lambda m: hypotheses_received.append(m))

        dummy_leaf = torch.randn(1, 3, 224, 224)

        # Trigger perception cycle
        obs_msg = self.perception_agent.perceive(image_path=dummy_leaf)

        # Reasoning agent automatically receives OBSERVATION and processes it
        self.assertEqual(len(observations_received), 1)
        self.assertEqual(len(hypotheses_received), 1)

        hyp_msg = hypotheses_received[0]
        self.assertEqual(hyp_msg.message_type, MessageType.HYPOTHESIS)
        self.assertEqual(hyp_msg.source, "reasoning_agent")
        self.assertEqual(hyp_msg.payload.associated_evidence_ids[0], obs_msg.payload.observation_id)

        print(f"\n[End-to-End Success] Class: {hyp_msg.payload.suspected_cause} | Prior: {hyp_msg.payload.prior_probability} | Likelihood: {hyp_msg.payload.likelihood_ratio} | Conf: {hyp_msg.confidence}")


if __name__ == "__main__":
    unittest.main()
