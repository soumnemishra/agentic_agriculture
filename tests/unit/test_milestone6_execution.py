"""
ACA Milestone 6 — Unit & Integration Tests (Decision & Execution)
=================================================================

Comprehensive tests covering:
    - Task 1: Actuator Tools (IrrigationControlTool, TreatmentAlertTool)
    - Task 2: Planning Agent (Hypothesis-to-Decision transformation)
    - Task 3: Execution Agent (Decision-to-Feedback execution loop)
    - Task 4: Complete Closed-Loop Multi-Agent Arc:
      Perception -> Reasoning -> Planning -> Execution -> Feedback
"""

from __future__ import annotations

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
from aca.agents.execution_agent import ExecutionAgent
from aca.agents.perception_agent import PerceptionAgent
from aca.agents.planning_agent import PlanningAgent
from aca.agents.reasoning_agent import ReasoningAgent
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
from aca.skills.tomato_diagnosis_skill import TomatoDiagnosisSkill
from aca.tools.actuator_tools import (
    IrrigationControlTool,
    TreatmentAlertTool,
)
from aca.tools.base_tool import BaseTool, ToolResult, ToolSchema
from aca.tools.registry import ToolRegistry
from simulation.telemetry_streamer import IoTStreamer

# Initialise logging once
setup_logging(ACAConfig().logging)


class MockAgronomyRuleTool(BaseTool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(name="agronomy_rules", description="Mock agronomy rules")

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, data={"rules_applied": True})


# =====================================================================
# Test 1: Actuator Tools
# =====================================================================

class TestActuatorTools(unittest.TestCase):
    def setUp(self) -> None:
        self.irr_tool = IrrigationControlTool()
        self.alert_tool = TreatmentAlertTool()

    def test_irrigation_control_valid_actions(self) -> None:
        """Verify all valid irrigation actions execute successfully."""
        for action in ["decrease", "increase", "stop", "start", "maintain"]:
            res = self.irr_tool.execute(action=action, zone="greenhouse_a", amount_litres=50.0, reason="test")
            self.assertTrue(res.success)
            data = res.data
            self.assertEqual(data["action"], action)
            self.assertEqual(data["zone"], "greenhouse_a")
            self.assertEqual(data["amount_litres"], 50.0)
            self.assertEqual(data["status"], "EXECUTED")

    def test_irrigation_control_invalid_parameters(self) -> None:
        """Verify invalid actions and missing parameters fail gracefully."""
        res_invalid = self.irr_tool.execute(action="flood_field", zone="zone_1")
        self.assertFalse(res_invalid.success)
        self.assertIn("Invalid action", res_invalid.error)

        res_no_zone = self.irr_tool.execute(action="stop", zone="")
        self.assertFalse(res_no_zone.success)

    def test_treatment_alert_valid_dispatch(self) -> None:
        """Verify treatment alert dispatch formatting and return schema."""
        res = self.alert_tool.execute(
            disease_name="Late_blight",
            treatment="Curative cymoxanil spray",
            urgency="CRITICAL",
            zone="zone_tomato_1",
            notes="Use protective respirator",
        )
        self.assertTrue(res.success)
        data = res.data
        self.assertEqual(data["disease_name"], "Late_blight")
        self.assertEqual(data["treatment"], "Curative cymoxanil spray")
        self.assertEqual(data["urgency"], "CRITICAL")
        self.assertEqual(data["zone"], "zone_tomato_1")
        self.assertIn("alert_id", data)

    def test_treatment_alert_missing_parameters(self) -> None:
        """Verify required parameter enforcement on treatment alert."""
        res_no_disease = self.alert_tool.execute(disease_name="", treatment="spray", zone="z1")
        self.assertFalse(res_no_disease.success)

        res_no_treatment = self.alert_tool.execute(disease_name="Early_blight", treatment="", zone="z1")
        self.assertFalse(res_no_treatment.success)


# =====================================================================
# Test 2: Planning Agent
# =====================================================================

class TestPlanningAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.bus = MessageBus(MessageBusConfig(enable_tracing=True))
        self.tool_reg = ToolRegistry()
        self.tool_reg.register(IrrigationControlTool())
        self.tool_reg.register(TreatmentAlertTool())
        self.tool_reg.register(MockAgronomyRuleTool())

        self.mem_gw = MemoryGateway({}, {"working": MemoryAccess.READ_WRITE, "semantic": MemoryAccess.READ})
        self.tool_gw = ToolGateway(self.tool_reg, {"irrigation_control", "treatment_alert", "agronomy_rules"})

        self.agent = PlanningAgent(
            message_bus=self.bus,
            memory_gateway=self.mem_gw,
            tool_gateway=self.tool_gw,
            ollama_model="gemma4:4b-q4_K_M",
            timeout_seconds=2.0,
        )
        self.agent.start()

    def test_contract_declaration(self) -> None:
        """Verify strict AgentContract compliance for PlanningAgent."""
        contract = self.agent.contract
        self.assertEqual(contract.agent_name, "planning_agent")
        self.assertEqual(contract.cognitive_layer, CognitiveLayer.PLANNING)
        self.assertIn(MessageType.HYPOTHESIS, contract.messages_subscribed)
        self.assertIn(MessageType.DECISION, contract.messages_published)

    def test_planning_process_fungal_hypothesis(self) -> None:
        """Verify planning on Late Blight hypothesis generates containment decision."""
        hyp = HypothesisPayload(
            hypothesis_id="hyp_test_001",
            associated_evidence_ids=["obs_test_001"],
            suspected_cause="Late_blight",
            prior_probability=0.88,
            likelihood_ratio=1.85,
        )
        msg = create_message(
            source="reasoning_agent",
            destination="BROADCAST",
            message_type=MessageType.HYPOTHESIS,
            payload=hyp,
            confidence=0.88,
            metadata={
                "target_zone": "bay_4",
                "environmental_context": {"temperature_c": 21.0, "humidity_percent": 92.0},
                "etiology_supported": True,
            },
        )

        dec_msg = self.agent.process(msg)
        self.assertIsNotNone(dec_msg)
        self.assertEqual(dec_msg.message_type, MessageType.DECISION)
        self.assertIsInstance(dec_msg.payload, DecisionPayload)

        payload: DecisionPayload = dec_msg.payload
        self.assertIn("hyp_test_001", payload.justification_ids)
        self.assertIn("obs_test_001", payload.justification_ids)
        self.assertEqual(payload.parameters["target_zone"], "bay_4")

        tool_calls = payload.parameters["tool_calls"]
        tool_names = [t["tool_name"] for t in tool_calls]
        self.assertIn("irrigation_control", tool_names)
        self.assertIn("treatment_alert", tool_names)

        # For Late Blight, irrigation must be decreased
        irr_call = [t for t in tool_calls if t["tool_name"] == "irrigation_control"][0]
        self.assertEqual(irr_call["parameters"]["action"], "decrease")

    def test_planning_process_mite_hypothesis(self) -> None:
        """Verify planning on Spider Mite hypothesis increases humidity/irrigation."""
        hyp = HypothesisPayload(
            hypothesis_id="hyp_test_002",
            associated_evidence_ids=["obs_test_002"],
            suspected_cause="Spider_mites Two-spotted_spider_mite",
            prior_probability=0.82,
            likelihood_ratio=1.9,
        )
        msg = create_message(
            source="reasoning_agent",
            destination="BROADCAST",
            message_type=MessageType.HYPOTHESIS,
            payload=hyp,
            metadata={"target_zone": "bay_2", "environmental_context": {}, "etiology_supported": True},
        )

        dec_msg = self.agent.process(msg)
        self.assertIsNotNone(dec_msg)
        tool_calls = dec_msg.payload.parameters["tool_calls"]
        irr_call = [t for t in tool_calls if t["tool_name"] == "irrigation_control"][0]
        self.assertEqual(irr_call["parameters"]["action"], "increase")


# =====================================================================
# Test 3: Execution Agent
# =====================================================================

class TestExecutionAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.bus = MessageBus(MessageBusConfig(enable_tracing=True))
        self.tool_reg = ToolRegistry()
        self.tool_reg.register(IrrigationControlTool())
        self.tool_reg.register(TreatmentAlertTool())

        self.mem_gw = MemoryGateway({}, {"working": MemoryAccess.READ_WRITE, "episodic": MemoryAccess.WRITE})
        self.tool_gw = ToolGateway(self.tool_reg, {"irrigation_control", "treatment_alert"})

        self.agent = ExecutionAgent(
            message_bus=self.bus,
            memory_gateway=self.mem_gw,
            tool_gateway=self.tool_gw,
        )
        self.agent.start()

    def test_contract_declaration(self) -> None:
        """Verify strict AgentContract compliance for ExecutionAgent."""
        contract = self.agent.contract
        self.assertEqual(contract.agent_name, "execution_agent")
        self.assertEqual(contract.cognitive_layer, CognitiveLayer.EXECUTION)
        self.assertIn(MessageType.DECISION, contract.messages_subscribed)
        self.assertIn(MessageType.FEEDBACK, contract.messages_published)

    def test_execution_process_decision_success(self) -> None:
        """Verify execution agent dispatches tools and publishes FeedbackPayload."""
        dec_payload = DecisionPayload(
            decision_id="dec_test_100",
            justification_ids=["hyp_test_001", "obs_test_001"],
            action_selected="FUNGAL_REMEDIATION",
            skill_name="actuator_dispatch",
            parameters={
                "tool_calls": [
                    {
                        "tool_name": "irrigation_control",
                        "parameters": {"action": "decrease", "zone": "bay_1", "reason": "prevent blight"},
                    },
                    {
                        "tool_name": "treatment_alert",
                        "parameters": {"disease_name": "Late_blight", "treatment": "Fungicide Spray", "zone": "bay_1", "urgency": "HIGH"},
                    },
                ],
                "target_zone": "bay_1",
                "justification_trace": ["hyp_test_001", "obs_test_001"],
            },
        )

        dec_msg = create_message(
            source="planning_agent",
            destination="BROADCAST",
            message_type=MessageType.DECISION,
            payload=dec_payload,
        )

        feedback_msg = self.agent.process(dec_msg)
        self.assertIsNotNone(feedback_msg)
        self.assertEqual(feedback_msg.message_type, MessageType.FEEDBACK)
        self.assertIsInstance(feedback_msg.payload, FeedbackPayload)

        fb: FeedbackPayload = feedback_msg.payload
        self.assertEqual(fb.action_id, "dec_test_100")
        self.assertEqual(fb.assessment, "SUCCESS")
        self.assertEqual(fb.deviation, 0.0)
        self.assertEqual(fb.actual_outcome["success_count"], 2)
        self.assertEqual(fb.actual_outcome["error_count"], 0)

        # Causality Trace check
        meta = feedback_msg.metadata
        self.assertEqual(meta["decision_id"], "dec_test_100")
        self.assertEqual(meta["justification_trace"], ["hyp_test_001", "obs_test_001"])


# =====================================================================
# Test 4: Complete Closed-Loop Multi-Agent Arc
# =====================================================================

class TestClosedLoopCognitiveArc(unittest.TestCase):
    def setUp(self) -> None:
        self.bus = MessageBus(MessageBusConfig(enable_tracing=True))
        self.tool_reg = ToolRegistry()
        self.tool_reg.register(IrrigationControlTool())
        self.tool_reg.register(TreatmentAlertTool())
        self.tool_reg.register(MockAgronomyRuleTool())

        # Gateways
        p_mem_gw = MemoryGateway({}, {"working": MemoryAccess.WRITE})
        p_tool_gw = ToolGateway(self.tool_reg, {"sensor_read", "camera_capture"})

        r_mem_gw = MemoryGateway({}, {"working": MemoryAccess.READ_WRITE, "semantic": MemoryAccess.READ})
        r_tool_gw = ToolGateway(self.tool_reg, {"llm_infer", "agronomy_rules"})

        pl_mem_gw = MemoryGateway({}, {"working": MemoryAccess.READ_WRITE, "semantic": MemoryAccess.READ})
        pl_tool_gw = ToolGateway(self.tool_reg, {"irrigation_control", "treatment_alert", "agronomy_rules"})

        ex_mem_gw = MemoryGateway({}, {"working": MemoryAccess.READ_WRITE, "episodic": MemoryAccess.WRITE})
        ex_tool_gw = ToolGateway(self.tool_reg, {"irrigation_control", "treatment_alert"})

        self.streamer = IoTStreamer(loop=True)
        self.skill = TomatoDiagnosisSkill(auto_load=True)

        # 4 Agents
        self.perception_agent = PerceptionAgent(self.bus, p_mem_gw, p_tool_gw, self.streamer, self.skill, "greenhouse_bay_1")
        self.reasoning_agent = ReasoningAgent(self.bus, r_mem_gw, r_tool_gw, "gemma4:4b-q4_K_M", timeout_seconds=2.0)
        self.planning_agent = PlanningAgent(self.bus, pl_mem_gw, pl_tool_gw, "gemma4:4b-q4_K_M", timeout_seconds=2.0)
        self.execution_agent = ExecutionAgent(self.bus, ex_mem_gw, ex_tool_gw)

        self.perception_agent.start()
        self.reasoning_agent.start()
        self.planning_agent.start()
        self.execution_agent.start()

    def test_full_closed_loop_execution_and_traceability(self) -> None:
        """
        Verify the full 5-stage cognitive arc:
        Perception -> OBSERVATION -> Reasoning -> HYPOTHESIS -> Planning -> DECISION -> Execution -> FEEDBACK.
        """
        observations: List[ACAMessage] = []
        hypotheses: List[ACAMessage] = []
        decisions: List[ACAMessage] = []
        feedbacks: List[ACAMessage] = []

        self.bus.subscribe(MessageType.OBSERVATION, lambda m: observations.append(m))
        self.bus.subscribe(MessageType.HYPOTHESIS, lambda m: hypotheses.append(m))
        self.bus.subscribe(MessageType.DECISION, lambda m: decisions.append(m))
        self.bus.subscribe(MessageType.FEEDBACK, lambda m: feedbacks.append(m))

        dummy_leaf = torch.randn(1, 3, 224, 224)

        # Step 1: Trigger Perception
        obs_msg = self.perception_agent.perceive(image_path=dummy_leaf)

        # Step 2..4: The MessageBus synchronously dispatches through the cognitive arc
        self.assertEqual(len(observations), 1, "Expected exactly 1 OBSERVATION message")
        self.assertEqual(len(hypotheses), 1, "Expected exactly 1 HYPOTHESIS message")
        self.assertEqual(len(decisions), 1, "Expected exactly 1 DECISION message")
        self.assertEqual(len(feedbacks), 1, "Expected exactly 1 FEEDBACK message")

        obs_id = obs_msg.payload.observation_id
        hyp_msg = hypotheses[0]
        dec_msg = decisions[0]
        fb_msg = feedbacks[0]

        # Traceability Verification Across the Entire Arc
        self.assertIn(obs_id, hyp_msg.payload.associated_evidence_ids, "Hypothesis must reference Observation ID")
        self.assertIn(hyp_msg.payload.hypothesis_id, dec_msg.payload.justification_ids, "Decision must reference Hypothesis ID")
        self.assertIn(obs_id, dec_msg.payload.justification_ids, "Decision must reference Observation ID")
        self.assertEqual(fb_msg.payload.action_id, dec_msg.payload.decision_id, "Feedback must reference Decision ID")
        self.assertEqual(fb_msg.payload.assessment, "SUCCESS", "Actuator execution must report SUCCESS")
        self.assertEqual(fb_msg.payload.deviation, 0.0, "Zero deviation expected on successful tool execution")

        print(
            f"\n[CLOSED LOOP VERIFIED]\n"
            f"  1. OBSERVATION: {obs_id} (Class: {obs_msg.metadata['predicted_class']})\n"
            f"  2. HYPOTHESIS:  {hyp_msg.payload.hypothesis_id} (Cause: {hyp_msg.payload.suspected_cause}, PostConf: {hyp_msg.confidence})\n"
            f"  3. DECISION:    {dec_msg.payload.decision_id} (Action: {dec_msg.payload.action_selected}, Tools: {dec_msg.metadata['tool_calls_count']})\n"
            f"  4. FEEDBACK:    {fb_msg.uuid} (ActionID: {fb_msg.payload.action_id}, Status: {fb_msg.payload.assessment})\n"
        )


if __name__ == "__main__":
    unittest.main()
