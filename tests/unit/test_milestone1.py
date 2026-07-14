"""
ACA Milestone 1 — Unit Tests
=============================

Comprehensive tests covering core infrastructure:
    - Configuration system
    - Message schemas and validation
    - MessageBus (pub/sub, priority queue, history)
    - Memory subsystem (Working, Episodic, Semantic, Farm)
    - Workflow Engine (task DAG, status tracking, replanning)
    - Scheduler (runtime assignment, policy, queues)
    - BaseAgent contracts (memory gateway, tool gateway, violations)
    - Tool and Skill registries
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional

# ── Configuration ─────────────────────────────────────────────────────
from aca.config import (
    ACAConfig,
    LoggingConfig,
    MemoryConfig,
    MessageBusConfig,
    SchedulerConfig,
)
from aca.logging_config import setup_logging

# ── Schemas ───────────────────────────────────────────────────────────
from aca.orchestration.schemas import (
    ACAMessage,
    BeliefPayload,
    DecisionPayload,
    EvidencePayload,
    ExplanationPayload,
    FeedbackPayload,
    GoalPayload,
    HypothesisPayload,
    MessageType,
    MissionPayload,
    ObservationPayload,
    TaskPayload,
    create_message,
)

# ── MessageBus ────────────────────────────────────────────────────────
from aca.orchestration.message_bus import MessageBus

# ── Memory ────────────────────────────────────────────────────────────
from aca.memory.working_memory import WorkingMemory
from aca.memory.episodic_memory import Episode, EpisodicMemory
from aca.memory.semantic_memory import SemanticMemory
from aca.memory.farm_memory import FarmMemory

# ── Orchestration ─────────────────────────────────────────────────────
from aca.orchestration.workflow_engine import (
    TaskNode,
    TaskStatus,
    Workflow,
    WorkflowEngine,
    WorkflowStatus,
)
from aca.orchestration.scheduler import (
    DefaultSchedulingPolicy,
    RuntimeTarget,
    Scheduler,
)
from aca.orchestration.supervisor import Supervisor

# ── Tools ─────────────────────────────────────────────────────────────
from aca.tools.base_tool import BaseTool, ToolParameter, ToolResult, ToolSchema
from aca.tools.registry import ToolRegistry

# ── Skills ────────────────────────────────────────────────────────────
from aca.skills.base_skill import BaseSkill, SkillParameter, SkillResult, SkillSchema
from aca.skills.registry import SkillRegistry

# ── Agents ────────────────────────────────────────────────────────────
from aca.agents.base_agent import (
    AgentContract,
    AgentContractViolation,
    BaseAgent,
    CognitiveLayer,
    MemoryAccess,
    MemoryGateway,
    ToolGateway,
)

# Initialise logging once
setup_logging(LoggingConfig(level="WARNING"))


# =====================================================================
# Test: Configuration
# =====================================================================

class TestConfiguration(unittest.TestCase):
    """Tests for the ACA configuration system."""

    def test_default_config(self):
        cfg = ACAConfig()
        self.assertEqual(cfg.environment, "development")
        self.assertEqual(cfg.message_bus.max_queue_size, 10_000)
        self.assertIsInstance(cfg.memory, MemoryConfig)

    def test_config_is_frozen(self):
        cfg = ACAConfig()
        with self.assertRaises(Exception):
            cfg.environment = "production"  # type: ignore

    def test_load_defaults(self):
        cfg = ACAConfig.load()
        self.assertEqual(cfg.environment, "development")


# =====================================================================
# Test: Message Schemas
# =====================================================================

class TestMessageSchemas(unittest.TestCase):
    """Tests for ACA message schemas and validation."""

    def test_create_mission_message(self):
        msg = create_message(
            source="supervisor",
            destination="BROADCAST",
            message_type=MessageType.MISSION,
            payload=MissionPayload(
                mission_id="m1",
                objective="Optimize yield",
                constraints={"max_water": 1000},
            ),
        )
        self.assertEqual(msg.message_type, MessageType.MISSION)
        self.assertEqual(msg.source, "supervisor")
        self.assertIsInstance(msg.payload, MissionPayload)
        self.assertEqual(msg.payload.mission_id, "m1")

    def test_create_all_message_types(self):
        payloads = {
            MessageType.MISSION: MissionPayload(mission_id="m1", objective="test"),
            MessageType.GOAL: GoalPayload(goal_id="g1", parent_mission_id="m1", target_metric="moisture"),
            MessageType.TASK: TaskPayload(task_id="t1", goal_id="g1", skill_required="irrigation"),
            MessageType.OBSERVATION: ObservationPayload(observation_id="o1"),
            MessageType.EVIDENCE: EvidencePayload(evidence_id="e1"),
            MessageType.HYPOTHESIS: HypothesisPayload(hypothesis_id="h1"),
            MessageType.BELIEF: BeliefPayload(),
            MessageType.DECISION: DecisionPayload(decision_id="d1"),
            MessageType.EXPLANATION: ExplanationPayload(explanation_id="ex1", decision_id="d1"),
            MessageType.FEEDBACK: FeedbackPayload(action_id="a1"),
        }
        for mt, payload in payloads.items():
            msg = create_message("src", "dst", mt, payload)
            self.assertEqual(msg.message_type, mt)
            msg.validate()  # should not raise

    def test_payload_type_mismatch_raises(self):
        msg = ACAMessage(
            uuid="x",
            timestamp="t",
            source="s",
            destination="d",
            message_type=MessageType.MISSION,
            confidence=1.0,
            priority=3,
            payload=GoalPayload(goal_id="g", parent_mission_id="m", target_metric="x"),
        )
        with self.assertRaises(TypeError):
            msg.validate()

    def test_confidence_out_of_range(self):
        with self.assertRaises(ValueError):
            create_message(
                "s", "d", MessageType.BELIEF,
                BeliefPayload(), confidence=1.5,
            )

    def test_priority_out_of_range(self):
        with self.assertRaises(ValueError):
            create_message(
                "s", "d", MessageType.BELIEF,
                BeliefPayload(), priority=0,
            )


# =====================================================================
# Test: MessageBus
# =====================================================================

class TestMessageBus(unittest.TestCase):
    """Tests for the pub/sub message bus."""

    def setUp(self):
        self.bus = MessageBus(MessageBusConfig())
        self.received: List[ACAMessage] = []

    def _handler(self, msg: ACAMessage):
        self.received.append(msg)

    def test_subscribe_and_publish(self):
        self.bus.subscribe(MessageType.OBSERVATION, self._handler)
        msg = create_message(
            "sensor", "bus", MessageType.OBSERVATION,
            ObservationPayload(observation_id="o1"),
        )
        count = self.bus.publish(msg)
        self.assertEqual(count, 1)
        self.assertEqual(len(self.received), 1)
        self.assertEqual(self.received[0].uuid, msg.uuid)

    def test_wildcard_subscriber(self):
        self.bus.subscribe_all(self._handler)
        msg = create_message(
            "src", "dst", MessageType.MISSION,
            MissionPayload(mission_id="m1", objective="test"),
        )
        self.bus.publish(msg)
        self.assertEqual(len(self.received), 1)

    def test_unsubscribe(self):
        self.bus.subscribe(MessageType.EVIDENCE, self._handler)
        self.bus.unsubscribe(MessageType.EVIDENCE, self._handler)
        msg = create_message(
            "src", "dst", MessageType.EVIDENCE,
            EvidencePayload(evidence_id="e1"),
        )
        count = self.bus.publish(msg)
        self.assertEqual(count, 0)

    def test_history(self):
        msg = create_message(
            "src", "dst", MessageType.BELIEF,
            BeliefPayload(),
        )
        self.bus.publish(msg)
        history = self.bus.get_history(MessageType.BELIEF)
        self.assertEqual(len(history), 1)

    def test_enqueue_drain_priority_order(self):
        low = create_message("s", "d", MessageType.TASK,
                             TaskPayload(task_id="t1", goal_id="g", skill_required="a"), priority=1)
        high = create_message("s", "d", MessageType.TASK,
                              TaskPayload(task_id="t2", goal_id="g", skill_required="b"), priority=5)
        self.bus.enqueue(low)
        self.bus.enqueue(high)
        drained = self.bus.drain(MessageType.TASK)
        self.assertEqual(len(drained), 2)
        self.assertEqual(drained[0].priority, 5)
        self.assertEqual(drained[1].priority, 1)

    def test_subscriber_count(self):
        self.bus.subscribe(MessageType.FEEDBACK, self._handler)
        self.bus.subscribe(MessageType.FEEDBACK, self._handler)
        counts = self.bus.subscriber_count
        self.assertEqual(counts["FEEDBACK"], 2)


# =====================================================================
# Test: Working Memory
# =====================================================================

class TestWorkingMemory(unittest.TestCase):
    """Tests for transient working memory."""

    def setUp(self):
        self.wm = WorkingMemory(MemoryConfig(working_memory_capacity=5))

    def test_store_and_retrieve(self):
        self.wm.store("goals", "g1", {"metric": "moisture"})
        val = self.wm.retrieve("goals", "g1")
        self.assertEqual(val["metric"], "moisture")

    def test_missing_key_returns_none(self):
        self.assertIsNone(self.wm.retrieve("goals", "nonexistent"))

    def test_capacity_eviction(self):
        for i in range(7):
            self.wm.store("ns", f"k{i}", i)
        self.assertLessEqual(self.wm.total_entries, 5)

    def test_remove(self):
        self.wm.store("ns", "k1", 1)
        self.assertTrue(self.wm.remove("ns", "k1"))
        self.assertIsNone(self.wm.retrieve("ns", "k1"))

    def test_clear_namespace(self):
        self.wm.store("ns", "a", 1)
        self.wm.store("ns", "b", 2)
        self.wm.clear_namespace("ns")
        self.assertEqual(self.wm.total_entries, 0)

    def test_list_namespace(self):
        self.wm.store("goals", "g1", 1)
        self.wm.store("goals", "g2", 2)
        self.assertEqual(sorted(self.wm.list_namespace("goals")), ["g1", "g2"])


# =====================================================================
# Test: Episodic Memory
# =====================================================================

class TestEpisodicMemory(unittest.TestCase):
    """Tests for append-only episodic memory."""

    def setUp(self):
        self.em = EpisodicMemory(MemoryConfig())

    def _make_episode(self, eid="ep1", zone="field_1", assessment="SUCCESS"):
        return Episode(
            episode_id=eid,
            timestamp="2025-07-01T00:00:00Z",
            zone=zone,
            initial_state={"moisture": 0.30},
            planned_actions=[{"type": "irrigate"}],
            executed_actions=[{"type": "irrigate"}],
            resulting_state={"moisture": 0.45},
            outcome_assessment=assessment,
            yield_impact=0.1,
            tags=("drought_response",),
        )

    def test_commit_and_get(self):
        ep = self._make_episode()
        self.em.commit(ep)
        result = self.em.get("ep1")
        self.assertIsNotNone(result)
        self.assertEqual(result.zone, "field_1")

    def test_duplicate_raises(self):
        self.em.commit(self._make_episode())
        with self.assertRaises(ValueError):
            self.em.commit(self._make_episode())

    def test_query_by_zone(self):
        self.em.commit(self._make_episode("ep1", "field_1"))
        self.em.commit(self._make_episode("ep2", "field_2"))
        results = self.em.query(zone="field_1")
        self.assertEqual(len(results), 1)

    def test_query_by_assessment(self):
        self.em.commit(self._make_episode("ep1", assessment="SUCCESS"))
        self.em.commit(self._make_episode("ep2", assessment="FAILED"))
        results = self.em.query(assessment="FAILED")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].episode_id, "ep2")

    def test_query_by_tag(self):
        self.em.commit(self._make_episode())
        results = self.em.query(tag="drought_response")
        self.assertEqual(len(results), 1)

    def test_count(self):
        self.em.commit(self._make_episode("ep1"))
        self.em.commit(self._make_episode("ep2", zone="field_2"))
        self.assertEqual(self.em.count(), 2)


# =====================================================================
# Test: Semantic Memory
# =====================================================================

class TestSemanticMemory(unittest.TestCase):
    """Tests for domain-partitioned semantic memory."""

    def setUp(self):
        self.sm = SemanticMemory(MemoryConfig(semantic_readonly=False))

    def test_store_and_retrieve(self):
        self.sm.store("thresholds", "rice_water_min", 0.35)
        self.assertEqual(self.sm.retrieve("thresholds", "rice_water_min"), 0.35)

    def test_missing_returns_none(self):
        self.assertIsNone(self.sm.retrieve("nonexistent", "key"))

    def test_bulk_load(self):
        self.sm.load_from_dict({
            "diseases": {"blast": {"severity": "high"}},
            "thresholds": {"temp_max": 42},
        })
        self.assertEqual(self.sm.retrieve("thresholds", "temp_max"), 42)
        self.assertIn("diseases", self.sm.domains)

    def test_freeze_prevents_writes(self):
        self.sm.freeze()
        with self.assertRaises(RuntimeError):
            self.sm.store("domain", "key", "value")

    def test_remove(self):
        self.sm.store("d", "k", 1)
        self.assertTrue(self.sm.remove("d", "k"))
        self.assertIsNone(self.sm.retrieve("d", "k"))


# =====================================================================
# Test: Farm Memory
# =====================================================================

class TestFarmMemory(unittest.TestCase):
    """Tests for spatial-temporal farm memory."""

    def setUp(self):
        self.fm = FarmMemory(MemoryConfig())

    def test_register_and_get_zone(self):
        self.fm.register_zone("f1_a", {"area_ha": 2.5, "soil": "clay"})
        zone = self.fm.get_zone("f1_a")
        self.assertEqual(zone["area_ha"], 2.5)

    def test_register_sensor_in_zone(self):
        self.fm.register_zone("f1_a", {})
        self.fm.register_sensor("s1", "f1_a", "moisture", (12.0, 77.0))
        sensors = self.fm.get_sensors_in_zone("f1_a")
        self.assertEqual(len(sensors), 1)
        self.assertEqual(sensors[0]["sensor_type"], "moisture")

    def test_actuator_registration(self):
        self.fm.register_zone("f1_a", {})
        self.fm.register_actuator("valve_1", "f1_a", "irrigation_valve")
        actuators = self.fm.get_actuators_in_zone("f1_a")
        self.assertEqual(len(actuators), 1)

    def test_yield_history(self):
        self.fm.register_zone("f1_a", {})
        self.fm.record_yield("f1_a", "2024_kharif", 4500.0)
        self.fm.record_yield("f1_a", "2025_kharif", 4800.0)
        history = self.fm.get_yield_history("f1_a")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[1]["yield_kg_per_hectare"], 4800.0)

    def test_bulk_load(self):
        self.fm.load_from_dict({
            "zones": {"f1_a": {"area": 2.5}},
            "sensors": {"s1": {"zone_id": "f1_a", "sensor_type": "temp"}},
            "actuators": {"v1": {"zone_id": "f1_a", "actuator_type": "valve"}},
        })
        self.assertIn("f1_a", self.fm.list_zones())
        self.assertEqual(len(self.fm.get_sensors_in_zone("f1_a")), 1)


# =====================================================================
# Test: Workflow Engine
# =====================================================================

class TestWorkflowEngine(unittest.TestCase):
    """Tests for workflow lifecycle and task DAG management."""

    def setUp(self):
        self.bus = MessageBus(MessageBusConfig(enable_tracing=False))
        self.engine = WorkflowEngine(self.bus)

    def test_create_workflow(self):
        wf = self.engine.create_workflow("mission_1")
        self.assertIsInstance(wf, Workflow)
        self.assertEqual(wf.status, WorkflowStatus.PENDING)

    def test_add_and_get_ready_tasks(self):
        wf = self.engine.create_workflow("m1")
        t1 = TaskNode(task_id="t1", goal_id="g1", skill_required="irrigate")
        t2 = TaskNode(task_id="t2", goal_id="g1", skill_required="spray",
                      depends_on={"t1"})
        self.engine.add_task(wf.workflow_id, t1)
        self.engine.add_task(wf.workflow_id, t2)
        ready = self.engine.get_ready_tasks(wf.workflow_id)
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].task_id, "t1")

    def test_dependency_resolution(self):
        wf = self.engine.create_workflow("m1")
        t1 = TaskNode(task_id="t1", goal_id="g1", skill_required="a")
        t2 = TaskNode(task_id="t2", goal_id="g1", skill_required="b",
                      depends_on={"t1"})
        self.engine.add_task(wf.workflow_id, t1)
        self.engine.add_task(wf.workflow_id, t2)

        self.engine.update_task_status(wf.workflow_id, "t1", TaskStatus.COMPLETED)
        ready = self.engine.get_ready_tasks(wf.workflow_id)
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].task_id, "t2")

    def test_workflow_completion(self):
        wf = self.engine.create_workflow("m1")
        t1 = TaskNode(task_id="t1", goal_id="g1", skill_required="a")
        self.engine.add_task(wf.workflow_id, t1)
        self.engine.update_task_status(wf.workflow_id, "t1", TaskStatus.COMPLETED)
        self.assertEqual(wf.status, WorkflowStatus.COMPLETED)

    def test_workflow_failure(self):
        wf = self.engine.create_workflow("m1")
        t1 = TaskNode(task_id="t1", goal_id="g1", skill_required="a")
        self.engine.add_task(wf.workflow_id, t1)
        self.engine.update_task_status(wf.workflow_id, "t1", TaskStatus.FAILED)
        self.assertEqual(wf.status, WorkflowStatus.FAILED)

    def test_replan(self):
        wf = self.engine.create_workflow("m1")
        self.engine.trigger_replan(wf.workflow_id)
        self.assertEqual(wf.status, WorkflowStatus.REPLANNING)


# =====================================================================
# Test: Scheduler
# =====================================================================

class TestScheduler(unittest.TestCase):
    """Tests for runtime assignment and queue management."""

    def setUp(self):
        self.scheduler = Scheduler(SchedulerConfig())

    def test_schedule_edge_by_default(self):
        task = TaskPayload(task_id="t1", goal_id="g1", skill_required="irrigation")
        scheduled = self.scheduler.schedule(task)
        self.assertEqual(scheduled.runtime, RuntimeTarget.EDGE)

    def test_schedule_cloud_for_heavy_skill(self):
        task = TaskPayload(task_id="t1", goal_id="g1", skill_required="yield_estimation")
        scheduled = self.scheduler.schedule(task)
        self.assertEqual(scheduled.runtime, RuntimeTarget.CLOUD)

    def test_pop_highest_priority(self):
        t1 = TaskPayload(task_id="t1", goal_id="g1", skill_required="a")
        t2 = TaskPayload(task_id="t2", goal_id="g1", skill_required="b")
        self.scheduler.schedule(t1, priority=2)
        self.scheduler.schedule(t2, priority=5)
        popped = self.scheduler.pop_next(RuntimeTarget.EDGE)
        self.assertIsNotNone(popped)
        self.assertEqual(popped.task_id, "t2")

    def test_queue_sizes(self):
        t = TaskPayload(task_id="t1", goal_id="g1", skill_required="a")
        self.scheduler.schedule(t)
        sizes = self.scheduler.queue_sizes
        self.assertEqual(sizes.get("EDGE", 0), 1)

    def test_remove_task(self):
        t = TaskPayload(task_id="t1", goal_id="g1", skill_required="a")
        self.scheduler.schedule(t)
        self.assertTrue(self.scheduler.remove_task("t1"))
        self.assertEqual(self.scheduler.total_pending, 0)


# =====================================================================
# Test: Tool Registry
# =====================================================================

class _MockTool(BaseTool):
    """A minimal mock tool for testing."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="mock_sensor",
            description="Mock sensor tool",
            parameters=[
                ToolParameter(name="sensor_id", description="ID of sensor"),
            ],
            returns="float",
        )

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, data=0.42)


class TestToolRegistry(unittest.TestCase):
    """Tests for tool registration, validation, and invocation."""

    def setUp(self):
        self.registry = ToolRegistry()

    def test_register_and_list(self):
        self.registry.register(_MockTool())
        self.assertIn("mock_sensor", self.registry.list_tools())

    def test_duplicate_registration_raises(self):
        self.registry.register(_MockTool())
        with self.assertRaises(ValueError):
            self.registry.register(_MockTool())

    def test_invoke_success(self):
        self.registry.register(_MockTool())
        result = self.registry.invoke("mock_sensor", sensor_id="s1")
        self.assertTrue(result.success)
        self.assertEqual(result.data, 0.42)

    def test_invoke_missing_tool(self):
        result = self.registry.invoke("nonexistent")
        self.assertFalse(result.success)

    def test_invoke_missing_param(self):
        self.registry.register(_MockTool())
        result = self.registry.invoke("mock_sensor")  # missing sensor_id
        self.assertFalse(result.success)

    def test_get_schemas(self):
        self.registry.register(_MockTool())
        schemas = self.registry.get_schemas()
        self.assertIn("mock_sensor", schemas)


# =====================================================================
# Test: Skill Registry
# =====================================================================

class _MockSkill(BaseSkill):
    """A minimal mock skill for testing."""

    @property
    def schema(self) -> SkillSchema:
        return SkillSchema(
            name="mock_detection",
            description="Mock disease detection",
            parameters=[
                SkillParameter(name="zone", description="Target zone"),
            ],
            tools_required=["mock_sensor"],
        )

    def execute(self, tool_registry: ToolRegistry, **kwargs) -> SkillResult:
        result = tool_registry.invoke("mock_sensor", sensor_id="auto")
        return SkillResult(
            success=True,
            data={"detection": "healthy"},
            tools_invoked=["mock_sensor"],
        )


class TestSkillRegistry(unittest.TestCase):
    """Tests for skill registration, validation, and invocation."""

    def setUp(self):
        self.tool_reg = ToolRegistry()
        self.tool_reg.register(_MockTool())
        self.skill_reg = SkillRegistry(self.tool_reg)

    def test_register_and_list(self):
        self.skill_reg.register(_MockSkill())
        self.assertIn("mock_detection", self.skill_reg.list_skills())

    def test_invoke_success(self):
        self.skill_reg.register(_MockSkill())
        result = self.skill_reg.invoke("mock_detection", zone="field_1")
        self.assertTrue(result.success)

    def test_invoke_missing_tool(self):
        empty_tool_reg = ToolRegistry()
        skill_reg = SkillRegistry(empty_tool_reg)
        skill_reg.register(_MockSkill())
        result = skill_reg.invoke("mock_detection", zone="field_1")
        self.assertFalse(result.success)
        self.assertIn("Missing required tools", result.error)

    def test_invoke_missing_param(self):
        self.skill_reg.register(_MockSkill())
        result = self.skill_reg.invoke("mock_detection")  # missing zone
        self.assertFalse(result.success)

    def test_get_skills_for_tool(self):
        self.skill_reg.register(_MockSkill())
        skills = self.skill_reg.get_skills_for_tool("mock_sensor")
        self.assertIn("mock_detection", skills)


# =====================================================================
# Test: BaseAgent Contracts
# =====================================================================

class _TestAgent(BaseAgent):
    """Concrete test agent for contract enforcement testing."""

    @property
    def contract(self) -> AgentContract:
        return AgentContract(
            agent_name="test_agent",
            purpose="Unit testing",
            cognitive_layer=CognitiveLayer.REASONING,
            inputs=["observations"],
            outputs=["evidence"],
            memory_permissions={
                "working": MemoryAccess.READ_WRITE,
                "semantic": MemoryAccess.READ,
            },
            tools_allowed={"mock_sensor"},
            messages_subscribed=[MessageType.OBSERVATION],
            messages_published=[MessageType.EVIDENCE],
        )

    def process(self, message: ACAMessage) -> Optional[ACAMessage]:
        return None


class TestBaseAgent(unittest.TestCase):
    """Tests for agent contract enforcement."""

    def setUp(self):
        self.bus = MessageBus(MessageBusConfig(enable_tracing=False))
        self.tool_reg = ToolRegistry()
        self.tool_reg.register(_MockTool())

        wm = WorkingMemory(MemoryConfig())
        sm = SemanticMemory(MemoryConfig(semantic_readonly=False))
        memories = {"working": wm, "semantic": sm}
        permissions = {
            "working": MemoryAccess.READ_WRITE,
            "semantic": MemoryAccess.READ,
        }

        self.mem_gw = MemoryGateway(memories, permissions)
        self.tool_gw = ToolGateway(self.tool_reg, {"mock_sensor"})
        self.agent = _TestAgent(self.bus, self.mem_gw, self.tool_gw)

    def test_agent_contract_name(self):
        self.assertEqual(self.agent.contract.agent_name, "test_agent")

    def test_memory_read_allowed(self):
        module = self.mem_gw.get_module("working", MemoryAccess.READ)
        self.assertIsInstance(module, WorkingMemory)

    def test_memory_write_on_readonly_raises(self):
        with self.assertRaises(AgentContractViolation):
            self.mem_gw.get_module("semantic", MemoryAccess.WRITE)

    def test_memory_unregistered_raises(self):
        with self.assertRaises(AgentContractViolation):
            self.mem_gw.get_module("episodic", MemoryAccess.READ)

    def test_tool_allowed(self):
        result = self.tool_gw.invoke("mock_sensor", sensor_id="s1")
        self.assertTrue(result.success)

    def test_tool_not_allowed_raises(self):
        with self.assertRaises(AgentContractViolation):
            self.tool_gw.invoke("forbidden_tool")

    def test_publish_allowed_type(self):
        self.agent.start()
        msg = create_message(
            "test_agent", "BROADCAST", MessageType.EVIDENCE,
            EvidencePayload(evidence_id="e1"),
        )
        count = self.agent.publish(msg)
        self.assertGreaterEqual(count, 0)

    def test_publish_disallowed_type_raises(self):
        self.agent.start()
        msg = create_message(
            "test_agent", "BROADCAST", MessageType.DECISION,
            DecisionPayload(decision_id="d1"),
        )
        with self.assertRaises(AgentContractViolation):
            self.agent.publish(msg)

    def test_lifecycle(self):
        self.assertFalse(self.agent.is_active)
        self.agent.start()
        self.assertTrue(self.agent.is_active)
        self.agent.stop()
        self.assertFalse(self.agent.is_active)


# =====================================================================
# Test: Supervisor
# =====================================================================

class TestSupervisor(unittest.TestCase):
    """Tests for the supervisor mission management."""

    def setUp(self):
        self.bus = MessageBus(MessageBusConfig(enable_tracing=False))
        self.engine = WorkflowEngine(self.bus)
        self.supervisor = Supervisor(self.bus, self.engine)

    def test_submit_mission(self):
        mid = self.supervisor.submit_mission("Optimize yield in field 1")
        self.assertIsNotNone(mid)
        status = self.supervisor.get_mission_status(mid)
        self.assertEqual(status["status"], "SUBMITTED")

    def test_list_missions(self):
        self.supervisor.submit_mission("Mission A")
        self.supervisor.submit_mission("Mission B")
        missions = self.supervisor.list_missions()
        self.assertEqual(len(missions), 2)


# =====================================================================
# Run
# =====================================================================

if __name__ == "__main__":
    unittest.main()
