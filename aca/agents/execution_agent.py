"""
ACA Execution Agent — Actuator Dispatch & Closed-Loop Feedback
===============================================================

Implements the Execution Agent responsible for dispatching physical and
phytosanitary interventions via the ``ToolGateway`` and publishing
closed-loop ``FeedbackPayload`` messages to the ``MessageBus``.

Architectural Guarantees:
    - Inherits from ``BaseAgent`` and conforms to ``AgentContract``.
    - Subscribes to ``MessageType.DECISION`` and emits ``MessageType.FEEDBACK``.
    - Enforces full causal traceability: links ``action_id`` back to the
      originating ``decision_id`` and `justification_ids`.
    - Evaluates expected vs actual execution outcomes, computing deviation
      and qualitative assessments for Digital Twin & Memory updates.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from aca.agents.base_agent import (
    AgentContract,
    BaseAgent,
    CognitiveLayer,
    MemoryAccess,
    MemoryGateway,
    ToolGateway,
)
from aca.logging_config import get_logger
from aca.orchestration.message_bus import MessageBus
from aca.orchestration.schemas import (
    ACAMessage,
    DecisionPayload,
    FeedbackPayload,
    MessageType,
    create_message,
)
from aca.tools.base_tool import ToolResult

logger = get_logger("agents.execution")


class ExecutionAgent(BaseAgent):
    """
    Execution Agent for the Agricultural Cognitive Architecture.

    Ingests committed ``DecisionPayload`` messages, dispatches the required
    physical actuator tools (e.g. irrigation valves, treatment alerts) via
    the ``ToolGateway``, and publishes a ``FeedbackPayload`` recording the
    empirical execution result.

    Args:
        message_bus: Central ACA pub/sub message broker.
        memory_gateway: Permission-gated memory access proxy.
        tool_gateway: Permission-gated tool invocation proxy.
    """

    def __init__(
        self,
        message_bus: MessageBus,
        memory_gateway: MemoryGateway,
        tool_gateway: ToolGateway,
    ) -> None:
        super(ExecutionAgent, self).__init__(
            message_bus=message_bus,
            memory_gateway=memory_gateway,
            tool_gateway=tool_gateway,
        )

    @property
    def contract(self) -> AgentContract:
        """Formal contract specification for the Execution Agent."""
        return AgentContract(
            agent_name="execution_agent",
            purpose="Execute committed intervention decisions by dispatching actuator tools and publishing closed-loop feedback",
            cognitive_layer=CognitiveLayer.EXECUTION,
            inputs=["decision_payload"],
            outputs=["feedback_payload"],
            memory_permissions={
                "working": MemoryAccess.READ_WRITE,
                "episodic": MemoryAccess.WRITE,
            },
            tools_allowed={"irrigation_control", "treatment_alert"},
            latency_budget_ms=5000.0,
            failure_modes={
                "tool_execution_failed": "Record tool error in FeedbackPayload, set deviation=1.0, and alert supervisor",
                "invalid_decision": "Skip message and log contract error",
            },
            messages_published=[MessageType.FEEDBACK],
            messages_subscribed=[MessageType.DECISION],
            confidence_range=(0.0, 1.0),
        )

    def process(self, message: ACAMessage) -> Optional[ACAMessage]:
        """
        Process incoming DECISION messages from the Planning Agent.

        Executes the planned tool calls via ToolGateway and publishes a
        FeedbackPayload closing the cognitive loop.
        """
        if message.message_type != MessageType.DECISION:
            return None

        t0 = time.perf_counter()

        payload = message.payload
        if not isinstance(payload, DecisionPayload):
            logger.warning("Received invalid payload type for DECISION: %s", type(payload))
            return None

        params = payload.parameters or {}
        tool_calls: List[Dict[str, Any]] = params.get("tool_calls", [])
        target_zone = params.get("target_zone", "unknown_zone")
        justification_trace = params.get("justification_trace", payload.justification_ids)

        logger.info(
            "ExecutionAgent executing decision [%s]: Action='%s', Tools=%d, TargetZone='%s'",
            payload.decision_id,
            payload.action_selected,
            len(tool_calls),
            target_zone,
        )

        executed_tools_data: List[Dict[str, Any]] = []
        success_count = 0
        error_count = 0
        errors: List[str] = []

        # 1. Execute Planned Tool Calls via ToolGateway
        for call_spec in tool_calls:
            tool_name = call_spec.get("tool_name", "")
            tool_params = call_spec.get("parameters", {})

            try:
                result: ToolResult = self.tool_gateway.invoke(tool_name, **tool_params)
                if result.success:
                    success_count += 1
                    executed_tools_data.append({
                        "tool_name": tool_name,
                        "success": True,
                        "data": result.data,
                    })
                else:
                    error_count += 1
                    errors.append(f"{tool_name}: {result.error}")
                    executed_tools_data.append({
                        "tool_name": tool_name,
                        "success": False,
                        "error": result.error,
                    })
            except Exception as exc:
                error_count += 1
                errors.append(f"{tool_name} exception: {str(exc)}")
                executed_tools_data.append({
                    "tool_name": tool_name,
                    "success": False,
                    "error": str(exc),
                })

        # 2. Evaluate Success & Compute Deviation
        total_calls = len(tool_calls)
        if total_calls == 0:
            assessment = "SUCCESS_NO_ACTION_NEEDED"
            deviation = 0.0
        elif success_count == total_calls:
            assessment = "SUCCESS"
            deviation = 0.0
        elif success_count > 0:
            assessment = "PARTIAL_SUCCESS"
            deviation = round(error_count / total_calls, 2)
        else:
            assessment = "FAILURE"
            deviation = 1.0

        # 3. Construct FeedbackPayload
        expected_outcome = {
            "action_selected": payload.action_selected,
            "target_zone": target_zone,
            "tools_planned": total_calls,
        }

        actual_outcome = {
            "tools_executed": executed_tools_data,
            "success_count": success_count,
            "error_count": error_count,
            "errors": errors,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

        feedback_payload = FeedbackPayload(
            action_id=payload.decision_id,
            expected_outcome=expected_outcome,
            actual_outcome=actual_outcome,
            deviation=deviation,
            assessment=assessment,
        )

        # 4. Build ACAMessage Envelope with Full Causality Trace
        lat_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        out_meta = {
            "decision_id": payload.decision_id,
            "justification_trace": justification_trace,
            "target_zone": target_zone,
            "assessment": assessment,
            "deviation": deviation,
            "execution_latency_ms": lat_ms,
        }

        feedback_message = create_message(
            source=self.contract.agent_name,
            destination="BROADCAST",
            message_type=MessageType.FEEDBACK,
            payload=feedback_payload,
            confidence=1.0 if deviation == 0.0 else 0.5,
            priority=message.priority,
            metadata=out_meta,
        )

        logger.info(
            "ExecutionAgent published FEEDBACK for decision [%s]: Assessment='%s', Deviation=%.2f (Trace: %s)",
            payload.decision_id,
            assessment,
            deviation,
            justification_trace,
        )
        return feedback_message
