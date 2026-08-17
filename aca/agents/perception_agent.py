"""
ACA Perception Agent
====================

Ingests streaming IoT telemetry data and visual imagery, executes perception
skills (e.g., TomatoDiagnosisSkill), and publishes unified ACAMessages with
ObservationPayloads to the system MessageBus.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional
import uuid

from aca.agents.base_agent import AgentContract, BaseAgent, CognitiveLayer, MemoryAccess
from aca.logging_config import get_logger
from aca.orchestration.message_bus import MessageBus
from aca.orchestration.schemas import (
    ACAMessage,
    MessageType,
    ObservationPayload,
    create_message,
)
from aca.skills.tomato_diagnosis_skill import TomatoDiagnosisSkill
from simulation.telemetry_streamer import IoTStreamer

logger = get_logger("agents.perception")


class PerceptionAgent(BaseAgent):
    """
    Perception Agent responsible for environmental & plant health observations.

    Combines streaming IoT sensor readings with visual plant health diagnosis
    and emits structured ObservationPayload messages.

    Args:
        message_bus: System MessageBus instance.
        memory_gateway: Gated memory proxy.
        tool_gateway: Gated tool proxy.
        telemetry_streamer: Injected IoTStreamer dataset simulator.
        diagnosis_skill: Injected TomatoDiagnosisSkill vision pipeline.
    """

    def __init__(
        self,
        message_bus: MessageBus,
        memory_gateway: Any,
        tool_gateway: Any,
        telemetry_streamer: IoTStreamer,
        diagnosis_skill: TomatoDiagnosisSkill,
    ) -> None:
        self._streamer = telemetry_streamer
        self._diagnosis_skill = diagnosis_skill
        super().__init__(
            message_bus=message_bus,
            memory_gateway=memory_gateway,
            tool_gateway=tool_gateway,
        )

    @property
    def contract(self) -> AgentContract:
        return AgentContract(
            agent_name="perception_agent",
            purpose="Ingests IoT telemetry streams and vision sensor frames to generate unified environmental observations.",
            cognitive_layer=CognitiveLayer.PERCEPTION,
            inputs=["iot_telemetry_stream", "camera_frame_path"],
            outputs=["observation_payload"],
            memory_permissions={"working": MemoryAccess.READ_WRITE},
            tools_allowed={"sensor_read", "camera_capture"},
            latency_budget_ms=250.0,
            messages_subscribed=[MessageType.TASK],
            messages_published=[MessageType.OBSERVATION],
            confidence_range=(0.0, 1.0),
        )

    def perceive(self, image_path: str = "") -> ACAMessage:
        """
        Main execution trigger for a perception sample.

        Fetches current IoT telemetry tick and runs vision diagnosis, creating
        a unified ObservationPayload and publishing it to the MessageBus.

        Args:
            image_path: Optional path to tomato leaf image frame.

        Returns:
            The created ACAMessage containing the ObservationPayload.
        """
        # 1. Fetch current IoT telemetry tick
        telemetry_data = self._streamer.step()
        if not telemetry_data:
            logger.warning("IoTStreamer returned empty telemetry tick; using default baseline.")
            telemetry_data = {
                "Entry_id": 0,
                "Timestamp": datetime.now(timezone.utc).isoformat(),
                "Environment Temperature": 25.0,
                "Environment Humidity": 60.0,
                "Soil Moisture": 50.0,
                "Soil Temperature": 20.0,
                "Soil pH": 6.5,
                "Solar Panel Battery Voltage": 3.6,
                "Water TDS": 120.0,
            }

        # 2. Run vision diagnosis skill
        skill_res = self._diagnosis_skill.execute(image_path=image_path)
        vision_info = skill_res.data if skill_res.success else {
            "predicted_class": "unknown",
            "confidence": 0.0,
            "inference_time_ms": 0.0,
        }

        # 3. Assemble combined measurement dictionary
        measurements: Dict[str, Any] = {}
        for k, v in telemetry_data.items():
            if isinstance(v, (int, float)):
                measurements[k] = float(v)
            else:
                measurements[k] = str(v)

        measurements["vision_predicted_class"] = vision_info.get("predicted_class", "unknown")
        measurements["vision_confidence"] = float(vision_info.get("confidence", 0.0))
        measurements["vision_inference_time_ms"] = float(vision_info.get("inference_time_ms", 0.0))

        # 4. Construct ObservationPayload
        obs_payload = ObservationPayload(
            observation_id=f"obs_{uuid.uuid4().hex[:8]}",
            source_sensors=["iot_telemetry_node", "tomato_vision_camera"],
            target_zone="Zone_A_Tomatoes",
            observation_time=str(telemetry_data.get("Timestamp", datetime.now(timezone.utc).isoformat())),
            measurements={k: float(v) for k, v in measurements.items() if isinstance(v, (int, float))},
        )

        # Attach string metadata into metadata dict
        metadata = {
            "vision_predicted_class": vision_info.get("predicted_class", "unknown"),
            "entry_id": telemetry_data.get("Entry_id", 0),
            "raw_timestamp": str(telemetry_data.get("Timestamp", "")),
        }

        msg = create_message(
            source=self.contract.agent_name,
            destination="BROADCAST",
            message_type=MessageType.OBSERVATION,
            payload=obs_payload,
            confidence=vision_info.get("confidence", 1.0),
            priority=3,
            metadata=metadata,
        )

        # 5. Publish to MessageBus if active
        if self.is_active:
            self.publish(msg)

        logger.info(
            "PerceptionAgent published OBSERVATION %s (Entry_id=%s, Class=%s, Conf=%.2f)",
            msg.uuid[:8],
            telemetry_data.get("Entry_id"),
            vision_info.get("predicted_class"),
            vision_info.get("confidence", 0.0),
        )

        return msg

    def process(self, message: ACAMessage) -> Optional[ACAMessage]:
        """Handles incoming TASK messages to trigger a perception tick."""
        if message.message_type == MessageType.TASK:
            img_path = message.metadata.get("image_path", "")
            return self.perceive(image_path=img_path)
        return None
