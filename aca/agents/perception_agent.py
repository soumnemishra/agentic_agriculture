"""
ACA Perception Agent — Multi-Modal Perception & Ingestion
==========================================================

Implements the Perception Agent responsible for ingesting synchronized
IoT environmental telemetry and executing deep-learning vision inference
for tomato crop health monitoring.

Architectural Guarantees:
    - Inherits from ``BaseAgent`` and conforms to ``AgentContract``.
    - Publishes typed ``ObservationPayload`` wrapped in ``ACAMessage`` to the ``MessageBus``.
    - Fuses real-time microclimate sensors (temperature, humidity, moisture, pH, etc.)
      with crop disease vision diagnoses into unified observation envelopes.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import torch

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
    MessageType,
    ObservationPayload,
    create_message,
)
from aca.skills.tomato_diagnosis_skill import TomatoDiagnosisSkill
from simulation.telemetry_streamer import IoTStreamer

logger = get_logger("agents.perception")


class PerceptionAgent(BaseAgent):
    """
    Perception Agent for the Agricultural Cognitive Architecture.

    Coordinates IoT telemetry streaming and vision-based disease classification,
    packaging multi-modal perceptual streams into validated ``ObservationPayload``
    messages dispatched over the ``MessageBus``.

    Args:
        message_bus: Central ACA pub/sub message broker.
        memory_gateway: Permission-gated memory access proxy.
        tool_gateway: Permission-gated tool invocation proxy.
        iot_streamer: Synchronized IoT dataset streamer instance.
        diagnosis_skill: Tomato diagnosis PyTorch vision skill.
        target_zone: Default farm/greenhouse zone name.
    """

    def __init__(
        self,
        message_bus: MessageBus,
        memory_gateway: MemoryGateway,
        tool_gateway: ToolGateway,
        iot_streamer: Optional[IoTStreamer] = None,
        diagnosis_skill: Optional[TomatoDiagnosisSkill] = None,
        target_zone: str = "tomato_greenhouse_zone_1",
    ) -> None:
        self._iot_streamer = iot_streamer or IoTStreamer()
        self._diagnosis_skill = diagnosis_skill or TomatoDiagnosisSkill()
        self._target_zone = target_zone

        super(PerceptionAgent, self).__init__(
            message_bus=message_bus,
            memory_gateway=memory_gateway,
            tool_gateway=tool_gateway,
        )

    @property
    def contract(self) -> AgentContract:
        """Formal contract specification for the Perception Agent."""
        return AgentContract(
            agent_name="perception_agent",
            purpose="Ingest multi-modal IoT telemetry and execute vision-based tomato disease diagnosis",
            cognitive_layer=CognitiveLayer.PERCEPTION,
            inputs=["iot_telemetry", "crop_image"],
            outputs=["fused_observation"],
            memory_permissions={"working": MemoryAccess.WRITE},
            tools_allowed={"sensor_read", "camera_capture"},
            latency_budget_ms=150.0,
            failure_modes={
                "streamer_exhausted": "Reset streamer to beginning and log warning",
                "vision_inference_failure": "Produce observation with sensor telemetry only and zero vision confidence",
            },
            messages_published=[MessageType.OBSERVATION],
            messages_subscribed=[MessageType.TASK],
            confidence_range=(0.0, 1.0),
        )

    def perceive(
        self,
        image_path: Optional[str] = None,
        zone: Optional[str] = None,
        publish_message: bool = True,
    ) -> ACAMessage:
        """
        Execute one perception cycle: fetch IoT telemetry, run vision diagnosis,
        package into ``ObservationPayload``, and optionally publish to the ``MessageBus``.

        Args:
            image_path: Optional path to tomato leaf image frame (or synthetic tensor).
            zone: Optional target zone override.
            publish_message: Whether to immediately publish to the MessageBus.

        Returns:
            The constructed ``ACAMessage`` envelope containing the ``ObservationPayload``.
        """
        target_zone = zone or self._target_zone

        # 1. Fetch synchronized IoT telemetry
        telemetry = self._iot_streamer.step()
        if telemetry is None:
            logger.warning("IoT streamer returned None; resetting streamer to loop")
            self._iot_streamer.reset()
            telemetry = self._iot_streamer.step() or {}

        # 2. Execute Vision Diagnosis Skill
        vision_result: Dict[str, Any] = {}
        vision_conf = 0.0
        pred_class = "unknown"

        if image_path is not None:
            skill_res = self._diagnosis_skill.execute(image_path=image_path)
            if skill_res.success and skill_res.data:
                vision_result = skill_res.data
                pred_class = vision_result.get("predicted_class", "unknown")
                vision_conf = float(vision_result.get("confidence", 0.0))
            else:
                logger.error("Vision diagnosis skill failed: %s", skill_res.error)

        # 3. Build Sensor Source List
        source_sensors = [
            "environment_humidity_sensor",
            "environment_temperature_sensor",
            "soil_moisture_sensor",
            "soil_ph_sensor",
            "soil_temperature_sensor",
            "solar_battery_sensor",
            "water_tds_sensor",
        ]
        if image_path is not None:
            source_sensors.append("rgb_crop_camera")

        # 4. Construct Numeric Measurements Dictionary
        measurements: Dict[str, float] = {
            "environment_humidity": float(telemetry.get("environment_humidity", 0.0)),
            "environment_light_lux": float(telemetry.get("environment_light_lux", 0.0)),
            "environment_temperature_c": float(telemetry.get("environment_temperature_c", 0.0)),
            "soil_moisture": float(telemetry.get("soil_moisture", 0.0)),
            "soil_ph": float(telemetry.get("soil_ph", 7.0)),
            "soil_temperature_c": float(telemetry.get("soil_temperature_c", 0.0)),
            "solar_battery_voltage": float(telemetry.get("solar_battery_voltage", 0.0)),
            "water_tds": float(telemetry.get("water_tds", 0.0)),
        }
        if image_path is not None:
            measurements["vision_confidence"] = vision_conf

        # 5. Build ObservationPayload
        obs_id = f"obs_{uuid.uuid4().hex[:12]}"
        obs_time = telemetry.get("timestamp") or datetime.now(timezone.utc).isoformat()

        payload = ObservationPayload(
            observation_id=obs_id,
            source_sensors=source_sensors,
            target_zone=target_zone,
            observation_time=obs_time,
            measurements=measurements,
        )

        # 6. Envelope Metadata
        metadata = {
            "entry_id": telemetry.get("entry_id"),
            "predicted_class": pred_class,
            "vision_diagnosis": vision_result,
            "image_path": "tensor_frame" if isinstance(image_path, torch.Tensor) else (str(image_path) if image_path is not None else None),
            "units": telemetry.get("units", {}),
        }

        # Determine urgency priority: 4 if suspected severe pathogen, else 3
        priority = 4 if pred_class not in ("healthy", "unknown") and vision_conf > 0.70 else 3
        overall_conf = max(0.5, vision_conf) if image_path is not None else 0.95

        message = create_message(
            source=self.contract.agent_name,
            destination="BROADCAST",
            message_type=MessageType.OBSERVATION,
            payload=payload,
            confidence=round(overall_conf, 4),
            priority=priority,
            metadata=metadata,
        )

        # 7. Optionally Publish to MessageBus
        if publish_message:
            self.publish(message)
            logger.info(
                "PerceptionAgent published OBSERVATION [%s] (entry=%s, class=%s, conf=%.2f%%)",
                obs_id,
                telemetry.get("entry_id"),
                pred_class,
                vision_conf * 100.0,
            )
        return message

    def process(self, message: ACAMessage) -> Optional[ACAMessage]:
        """
        Process incoming trigger tasks or observation requests.

        Args:
            message: Incoming ``ACAMessage`` of type ``MessageType.TASK``.

        Returns:
            Constructed ``ACAMessage`` response (published by BaseAgent dispatch).
        """
        if message.message_type != MessageType.TASK:
            return None

        # Extract task parameters if provided
        params = getattr(message.payload, "parameters", {}) if hasattr(message.payload, "parameters") else {}
        image_path = params.get("image_path")
        zone = params.get("target_zone") or getattr(message.payload, "target_zone", self._target_zone)

        return self.perceive(image_path=image_path, zone=zone, publish_message=False)
