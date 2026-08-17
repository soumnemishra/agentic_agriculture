"""
ACA Reasoning Agent (Sensor Fusion)
====================================

Subscribes to MessageType.OBSERVATION messages from the Perception Agent,
extracts vision predictions and IoT microclimate telemetry, and constructs
prompts for the local Ollama LLM (`gemma4:4b-q4_K_M`) to perform sensor fusion
and yield diagnostic hypotheses (MessageType.HYPOTHESIS).
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any, Dict, Optional
import uuid

from aca.agents.base_agent import AgentContract, BaseAgent, CognitiveLayer, MemoryAccess
from aca.logging_config import get_logger
from aca.orchestration.message_bus import MessageBus
from aca.orchestration.schemas import (
    ACAMessage,
    HypothesisPayload,
    MessageType,
    ObservationPayload,
    create_message,
)

logger = get_logger("agents.reasoning")


class ReasoningAgent(BaseAgent):
    """
    Reasoning Agent implementing multi-modal agronomic sensor fusion.

    Fuses visual disease predictions with microclimate sensor telemetry
    via a local Ollama LLM (`gemma4:4b-q4_K_M`) or fallback agronomic engine.

    Args:
        message_bus: System MessageBus.
        memory_gateway: Gated memory proxy.
        tool_gateway: Gated tool proxy.
        ollama_model_name: Ollama model identifier (default: `gemma4:4b-q4_K_M`).
        ollama_host: Ollama server endpoint URL.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        message_bus: MessageBus,
        memory_gateway: Any,
        tool_gateway: Any,
        ollama_model_name: str = "gemma4:4b-q4_K_M",
        ollama_host: str = "http://localhost:11434",
        timeout: float = 10.0,
    ) -> None:
        self.ollama_model_name = ollama_model_name
        self.ollama_host = ollama_host.rstrip("/")
        self.timeout = timeout
        super().__init__(
            message_bus=message_bus,
            memory_gateway=memory_gateway,
            tool_gateway=tool_gateway,
        )

    @property
    def contract(self) -> AgentContract:
        return AgentContract(
            agent_name="reasoning_agent",
            purpose="Performs multi-modal sensor fusion combining vision disease predictions with microclimate telemetry via local Ollama LLM.",
            cognitive_layer=CognitiveLayer.REASONING,
            inputs=["observation_payload"],
            outputs=["hypothesis_payload"],
            memory_permissions={"working": MemoryAccess.READ_WRITE, "episodic": MemoryAccess.READ},
            tools_allowed={"agronomy_rule_engine", "llm_query"},
            latency_budget_ms=1000.0,
            messages_subscribed=[MessageType.OBSERVATION],
            messages_published=[MessageType.HYPOTHESIS],
            confidence_range=(0.0, 1.0),
        )

    def _query_ollama(self, prompt: str) -> Optional[str]:
        """
        Sends prompt to the local Ollama LLM endpoint.

        Returns response text or None if server is unavailable.
        """
        url = f"{self.ollama_host}/api/generate"
        payload = {
            "model": self.ollama_model_name,
            "prompt": prompt,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 200:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    response_text = resp_data.get("response", "").strip()
                    logger.info("Ollama LLM successfully generated sensor fusion response.")
                    return response_text
        except (urllib.error.URLError, TimeoutError, Exception) as e:
            logger.warning(
                "Ollama endpoint %s unavailable (%s). Falling back to rule-based agronomic synthesis.",
                url,
                e,
            )
        return None

    def _fallback_agronomic_fusion(
        self,
        predicted_class: str,
        confidence: float,
        temp: float,
        humidity: float,
        moisture: float,
    ) -> str:
        """Fallback agronomic synthesis when local LLM server is offline."""
        is_healthy = predicted_class.lower() == "healthy"

        if is_healthy:
            if humidity > 85.0 or moisture > 70.0:
                return (
                    f"Plant appears healthy (conf={confidence*100:.1f}%), but microclimate "
                    f"shows elevated humidity ({humidity:.1f}%) and moisture ({moisture:.1f}%). "
                    "Recommend preventative ventilation to avoid fungal onset."
                )
            return (
                f"Plant health confirmed healthy with {confidence*100:.1f}% confidence. "
                f"Microclimate conditions (Temp={temp:.1f}°C, Humidity={humidity:.1f}%, Moisture={moisture:.1f}%) "
                "are within optimal agronomic parameters."
            )
        else:
            supports = humidity >= 70.0 or temp >= 22.0
            etiology_verdict = "supported" if supports else "partially aligned with"
            return (
                f"Agronomic Etiology: Visual diagnosis of '{predicted_class}' ({confidence*100:.1f}% confidence) is "
                f"{etiology_verdict} environmental telemetry (Temp={temp:.1f}°C, Humidity={humidity:.1f}%, Moisture={moisture:.1f}%). "
                "Recommended next steps: Inspect lower leaves, adjust drip irrigation, and isolate zone."
            )

    def process(self, message: ACAMessage) -> Optional[ACAMessage]:
        """
        Processes OBSERVATION messages, performs sensor fusion, and emits HYPOTHESIS.
        """
        if message.message_type != MessageType.OBSERVATION:
            return None

        payload: ObservationPayload = message.payload
        measurements = payload.measurements

        # Extract telemetry parameters
        temp = float(measurements.get("Environment Temperature", 25.0))
        humidity = float(measurements.get("Environment Humidity", 60.0))
        moisture = float(measurements.get("Soil Moisture", 50.0))

        # Extract vision diagnosis from metadata
        predicted_class = message.metadata.get("vision_predicted_class", "healthy")
        confidence = float(message.confidence)

        # Construct prompt for local Ollama LLM (`gemma4:4b-q4_K_M`)
        prompt = (
            "You are an agronomic reasoning engine. "
            f"The vision model suspects {predicted_class} with {confidence*100:.1f}% confidence. "
            f"Current conditions: Temperature={temp:.1f}C, Humidity={humidity:.1f}%, Soil Moisture={moisture:.1f}%. "
            "Do these environmental conditions support this disease etiology? Recommend next steps."
        )

        # Query LLM with fallback
        llm_response = self._query_ollama(prompt)
        if not llm_response:
            llm_response = self._fallback_agronomic_fusion(
                predicted_class=predicted_class,
                confidence=confidence,
                temp=temp,
                humidity=humidity,
                moisture=moisture,
            )

        # Compute likelihood update
        supports = humidity >= 75.0 or moisture >= 65.0
        likelihood_ratio = 1.3 if supports else 0.85

        hyp_payload = HypothesisPayload(
            hypothesis_id=f"hyp_{uuid.uuid4().hex[:8]}",
            associated_evidence_ids=[payload.observation_id],
            suspected_cause=f"{predicted_class}: {llm_response[:120]}...",
            prior_probability=confidence,
            likelihood_ratio=likelihood_ratio,
        )

        response_msg = create_message(
            source=self.contract.agent_name,
            destination="BROADCAST",
            message_type=MessageType.HYPOTHESIS,
            payload=hyp_payload,
            confidence=round(min(confidence * likelihood_ratio, 1.0), 4),
            priority=3,
            metadata={
                "prompt": prompt,
                "ollama_model_name": self.ollama_model_name,
                "llm_reasoning": llm_response,
                "predicted_class": predicted_class,
                "environment": {
                    "temperature": temp,
                    "humidity": humidity,
                    "soil_moisture": moisture,
                },
            },
        )

        logger.info(
            "ReasoningAgent published HYPOTHESIS %s for cause '%s'",
            response_msg.uuid[:8],
            predicted_class,
        )

        return response_msg
