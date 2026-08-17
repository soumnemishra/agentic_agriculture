"""
ACA Reasoning Agent — Sensor Fusion & Agronomic Cognition
=========================================================

Implements the Reasoning Agent responsible for fusing multi-modal perceptual
evidence (vision classification + IoT microclimate telemetry) using a local
Ollama LLM (``gemma4:4b-q4_K_M``) to assess disease etiology and generate
probabilistic hypotheses.

Architectural Guarantees:
    - Inherits from ``BaseAgent`` and conforms to ``AgentContract``.
    - Subscribes to ``MessageType.OBSERVATION`` and publishes ``MessageType.HYPOTHESIS``.
    - Evaluates whether environmental microclimate conditions (temperature,
      humidity, soil moisture, pH) corroborate the optical disease diagnosis.
    - Employs local Ollama LLM with robust timeout and embedded agronomic
      etiology rules as a resilient fallback.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

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
    HypothesisPayload,
    MessageType,
    ObservationPayload,
    create_message,
)

logger = get_logger("agents.reasoning")

# Agronomic Etiology Profiles for Validation & Heuristic Fusion
ETIOLOGY_PROFILES: Dict[str, Dict[str, Any]] = {
    "Late_blight": {
        "favorable_temp_range": (15.0, 24.0),
        "favorable_humidity_min": 85.0,
        "favorable_soil_moisture_min": 60.0,
        "etiology_text": "Favored by cool to moderate temperatures (15-24°C) and prolonged high relative humidity (>85%).",
        "default_recommendations": [
            "Reduce overhead canopy irrigation to lower leaf wetness.",
            "Apply preventative copper-based or targeted oomycete fungicide.",
            "Inspect neighboring tomato plots for rapid sporangia spread.",
        ],
    },
    "Early_blight": {
        "favorable_temp_range": (24.0, 32.0),
        "favorable_humidity_min": 75.0,
        "favorable_soil_moisture_min": 50.0,
        "etiology_text": "Favored by warm temperatures (24-32°C) combined with intermittent wet/dry foliage cycles.",
        "default_recommendations": [
            "Prune lower infected foliage and dispose off-site.",
            "Apply chlorothalonil or bio-fungicide formulation.",
            "Ensure adequate nitrogen-potassium balance to boost plant vigor.",
        ],
    },
    "Bacterial_spot": {
        "favorable_temp_range": (24.0, 30.0),
        "favorable_humidity_min": 80.0,
        "favorable_soil_moisture_min": 55.0,
        "etiology_text": "Favored by warm, moist conditions (24-30°C) with water splashing or high humidity.",
        "default_recommendations": [
            "Avoid handling plants when leaves are wet.",
            "Apply preventative copper bactericide combined with mancozeb.",
            "Sanitize pruning tools between rows.",
        ],
    },
    "Leaf_Mold": {
        "favorable_temp_range": (20.0, 26.0),
        "favorable_humidity_min": 85.0,
        "favorable_soil_moisture_min": 50.0,
        "etiology_text": "High greenhouse relative humidity (>85%) and moderate temperatures (20-26°C) accelerate mold proliferation.",
        "default_recommendations": [
            "Increase greenhouse air ventilation and circulation fans.",
            "Space plants to improve air penetration.",
            "Apply protective fungicide if humidity cannot be kept below 80%.",
        ],
    },
    "Septoria_leaf_spot": {
        "favorable_temp_range": (18.0, 27.0),
        "favorable_humidity_min": 80.0,
        "favorable_soil_moisture_min": 50.0,
        "etiology_text": "Favored by moderate temperatures (18-27°C) and extended wet periods.",
        "default_recommendations": [
            "Remove lower diseased leaves showing pycnidia specks.",
            "Mulch soil surface to prevent soil splash onto foliage.",
            "Apply protective fungicide spray.",
        ],
    },
    "powdery_mildew": {
        "favorable_temp_range": (18.0, 28.0),
        "favorable_humidity_min": 50.0,
        "favorable_soil_moisture_min": 30.0,
        "etiology_text": "Favored by moderate warmth (18-28°C) and dry foliage with moderate ambient humidity.",
        "default_recommendations": [
            "Apply potassium bicarbonate, sulfur, or neem oil spray.",
            "Improve sunlight exposure and plant spacing.",
        ],
    },
    "Spider_mites Two-spotted_spider_mite": {
        "favorable_temp_range": (26.0, 38.0),
        "favorable_humidity_min": 20.0,
        "favorable_soil_moisture_min": 20.0,
        "etiology_text": "Favored by hot, dry conditions (>26°C, low humidity <60%) and drought-stressed plants.",
        "default_recommendations": [
            "Introduce predatory mites (Phytoseiulus persimilis).",
            "Misting or overhead humidification to deter mite reproduction.",
            "Apply insecticidal soap or horticultural oil spray.",
        ],
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "favorable_temp_range": (25.0, 38.0),
        "favorable_humidity_min": 30.0,
        "favorable_soil_moisture_min": 30.0,
        "etiology_text": "Transmitted by whiteflies (Bemisia tabaci); favored by warm greenhouse or field climates.",
        "default_recommendations": [
            "Deploy yellow sticky traps for whitefly vector monitoring.",
            "Apply targeted insecticidal soap or biological controls for whiteflies.",
            "Rogue and remove severely stunted virus-infected plants.",
        ],
    },
    "Tomato_mosaic_virus": {
        "favorable_temp_range": (20.0, 32.0),
        "favorable_humidity_min": 40.0,
        "favorable_soil_moisture_min": 40.0,
        "etiology_text": "Mechanically transmitted tobamovirus persisting on seed coats, hands, and tools.",
        "default_recommendations": [
            "Strict sanitation: wash hands with milk/detergent solution and disinfect tools.",
            "Isolate and safely remove symptomatic plants.",
        ],
    },
    "Target_Spot": {
        "favorable_temp_range": (20.0, 30.0),
        "favorable_humidity_min": 80.0,
        "favorable_soil_moisture_min": 50.0,
        "etiology_text": "Favored by warm, humid environments and free moisture on leaves.",
        "default_recommendations": [
            "Improve canopy airflow and apply appropriate protective fungicides.",
        ],
    },
    "healthy": {
        "favorable_temp_range": (18.0, 28.0),
        "favorable_humidity_min": 60.0,
        "favorable_soil_moisture_min": 50.0,
        "etiology_text": "Optimal microclimate maintaining physiological vigor and high disease resistance.",
        "default_recommendations": [
            "Maintain current irrigation, nutrient, and ventilation schedules.",
            "Continue routine scout monitoring.",
        ],
    },
}


class ReasoningAgent(BaseAgent):
    """
    Reasoning Agent for Multi-Modal Sensor Fusion and Agronomic Etiology Analysis.

    Subscribes to ``MessageType.OBSERVATION`` from the Perception Agent,
    fuses optical vision diagnoses with IoT environmental readings, constructs
    an agronomic prompt for the local Ollama LLM (``gemma4:4b-q4_K_M``), and
    emits calibrated ``HypothesisPayload`` messages.

    Args:
        message_bus: Central ACA pub/sub message broker.
        memory_gateway: Permission-gated memory access proxy.
        tool_gateway: Permission-gated tool invocation proxy.
        ollama_model: Name of the local LLM model (default: ``gemma4:4b-q4_K_M``).
        ollama_endpoint: Ollama HTTP API endpoint (default: ``http://localhost:11434``).
        timeout_seconds: Maximum latency allowed for LLM inference.
    """

    def __init__(
        self,
        message_bus: MessageBus,
        memory_gateway: MemoryGateway,
        tool_gateway: ToolGateway,
        ollama_model: str = "gemma4:4b-q4_K_M",
        ollama_endpoint: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 30.0,
    ) -> None:
        self.ollama_model = ollama_model
        self.ollama_endpoint = ollama_endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._ollama_checked = False
        self._ollama_available = False

        super(ReasoningAgent, self).__init__(
            message_bus=message_bus,
            memory_gateway=memory_gateway,
            tool_gateway=tool_gateway,
        )

    def _is_ollama_online(self) -> bool:
        if not self._ollama_checked:
            self._ollama_checked = True
            import socket
            try:
                # Direct socket probe to avoid Windows localhost DNS resolution delay
                with socket.create_connection(("127.0.0.1", 11434), timeout=0.05):
                    self._ollama_available = True
            except Exception:
                self._ollama_available = False
            if not self._ollama_available:
                logger.info("Ollama offline at %s; using embedded agronomic rules engine.", self.ollama_endpoint)
        return self._ollama_available

    @property
    def contract(self) -> AgentContract:
        """Formal contract specification for the Reasoning Agent."""
        return AgentContract(
            agent_name="reasoning_agent",
            purpose="Perform multi-modal agronomic sensor fusion using local Ollama LLM to assess disease etiology against microclimate conditions",
            cognitive_layer=CognitiveLayer.REASONING,
            inputs=["observation_payload"],
            outputs=["hypothesis_payload"],
            memory_permissions={
                "working": MemoryAccess.READ_WRITE,
                "semantic": MemoryAccess.READ,
            },
            tools_allowed={"llm_infer", "agronomy_rules"},
            latency_budget_ms=30000.0,
            failure_modes={
                "ollama_unreachable": "Fallback to embedded agronomic etiology rule engine",
                "malformed_observation": "Skip message and log contract error",
            },
            messages_published=[MessageType.HYPOTHESIS],
            messages_subscribed=[MessageType.OBSERVATION],
            confidence_range=(0.0, 1.0),
        )

    def process(self, message: ACAMessage) -> Optional[ACAMessage]:
        """
        Process incoming OBSERVATION messages from the Perception Agent.

        Extracts vision class, confidence, and environmental sensors, performs
        sensor fusion reasoning via Ollama LLM, and publishes a HYPOTHESIS.
        """
        if message.message_type != MessageType.OBSERVATION:
            return None

        t0 = time.perf_counter()

        payload = message.payload
        if not isinstance(payload, ObservationPayload):
            logger.warning("Received invalid payload type for OBSERVATION: %s", type(payload))
            return None

        # 1. Extract Perceptual and Environmental Data
        meta = message.metadata or {}
        predicted_class = meta.get("predicted_class", "unknown")
        vision_conf = float(meta.get("confidence", payload.measurements.get("vision_confidence", 0.0)))

        temp_c = payload.measurements.get("environment_temperature_c", 25.0)
        humidity = payload.measurements.get("environment_humidity", 70.0)
        soil_moisture = payload.measurements.get("soil_moisture", 50.0)
        soil_ph = payload.measurements.get("soil_ph", 6.5)
        soil_temp = payload.measurements.get("soil_temperature_c", 22.0)
        light_lux = payload.measurements.get("environment_light_lux", 50.0)

        logger.info(
            "ReasoningAgent processing observation [%s]: Vision=%s (%.1f%%), Temp=%.1f°C, Hum=%.1f%%, Moisture=%.1f%%",
            payload.observation_id,
            predicted_class,
            vision_conf * 100.0,
            temp_c,
            humidity,
            soil_moisture,
        )

        # 2. Build Agronomic Sensor Fusion Prompt
        prompt = self.construct_fusion_prompt(
            predicted_class=predicted_class,
            confidence=vision_conf,
            temperature_c=temp_c,
            humidity=humidity,
            soil_moisture=soil_moisture,
            soil_ph=soil_ph,
            soil_temp_c=soil_temp,
            light_lux=light_lux,
        )

        # 3. Query Local Ollama LLM (with Rule Engine Fallback)
        llm_response, etiology_supported, likelihood_ratio, recommendations = self.query_sensor_fusion_llm(
            prompt=prompt,
            predicted_class=predicted_class,
            temp_c=temp_c,
            humidity=humidity,
            soil_moisture=soil_moisture,
            soil_ph=soil_ph,
        )

        # 4. Construct HypothesisPayload
        hyp_id = f"hyp_{uuid.uuid4().hex[:12]}"
        prior_prob = vision_conf if vision_conf > 0.0 else 0.5

        # Calculate posterior confidence update using likelihood ratio
        updated_prob = min(0.99, max(0.05, (prior_prob * likelihood_ratio) / ((prior_prob * likelihood_ratio) + (1.0 - prior_prob))))

        hypothesis_payload = HypothesisPayload(
            hypothesis_id=hyp_id,
            associated_evidence_ids=[payload.observation_id],
            suspected_cause=predicted_class,
            prior_probability=round(prior_prob, 4),
            likelihood_ratio=round(likelihood_ratio, 3),
        )

        # 5. Build ACAMessage Envelope
        lat_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        out_meta = {
            "observation_id": payload.observation_id,
            "target_zone": payload.target_zone,
            "predicted_class": predicted_class,
            "etiology_supported": etiology_supported,
            "llm_reasoning": llm_response,
            "recommendations": recommendations,
            "environmental_context": {
                "temperature_c": temp_c,
                "humidity_percent": humidity,
                "soil_moisture_percent": soil_moisture,
                "soil_ph": soil_ph,
            },
            "model_engine": self.ollama_model,
            "reasoning_latency_ms": lat_ms,
        }

        priority = 4 if predicted_class not in ("healthy", "unknown") and updated_prob > 0.70 else 3

        hypothesis_message = create_message(
            source=self.contract.agent_name,
            destination="BROADCAST",
            message_type=MessageType.HYPOTHESIS,
            payload=hypothesis_payload,
            confidence=round(updated_prob, 4),
            priority=priority,
            metadata=out_meta,
        )

        # 6. Log and Return (BaseAgent will automatically publish non-None return values)
        logger.info(
            "ReasoningAgent generated HYPOTHESIS [%s] -> %s (prior=%.2f, likelihood=%.2f, posterior=%.2f, supported=%s)",
            hyp_id,
            predicted_class,
            prior_prob,
            likelihood_ratio,
            updated_prob,
            etiology_supported,
        )
        return hypothesis_message

    def construct_fusion_prompt(
        self,
        predicted_class: str,
        confidence: float,
        temperature_c: float,
        humidity: float,
        soil_moisture: float,
        soil_ph: float,
        soil_temp_c: float,
        light_lux: float,
    ) -> str:
        """
        Construct strict sensor fusion prompt for the local Ollama LLM.
        """
        return (
            f"You are an expert agronomic reasoning engine. "
            f"The vision model suspects {predicted_class} with {confidence * 100:.1f}% confidence.\n"
            f"Current microclimate conditions:\n"
            f"- Ambient Temperature: {temperature_c:.1f}°C\n"
            f"- Relative Humidity: {humidity:.1f}%\n"
            f"- Soil Moisture: {soil_moisture:.1f}%\n"
            f"- Soil pH: {soil_ph:.2f}\n"
            f"- Soil Temperature: {soil_temp_c:.1f}°C\n"
            f"- Light Intensity: {light_lux:.1f} Lux\n\n"
            f"Task:\n"
            f"1. Do these environmental conditions support this disease etiology? Answer YES or NO and explain why.\n"
            f"2. Assess the physiological risk to the tomato crop.\n"
            f"3. Recommend 2 to 3 actionable next steps for greenhouse management.\n"
        )

    def query_sensor_fusion_llm(
        self,
        prompt: str,
        predicted_class: str,
        temp_c: float,
        humidity: float,
        soil_moisture: float,
        soil_ph: float,
    ) -> Tuple[str, bool, float, List[str]]:
        """
        Execute Ollama query with robust error handling and rule-based fallback.

        Returns:
            Tuple of (llm_text, etiology_supported_bool, likelihood_ratio, recommendations_list)
        """
        # 1. First assess baseline heuristic profile
        profile = ETIOLOGY_PROFILES.get(predicted_class, {})
        temp_min, temp_max = profile.get("favorable_temp_range", (15.0, 35.0))
        hum_min = profile.get("favorable_humidity_min", 60.0)
        moist_min = profile.get("favorable_soil_moisture_min", 40.0)
        default_recs = profile.get("default_recommendations", [
            "Monitor crop canopy for symptom progression.",
            "Verify irrigation line pressure and soil drainage.",
        ])

        temp_match = temp_min <= temp_c <= temp_max
        hum_match = humidity >= hum_min if "Spider" not in predicted_class else humidity <= 60.0
        moist_match = soil_moisture >= moist_min

        # Compute heuristic support
        heuristic_supported = (temp_match and hum_match) or (hum_match and moist_match)
        if predicted_class == "healthy":
            heuristic_supported = (18.0 <= temp_c <= 28.0) and (50.0 <= humidity <= 75.0)

        heuristic_likelihood = 1.85 if heuristic_supported else 0.55

        # 2. Attempt Ollama API query if daemon is online
        if self._is_ollama_online():
            try:
                url = f"{self.ollama_endpoint}/api/generate"
                body = {
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "top_p": 0.9,
                        "num_predict": 256,
                    },
                }
                resp = requests.post(url, json=body, timeout=self.timeout_seconds)
                if resp.status_code == 200:
                    data = resp.json()
                    text = data.get("response", "").strip()
                    if text:
                        # Parse affirmation
                        lower_t = text.lower()
                        supported = "yes" in lower_t[:50] or "support" in lower_t[:100]
                        likelihood = 2.0 if supported else 0.5
                        logger.info("Ollama LLM successfully generated agronomic reasoning.")
                        return text, supported, likelihood, default_recs
            except Exception as exc:
                logger.warning(
                    "Ollama endpoint (%s) request failed (%s). Using embedded agronomic etiology rules.",
                    self.ollama_endpoint,
                    exc,
                )

        # 3. Fallback Agronomic Rule Engine
        reasoning_text = (
            f"[Agronomic Rule Engine Analysis for {predicted_class}]\n"
            f"Etiology Context: {profile.get('etiology_text', 'Microclimate monitoring active.')}\n"
            f"Observation vs Profile:\n"
            f"- Temperature ({temp_c:.1f}°C): {'Optimal/Favorable' if temp_match else 'Sub-optimal'} for pathogen.\n"
            f"- Relative Humidity ({humidity:.1f}%): {'High risk / favorable' if hum_match else 'Low risk'} threshold.\n"
            f"- Soil Moisture ({soil_moisture:.1f}%): {'Sufficient moisture' if moist_match else 'Dry profile'}.\n"
            f"Conclusion: Microclimate {'STRONGLY SUPPORTS' if heuristic_supported else 'DOES NOT FAVOR'} "
            f"active {predicted_class} pathogen spread."
        )

        return reasoning_text, heuristic_supported, heuristic_likelihood, default_recs
