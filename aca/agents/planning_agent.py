"""
ACA Planning Agent — Agronomic Decision Formulation
====================================================

Implements the Planning Agent responsible for translating disease hypotheses
and microclimate evidence into committed, executable physical intervention
decisions.

Architectural Guarantees:
    - Inherits from ``BaseAgent`` and conforms to ``AgentContract``.
    - Subscribes to ``MessageType.HYPOTHESIS`` and emits ``MessageType.DECISION``.
    - Enforces full causal traceability: links ``justification_ids`` back to
      the origin ``hypothesis_id`` and `observation_id`.
    - Local LLM Planning (``gemma4:4b-q4_K_M``) constrained to max 256 output tokens
      with embedded agronomic decision matrix fallback.
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
    DecisionPayload,
    HypothesisPayload,
    MessageType,
    create_message,
)

logger = get_logger("agents.planning")

# Agronomic Decision Matrix for Deterministic Fallback & Fast Planning
DECISION_MATRIX: Dict[str, Dict[str, Any]] = {
    "Late_blight": {
        "action_type": "EMERGENCY_FUNGAL_CONTAINMENT",
        "irrigation_action": "decrease",
        "irrigation_reason": "Throttling irrigation to eliminate canopy free moisture and halt sporangia germination.",
        "treatment": "Apply curative translaminar fungicide (e.g. cymoxanil + mancozeb).",
        "urgency": "CRITICAL",
    },
    "Early_blight": {
        "action_type": "FUNGAL_REMEDIATION",
        "irrigation_action": "decrease",
        "irrigation_reason": "Reduce leaf wetness hours while maintaining root zone hydration.",
        "treatment": "Apply protective chlorothalonil / azoxystrobin spray and prune infected lower leaves.",
        "urgency": "HIGH",
    },
    "Bacterial_spot": {
        "action_type": "BACTERIAL_CONTAINMENT",
        "irrigation_action": "stop",
        "irrigation_reason": "Halt overhead irrigation to prevent splash dissemination of bacterial exudate.",
        "treatment": "Apply copper hydroxide + mancozeb tank mix and enforce strict tool sanitation.",
        "urgency": "HIGH",
    },
    "Leaf_Mold": {
        "action_type": "GREENHOUSE_HUMIDITY_CONTROL",
        "irrigation_action": "decrease",
        "irrigation_reason": "Reduce greenhouse relative humidity below 80% to suppress Passalora fulva sporulation.",
        "treatment": "Apply bio-fungicide (Bacillus amyloliquefaciens) and maximize exhaust ventilation.",
        "urgency": "MEDIUM",
    },
    "Septoria_leaf_spot": {
        "action_type": "FOLIAR_SANITATION",
        "irrigation_action": "decrease",
        "irrigation_reason": "Minimize foliage wetness duration to prevent pycnidial release.",
        "treatment": "Prune symptomatic foliage and spray protective copper/copper-soap formulation.",
        "urgency": "MEDIUM",
    },
    "powdery_mildew": {
        "action_type": "MILDEW_MANAGEMENT",
        "irrigation_action": "maintain",
        "irrigation_reason": "Maintain normal subsurface root hydration.",
        "treatment": "Apply potassium bicarbonate or horticultural neem oil spray.",
        "urgency": "MEDIUM",
    },
    "Spider_mites Two-spotted_spider_mite": {
        "action_type": "ACARICIDE_AND_HUMIDIFICATION",
        "irrigation_action": "increase",
        "irrigation_reason": "Raise soil moisture and ambient humidity to suppress mite reproduction and plant drought stress.",
        "treatment": "Release predatory mites (Phytoseiulus persimilis) and apply insecticidal potassium soap.",
        "urgency": "HIGH",
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "action_type": "VECTOR_CONTROL_AND_ROGUING",
        "irrigation_action": "maintain",
        "irrigation_reason": "Maintain standard irrigation regime.",
        "treatment": "Deploy mass yellow sticky vector traps, apply acetamiprid for whiteflies, and rogue infected plants.",
        "urgency": "CRITICAL",
    },
    "Tomato_mosaic_virus": {
        "action_type": "VIRUS_CONTAINMENT_AND_HYGIENE",
        "irrigation_action": "maintain",
        "irrigation_reason": "Maintain balanced root hydration without plant contact.",
        "treatment": "Implement strict hygiene: 20% skim milk dip for workers, disinfect shears with trisodium phosphate.",
        "urgency": "HIGH",
    },
    "Target_Spot": {
        "action_type": "CANOPY_AIRFLOW_AND_FUNGICIDE",
        "irrigation_action": "decrease",
        "irrigation_reason": "Decrease irrigation to reduce humidity pocketing in lower canopy.",
        "treatment": "Apply broad-spectrum protective fungicide.",
        "urgency": "MEDIUM",
    },
    "healthy": {
        "action_type": "PREVENTATIVE_MAINTENANCE",
        "irrigation_action": "maintain",
        "irrigation_reason": "Current moisture levels optimal; keep existing fertigation schedule.",
        "treatment": "Routine weekly scouting and baseline crop monitoring.",
        "urgency": "LOW",
    },
}


class PlanningAgent(BaseAgent):
    """
    Planning Agent for the Agricultural Cognitive Architecture.

    Translates disease hypotheses and sensor context into physical tool
    execution plans, selecting specific actuator commands and packaging
    them into verified ``DecisionPayload`` messages.

    Args:
        message_bus: Central ACA pub/sub message broker.
        memory_gateway: Permission-gated memory access proxy.
        tool_gateway: Permission-gated tool invocation proxy.
        ollama_model: Local LLM model identifier (default: ``gemma4:4b-q4_K_M``).
        ollama_endpoint: Ollama HTTP API endpoint (default: ``http://localhost:11434``).
        timeout_seconds: Maximum allowed time for LLM generation.
    """

    def __init__(
        self,
        message_bus: MessageBus,
        memory_gateway: MemoryGateway,
        tool_gateway: ToolGateway,
        ollama_model: str = "gemma4:4b-q4_K_M",
        ollama_endpoint: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 15.0,
    ) -> None:
        self.ollama_model = ollama_model
        self.ollama_endpoint = ollama_endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._ollama_checked = False
        self._ollama_available = False

        super(PlanningAgent, self).__init__(
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
                logger.info("Ollama offline at %s; using embedded agronomic decision matrix.", self.ollama_endpoint)
        return self._ollama_available

    @property
    def contract(self) -> AgentContract:
        """Formal contract specification for the Planning Agent."""
        return AgentContract(
            agent_name="planning_agent",
            purpose="Formulate agronomic intervention decisions and tool actions based on disease hypotheses and environmental context",
            cognitive_layer=CognitiveLayer.PLANNING,
            inputs=["hypothesis_payload"],
            outputs=["decision_payload"],
            memory_permissions={
                "working": MemoryAccess.READ_WRITE,
                "semantic": MemoryAccess.READ,
            },
            tools_allowed={"irrigation_control", "treatment_alert", "agronomy_rules"},
            latency_budget_ms=15000.0,
            failure_modes={
                "ollama_unreachable": "Fallback to embedded agronomic decision matrix",
                "invalid_hypothesis": "Skip message and log contract error",
            },
            messages_published=[MessageType.DECISION],
            messages_subscribed=[MessageType.HYPOTHESIS],
            confidence_range=(0.0, 1.0),
        )

    def process(self, message: ACAMessage) -> Optional[ACAMessage]:
        """
        Process incoming HYPOTHESIS messages from the Reasoning Agent.

        Constructs planning prompt, decides on required physical and treatment
        interventions, and emits a ``DecisionPayload`` with complete trace IDs.
        """
        if message.message_type != MessageType.HYPOTHESIS:
            return None

        t0 = time.perf_counter()

        payload = message.payload
        if not isinstance(payload, HypothesisPayload):
            logger.warning("Received invalid payload type for HYPOTHESIS: %s", type(payload))
            return None

        meta = message.metadata or {}
        suspected_cause = payload.suspected_cause or meta.get("predicted_class", "healthy")
        prior_conf = payload.prior_probability
        likelihood = payload.likelihood_ratio
        target_zone = meta.get("target_zone", "tomato_greenhouse_zone_1")
        env_ctx = meta.get("environmental_context", {})
        etiology_supported = meta.get("etiology_supported", True)

        logger.info(
            "PlanningAgent formulating plan for hypothesis [%s]: Cause='%s', Prior=%.2f, Likelihood=%.2f, Supported=%s",
            payload.hypothesis_id,
            suspected_cause,
            prior_conf,
            likelihood,
            etiology_supported,
        )

        # 1. Build Concise Planning Prompt for Local Ollama LLM
        prompt = self.construct_planning_prompt(
            pathogen=suspected_cause,
            confidence=prior_conf,
            likelihood_ratio=likelihood,
            etiology_supported=etiology_supported,
            target_zone=target_zone,
            env_context=env_ctx,
        )

        # 2. Formulate Plan (Ollama LLM with Rule Matrix Fallback)
        tool_calls, action_selected, rationale = self.formulate_plan_llm(
            prompt=prompt,
            pathogen=suspected_cause,
            target_zone=target_zone,
            env_context=env_ctx,
            etiology_supported=etiology_supported,
        )

        # 3. Build DecisionPayload with Full Traceability
        decision_id = f"dec_{uuid.uuid4().hex[:12]}"
        justification_ids = [payload.hypothesis_id]
        if payload.associated_evidence_ids:
            justification_ids.extend(payload.associated_evidence_ids)

        decision_payload = DecisionPayload(
            decision_id=decision_id,
            justification_ids=justification_ids,
            action_selected=action_selected,
            skill_name="actuator_dispatch",
            parameters={
                "tool_calls": tool_calls,
                "target_zone": target_zone,
                "pathogen": suspected_cause,
                "justification_trace": justification_ids,
            },
        )

        # 4. Wrap into ACAMessage Envelope
        lat_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        out_meta = {
            "hypothesis_id": payload.hypothesis_id,
            "target_zone": target_zone,
            "pathogen": suspected_cause,
            "planning_rationale": rationale,
            "tool_calls_count": len(tool_calls),
            "model_used": self.ollama_model,
            "planning_latency_ms": lat_ms,
        }

        decision_message = create_message(
            source=self.contract.agent_name,
            destination="BROADCAST",
            message_type=MessageType.DECISION,
            payload=decision_payload,
            confidence=round(message.confidence, 4),
            priority=message.priority,
            metadata=out_meta,
        )

        logger.info(
            "PlanningAgent generated DECISION [%s] -> Action: '%s' with %d tool calls (linked to %s)",
            decision_id,
            action_selected,
            len(tool_calls),
            justification_ids,
        )
        return decision_message

    def construct_planning_prompt(
        self,
        pathogen: str,
        confidence: float,
        likelihood_ratio: float,
        etiology_supported: bool,
        target_zone: str,
        env_context: Dict[str, Any],
    ) -> str:
        """
        Construct strict, token-efficient planning prompt for gemma4:4b-q4_K_M.
        """
        temp = env_context.get("temperature_c", "N/A")
        hum = env_context.get("humidity_percent", "N/A")
        moist = env_context.get("soil_moisture_percent", "N/A")

        return (
            f"You are an agronomic decision engine for precision greenhouse control.\n"
            f"Hypothesis Data:\n"
            f"- Pathogen: {pathogen}\n"
            f"- Optical Confidence: {confidence * 100:.1f}%\n"
            f"- Microclimate Corroboration: {'CONFIRMED' if etiology_supported else 'UNCONFIRMED'}\n"
            f"- Conditions: Temp={temp}°C, Humidity={hum}%, SoilMoisture={moist}%\n"
            f"- Target Zone: {target_zone}\n\n"
            f"Available Actuator Tools:\n"
            f"1. irrigation_control (actions: decrease, increase, stop, maintain)\n"
            f"2. treatment_alert (urgency: LOW, MEDIUM, HIGH, CRITICAL)\n\n"
            f"Task: Prescribe the optimal tool actions to mitigate this risk. Be concise."
        )

    def formulate_plan_llm(
        self,
        prompt: str,
        pathogen: str,
        target_zone: str,
        env_context: Dict[str, Any],
        etiology_supported: bool,
    ) -> Tuple[List[Dict[str, Any]], str, str]:
        """
        Formulate tool calls and action strategy via Ollama with Rule Matrix fallback.

        Returns:
            Tuple of (tool_calls_list, action_selected_str, rationale_str)
        """
        matrix = DECISION_MATRIX.get(pathogen, DECISION_MATRIX["healthy"])
        action_type = matrix["action_type"]
        irr_action = matrix["irrigation_action"]
        irr_reason = matrix["irrigation_reason"]
        treatment = matrix["treatment"]
        urgency = matrix["urgency"]

        # Default structured tool calls from matrix
        default_tool_calls: List[Dict[str, Any]] = []

        if irr_action != "maintain" or pathogen == "healthy":
            default_tool_calls.append({
                "tool_name": "irrigation_control",
                "parameters": {
                    "action": irr_action,
                    "zone": target_zone,
                    "reason": irr_reason,
                },
            })

        if pathogen != "healthy":
            default_tool_calls.append({
                "tool_name": "treatment_alert",
                "parameters": {
                    "disease_name": pathogen,
                    "treatment": treatment,
                    "urgency": urgency,
                    "zone": target_zone,
                    "notes": f"Automated recommendation triggered by {pathogen} diagnosis.",
                },
            })

        # Attempt querying Ollama with strict max_tokens = 256 if daemon is online
        llm_text = ""
        if self._is_ollama_online():
            try:
                url = f"{self.ollama_endpoint}/api/generate"
                body = {
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 256,
                    },
                }
                resp = requests.post(url, json=body, timeout=self.timeout_seconds)
                if resp.status_code == 200:
                    data = resp.json()
                    llm_text = data.get("response", "").strip()
                    logger.info("PlanningAgent Ollama decision generated.")
            except Exception as exc:
                logger.warning(
                    "Ollama unavailable for planning (%s). Using embedded agronomic decision matrix.",
                    exc,
                )

        rationale = llm_text if llm_text else f"Matrix Decision for {pathogen}: {irr_reason} | {treatment}"
        return default_tool_calls, action_type, rationale
