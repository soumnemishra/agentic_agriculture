"""
ACA Evaluation — Cognitive Metrics & Causal Chain Logger
=========================================================

Captures empirical performance data, latency profiles, and end-to-end causal
traceability for the Agricultural Cognitive Architecture (ACA) v1.0.

Outputs:
    - Structured CSV dataset: ``datasets/experiment_results.csv``
    - Full Causal JSONL log:  ``datasets/experiment_traces.jsonl``
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from aca.logging_config import get_logger
from aca.orchestration.message_bus import MessageBus
from aca.orchestration.schemas import (
    ACAMessage,
    DecisionPayload,
    FeedbackPayload,
    HypothesisPayload,
    MessageType,
    ObservationPayload,
)

logger = get_logger("evaluation.metrics_logger")


def _extract_numeric_metric(payload: Any, *keys: str, default: float = 0.0) -> float:
    """Helper to extract a numeric metric from payload direct attributes or measurements dictionary."""
    if payload is None:
        return default

    # 1. Direct attribute access on payload (e.g. payload.environment_temperature_c)
    for k in keys:
        if hasattr(payload, k):
            val = getattr(payload, k)
            if val is not None and not isinstance(val, (dict, list, tuple)):
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass

    # 2. Key lookup in measurements dictionary if present
    measurements = getattr(payload, "measurements", None)
    if isinstance(measurements, dict):
        for k in keys:
            if k in measurements and measurements[k] is not None:
                try:
                    return float(measurements[k])
                except (ValueError, TypeError):
                    pass

    return default


@dataclass
class CognitiveCycleRecord:
    """
    Complete empirical trace record for a single perception-to-execution cycle.
    """
    step_index: int = 0
    entry_id: int = 0
    timestamp: str = ""

    # Microclimate IoT Telemetry
    env_temp_c: float = 0.0
    env_humidity_pct: float = 0.0
    env_light_lux: float = 0.0
    soil_moisture_pct: float = 0.0
    soil_ph: float = 0.0
    soil_temp_c: float = 0.0
    solar_battery_v: float = 0.0
    water_tds_mg_l: float = 0.0

    # Perception & Vision Layer
    vision_predicted_class: str = ""
    vision_confidence: float = 0.0
    vision_latency_ms: float = 0.0

    # Reasoning Layer (Sensor Fusion & LLM)
    reasoning_cause: str = ""
    reasoning_prior: float = 0.0
    reasoning_likelihood: float = 0.0
    reasoning_posterior: float = 0.0
    reasoning_etiology_supported: bool = False
    reasoning_llm_used: str = ""
    reasoning_latency_ms: float = 0.0

    # Planning Layer (Action Formulation)
    planning_action: str = ""
    planning_tool_count: int = 0
    planning_tools_planned: str = ""
    planning_latency_ms: float = 0.0

    # Execution Layer (Actuators & Feedback)
    execution_assessment: str = ""
    execution_deviation: float = 0.0
    execution_success_count: int = 0
    execution_error_count: int = 0
    execution_latency_ms: float = 0.0

    # Total End-to-End Cycle Performance
    total_cycle_latency_ms: float = 0.0

    # Causal Traceability Chain
    observation_id: str = ""
    observation_msg_uuid: str = ""
    hypothesis_id: str = ""
    hypothesis_msg_uuid: str = ""
    decision_id: str = ""
    decision_msg_uuid: str = ""
    feedback_id: str = ""
    feedback_msg_uuid: str = ""
    causal_chain_valid: bool = False


class CognitiveMetricsLogger:
    """
    Subscribes to MessageBus events, reconstructs the causal chain,
    and logs empirical experimental records into CSV and JSONL datasets.
    """

    CSV_HEADERS = [
        "step_index",
        "entry_id",
        "timestamp",
        "env_temp_c",
        "env_humidity_pct",
        "env_light_lux",
        "soil_moisture_pct",
        "soil_ph",
        "soil_temp_c",
        "solar_battery_v",
        "water_tds_mg_l",
        "vision_predicted_class",
        "vision_confidence",
        "vision_latency_ms",
        "reasoning_cause",
        "reasoning_prior",
        "reasoning_likelihood",
        "reasoning_posterior",
        "reasoning_etiology_supported",
        "reasoning_llm_used",
        "reasoning_latency_ms",
        "planning_action",
        "planning_tool_count",
        "planning_tools_planned",
        "planning_latency_ms",
        "execution_assessment",
        "execution_deviation",
        "execution_success_count",
        "execution_error_count",
        "execution_latency_ms",
        "total_cycle_latency_ms",
        "observation_id",
        "observation_msg_uuid",
        "hypothesis_id",
        "hypothesis_msg_uuid",
        "decision_id",
        "decision_msg_uuid",
        "feedback_id",
        "feedback_msg_uuid",
        "causal_chain_valid",
    ]

    def __init__(
        self,
        message_bus: MessageBus,
        csv_output_path: str = "datasets/experiment_results.csv",
        jsonl_output_path: str = "datasets/experiment_traces.jsonl",
    ) -> None:
        self.message_bus = message_bus
        self.csv_output_path = os.path.abspath(csv_output_path)
        self.jsonl_output_path = os.path.abspath(jsonl_output_path)

        self._current_step = 0
        self._active_records: Dict[int, CognitiveCycleRecord] = {}
        self._records: List[CognitiveCycleRecord] = []
        self._phase_start_times: Dict[str, float] = {}

        # Ensure output directories exist
        os.makedirs(os.path.dirname(self.csv_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.jsonl_output_path), exist_ok=True)

        self._subscribe_to_bus()

    def _subscribe_to_bus(self) -> None:
        """Register listeners on all key message types."""
        self.message_bus.subscribe(MessageType.OBSERVATION, self._on_observation)
        self.message_bus.subscribe(MessageType.HYPOTHESIS, self._on_hypothesis)
        self.message_bus.subscribe(MessageType.DECISION, self._on_decision)
        self.message_bus.subscribe(MessageType.FEEDBACK, self._on_feedback)

    def start_step(self, step_index: int) -> None:
        """Signal start of a new experimental simulation step."""
        self._current_step = step_index
        rec = CognitiveCycleRecord(
            step_index=step_index,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._active_records[step_index] = rec
        self._phase_start_times["cycle_start"] = time.perf_counter()

    def _on_observation(self, msg: ACAMessage) -> None:
        """Capture ObservationPayload data with multi-path extraction."""
        rec = self._active_records.get(self._current_step)
        if not rec:
            return

        payload = msg.payload
        meta = msg.metadata or {}

        # Capture observation identifier and message UUID
        rec.observation_id = getattr(payload, "observation_id", "") or msg.uuid
        rec.observation_msg_uuid = msg.uuid
        rec.entry_id = int(meta.get("entry_id") or getattr(payload, "entry_id", 0) or 0)

        # Extract numeric IoT telemetry metrics directly from payload attributes or measurements dict
        rec.env_temp_c = _extract_numeric_metric(payload, "environment_temperature_c", "env_temp_c", "temperature_c")
        rec.env_humidity_pct = _extract_numeric_metric(payload, "environment_humidity", "env_humidity_pct", "humidity")
        rec.env_light_lux = _extract_numeric_metric(payload, "environment_light_lux", "env_light_lux", "light_lux")
        rec.soil_moisture_pct = _extract_numeric_metric(payload, "soil_moisture", "soil_moisture_pct")
        rec.soil_ph = _extract_numeric_metric(payload, "soil_ph", default=7.0)
        rec.soil_temp_c = _extract_numeric_metric(payload, "soil_temperature_c", "soil_temp_c")
        rec.solar_battery_v = _extract_numeric_metric(payload, "solar_battery_voltage", "solar_battery_v")
        rec.water_tds_mg_l = _extract_numeric_metric(payload, "water_tds", "water_tds_mg_l")

        # Vision prediction and confidence extraction
        rec.vision_predicted_class = str(
            meta.get("predicted_class")
            or getattr(payload, "predicted_class", "")
            or "healthy"
        )
        rec.vision_confidence = float(
            meta.get("confidence")
            or _extract_numeric_metric(payload, "vision_confidence")
            or msg.confidence
            or 0.0
        )
        vision_diag = meta.get("vision_diagnosis", {})
        rec.vision_latency_ms = float(vision_diag.get("inference_time_ms", 0.0))

    def _on_hypothesis(self, msg: ACAMessage) -> None:
        """Capture HypothesisPayload data."""
        rec = self._active_records.get(self._current_step)
        if not rec:
            return

        payload = msg.payload
        meta = msg.metadata or {}

        rec.hypothesis_id = getattr(payload, "hypothesis_id", "") or msg.uuid
        rec.hypothesis_msg_uuid = msg.uuid
        rec.reasoning_cause = str(getattr(payload, "suspected_cause", "") or meta.get("predicted_class", ""))
        rec.reasoning_prior = float(getattr(payload, "prior_probability", 0.0))
        rec.reasoning_likelihood = float(getattr(payload, "likelihood_ratio", 1.0))
        rec.reasoning_posterior = float(msg.confidence)
        rec.reasoning_etiology_supported = bool(meta.get("etiology_supported", False))
        rec.reasoning_llm_used = str(meta.get("model_engine", "gemma4:4b-q4_K_M"))
        rec.reasoning_latency_ms = float(meta.get("reasoning_latency_ms", 0.0))

    def _on_decision(self, msg: ACAMessage) -> None:
        """Capture DecisionPayload data."""
        rec = self._active_records.get(self._current_step)
        if not rec:
            return

        payload = msg.payload
        meta = msg.metadata or {}
        params = getattr(payload, "parameters", {}) or {}
        tool_calls = params.get("tool_calls", [])

        rec.decision_id = getattr(payload, "decision_id", "") or msg.uuid
        rec.decision_msg_uuid = msg.uuid
        rec.planning_action = str(getattr(payload, "action_selected", "") or meta.get("action_selected", ""))
        rec.planning_tool_count = len(tool_calls)
        rec.planning_tools_planned = ";".join([t.get("tool_name", "") for t in tool_calls])
        rec.planning_latency_ms = float(meta.get("planning_latency_ms", 0.0))

    def _on_feedback(self, msg: ACAMessage) -> None:
        """Capture FeedbackPayload data."""
        rec = self._active_records.get(self._current_step)
        if not rec:
            return

        payload = msg.payload
        actual = getattr(payload, "actual_outcome", {}) or {}
        meta = msg.metadata or {}

        rec.feedback_id = str(getattr(payload, "action_id", "") or msg.uuid)
        rec.feedback_msg_uuid = msg.uuid
        rec.execution_assessment = str(getattr(payload, "assessment", "SUCCESS"))
        rec.execution_deviation = float(getattr(payload, "deviation", 0.0))
        rec.execution_success_count = int(actual.get("success_count", 0))
        rec.execution_error_count = int(actual.get("error_count", 0))
        rec.execution_latency_ms = float(meta.get("execution_latency_ms", 0.0))

    def end_step(self, step_index: int) -> Optional[CognitiveCycleRecord]:
        """Finalize and seal the step record."""
        rec = self._active_records.pop(step_index, None)
        if not rec:
            return None

        start_c = self._phase_start_times.get("cycle_start", time.perf_counter())
        rec.total_cycle_latency_ms = round((time.perf_counter() - start_c) * 1000.0, 2)

        # Causal Chain Integrity Validation:
        # All 4 cognitive layer IDs must exist, and Feedback action_id must match Decision ID
        rec.causal_chain_valid = bool(
            rec.observation_id
            and rec.hypothesis_id
            and rec.decision_id
            and rec.feedback_id
            and (rec.feedback_id == rec.decision_id)
        )

        self._records.append(rec)
        return rec

    def flush_to_disk(self) -> None:
        """Write all logged records to CSV and JSONL."""
        if not self._records:
            logger.warning("No records to write.")
            return

        # 1. Write CSV
        with open(self.csv_output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
            writer.writeheader()
            for r in self._records:
                writer.writerow(asdict(r))

        # 2. Write JSONL
        with open(self.jsonl_output_path, "w", encoding="utf-8") as f:
            for r in self._records:
                f.write(json.dumps(asdict(r)) + "\n")

        logger.info(
            "Successfully saved %d experimental records to %s and %s",
            len(self._records),
            self.csv_output_path,
            self.jsonl_output_path,
        )

    def generate_summary(self) -> Dict[str, Any]:
        """Compute aggregate summary metrics for the experimental evaluation."""
        if not self._records:
            return {}

        total = len(self._records)
        valid_chains = sum(1 for r in self._records if r.causal_chain_valid)
        etiology_agreed = sum(1 for r in self._records if r.reasoning_etiology_supported)
        successes = sum(1 for r in self._records if r.execution_assessment == "SUCCESS")

        avg_vision_lat = sum(r.vision_latency_ms for r in self._records) / total
        avg_reason_lat = sum(r.reasoning_latency_ms for r in self._records) / total
        avg_plan_lat = sum(r.planning_latency_ms for r in self._records) / total
        avg_exec_lat = sum(r.execution_latency_ms for r in self._records) / total
        avg_total_lat = sum(r.total_cycle_latency_ms for r in self._records) / total

        # Distribution of suspected diseases
        disease_counts: Dict[str, int] = {}
        action_counts: Dict[str, int] = {}
        for r in self._records:
            disease_counts[r.reasoning_cause] = disease_counts.get(r.reasoning_cause, 0) + 1
            action_counts[r.planning_action] = action_counts.get(r.planning_action, 0) + 1

        return {
            "total_cycles": total,
            "causal_chain_integrity_pct": round((valid_chains / total) * 100.0, 2),
            "etiology_agreement_rate_pct": round((etiology_agreed / total) * 100.0, 2),
            "actuator_execution_success_pct": round((successes / total) * 100.0, 2),
            "latencies_ms": {
                "vision_inference_mean": round(avg_vision_lat, 2),
                "reasoning_layer_mean": round(avg_reason_lat, 2),
                "planning_layer_mean": round(avg_plan_lat, 2),
                "execution_layer_mean": round(avg_exec_lat, 2),
                "end_to_end_cycle_mean": round(avg_total_lat, 2),
            },
            "disease_distribution": disease_counts,
            "action_distribution": action_counts,
        }
