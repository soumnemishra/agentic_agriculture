"""
Agricultural Cognitive Architecture (ACA) v1.0 — Milestone 5 Live Simulation
=============================================================================

Live demonstration script for the ACA multi-modal sensor fusion pipeline.

Pipeline Flow:
    1. IoTStreamer (CPU/Pandas) fetches 8-channel synchronized telemetry.
    2. PerceptionAgent executes TomatoDiagnosisSkill (CondConViT_V2) on GPU/CPU.
    3. PerceptionAgent publishes ACAMessage(MessageType.OBSERVATION) to MessageBus.
    4. MessageBus routes OBSERVATION to ReasoningAgent.
    5. ReasoningAgent performs sensor fusion via local Ollama LLM (gemma4:4b-q4_K_M)
       or fallback agronomic engine, publishing ACAMessage(MessageType.HYPOTHESIS).

Hardware & VRAM Safety:
    - 16GB System RAM / 4GB GTX 1650 VRAM target.
    - Sequential processing with torch.no_grad() and CUDA cache clearing.
    - 2-second sleep delay between edge ticks to prevent thermal/VRAM throttling.
"""

from __future__ import annotations

import glob
import os
import sys
import time
from typing import List, Optional

# Force UTF-8 stdout encoding for Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from aca.agents.base_agent import MemoryGateway, ToolGateway
from aca.agents.perception_agent import PerceptionAgent
from aca.agents.reasoning_agent import ReasoningAgent
from aca.config import MessageBusConfig
from aca.logging_config import get_logger
from aca.orchestration.message_bus import MessageBus
from aca.orchestration.schemas import ACAMessage, MessageType
from aca.skills.tomato_diagnosis_skill import TomatoDiagnosisSkill
from aca.tools.registry import ToolRegistry
from simulation.telemetry_streamer import IoTStreamer

logger = get_logger("simulation.runner")


def find_sample_image(dataset_base: str) -> str:
    """Discovers a sample tomato leaf image from dataset, or creates a dummy fallback."""
    pattern_jpg = os.path.join(dataset_base, "train", "*", "*.jpg")
    pattern_JPG = os.path.join(dataset_base, "train", "*", "*.JPG")
    pattern_png = os.path.join(dataset_base, "train", "*", "*.png")

    found_files = glob.glob(pattern_jpg) + glob.glob(pattern_JPG) + glob.glob(pattern_png)
    if found_files:
        sample_path = found_files[0]
        logger.info("Found sample leaf image: %s", sample_path)
        return sample_path

    # Fallback to dummy placeholder image
    dummy_dir = os.path.join(os.path.expanduser("~"), ".gemini", "antigravity", "brain", "scratch")
    os.makedirs(dummy_dir, exist_ok=True)
    dummy_path = os.path.join(dummy_dir, "sample_leaf_placeholder.jpg")
    if not os.path.exists(dummy_path):
        from PIL import Image
        img = Image.new("RGB", (224, 224), color=(34, 139, 34))
        img.save(dummy_path)
    logger.warning("No dataset images found; using dummy placeholder image: %s", dummy_path)
    return dummy_path


def main() -> None:
    print("=" * 80, flush=True)
    print("       AGRICULTURAL COGNITIVE ARCHITECTURE (ACA v1.0)", flush=True)
    print("    Milestone 5: Simulation & Multi-Modal Perception Execution", flush=True)
    print("=" * 80, flush=True)

    # 1. Initialize Configuration & Communication Infrastructure
    config = MessageBusConfig(max_queue_size=100, enable_tracing=True)
    bus = MessageBus(config)

    # 2. Gateways & Tool Registry
    memories = {"working": {}, "episodic": {}}
    memory_gateway = MemoryGateway(memories, {})
    tool_registry = ToolRegistry()
    tool_gateway = ToolGateway(tool_registry, set())

    # 3. Instantiate Sensor Streamer & Vision Skill
    dataset_path = r"d:\agentic_agriculture\datasets\Smart Agriculture and Plant Health Monitoring using IoT"
    streamer = IoTStreamer(dataset_dir=dataset_path, loop=True)

    vision_model_path = r"D:\agentic_agriculture\model_\best_model_v5.pth"
    diagnosis_skill = TomatoDiagnosisSkill(model_path=vision_model_path)

    # 4. Instantiate Perception & Reasoning Agents
    perception_agent = PerceptionAgent(
        message_bus=bus,
        memory_gateway=memory_gateway,
        tool_gateway=tool_gateway,
        telemetry_streamer=streamer,
        diagnosis_skill=diagnosis_skill,
    )

    reasoning_agent = ReasoningAgent(
        message_bus=bus,
        memory_gateway=memory_gateway,
        tool_gateway=tool_gateway,
        ollama_model_name="gemma4:4b-q4_K_M",
        ollama_host="http://localhost:11434",
    )

    # Start Agents (enables bus subscriptions)
    perception_agent.start()
    reasoning_agent.start()

    # 5. Capture Published Hypothesis Messages for Terminal Display
    latest_hypotheses: List[ACAMessage] = []

    def on_hypothesis_received(msg: ACAMessage) -> None:
        latest_hypotheses.append(msg)

    bus.subscribe(MessageType.HYPOTHESIS, on_hypothesis_received)

    # 6. Prepare Sample Vision Asset
    sample_image_path = find_sample_image(r"d:\agentic_agriculture\datasets\Tomato_dataset")
    print(f"\n[ASSETS] Using Vision Input Image: {os.path.basename(sample_image_path)}", flush=True)
    print(f"[ASSETS] Total IoT Telemetry Records: {streamer.total_records}", flush=True)
    print("[SYSTEM] Starting 5-step live cognitive simulation...\n", flush=True)

    # 7. Run 5-Step Simulation Loop
    NUM_STEPS = 5

    for step_idx in range(1, NUM_STEPS + 1):
        latest_hypotheses.clear()
        print("-" * 80, flush=True)
        print(f" TIME STEP {step_idx} / {NUM_STEPS} | {time.strftime('%H:%M:%S')}", flush=True)
        print("-" * 80, flush=True)

        # Trigger Perception Agent tick (fetches IoT row + runs vision inference + publishes OBSERVATION)
        obs_msg = perception_agent.perceive(image_path=sample_image_path)
        obs_payload = obs_msg.payload
        measurements = obs_payload.measurements

        # Print Ingested Telemetry
        print(" [1] IOT TELEMETRY INGESTION (8 CHANNELS)", flush=True)
        print(f"     • Entry ID     : {int(measurements.get('Entry_id', 0))}", flush=True)
        print(f"     • Timestamp    : {obs_msg.metadata.get('raw_timestamp', 'N/A')}", flush=True)
        print(f"     • Air Temp     : {measurements.get('Environment Temperature', 0.0):.1f} °C", flush=True)
        print(f"     • Air Humidity : {measurements.get('Environment Humidity', 0.0):.1f} %", flush=True)
        print(f"     • Soil Moisture: {measurements.get('Soil Moisture', 0.0):.1f} %", flush=True)
        print(f"     • Soil Temp    : {measurements.get('Soil Temperature', 0.0):.1f} °C", flush=True)
        print(f"     • Soil pH      : {measurements.get('Soil pH', 0.0):.2f}", flush=True)
        print(f"     • Battery Volts: {measurements.get('Solar Panel Battery Voltage', 0.0):.3f} V", flush=True)
        print(f"     • Water TDS    : {measurements.get('Water TDS', 0.0):.1f} mg/L", flush=True)

        # Print Vision Model Diagnostic Output
        vision_class = obs_msg.metadata.get("vision_predicted_class", "unknown")
        vision_conf = obs_msg.confidence * 100.0
        infer_time = measurements.get("vision_inference_time_ms", 0.0)

        print("\n [2] PERCEPTION VISION MODEL (CondConViT_V2)", flush=True)
        print(f"     • Diagnosis    : {vision_class}", flush=True)
        print(f"     • Confidence   : {vision_conf:.2f} %", flush=True)
        print(f"     • Latency      : {infer_time:.2f} ms", flush=True)

        # Print Sensor Fusion Reasoning Output from ReasoningAgent
        print("\n [3] REASONING & SENSOR FUSION (Ollama gemma4:4b-q4_K_M)", flush=True)
        if latest_hypotheses:
            hyp_msg = latest_hypotheses[-1]
            hyp_payload = hyp_msg.payload
            llm_text = hyp_msg.metadata.get("llm_reasoning", hyp_payload.suspected_cause)
            print(f"     • Suspected Cause  : {hyp_payload.suspected_cause[:60]}...", flush=True)
            print(f"     • Prior Conf / Ratio: {hyp_payload.prior_probability:.2f} / {hyp_payload.likelihood_ratio:.2f}", flush=True)
            print(f"     • Agronomic Guidance:\n       \"{llm_text}\"", flush=True)
        else:
            print("     • [WARNING] No hypothesis received from ReasoningAgent.", flush=True)

        print("\n" + "." * 80, flush=True)
        time.sleep(2)  # Hardware throttling delay

    # Clean shutdown
    perception_agent.stop()
    reasoning_agent.stop()

    print("\n" + "=" * 80, flush=True)
    print("      ACA MILESTONE 5 LIVE SIMULATION COMPLETE (5/5 TICKS)", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
