# Agricultural Cognitive Architecture (ACA) v1.0

A layered, cognition-oriented architecture for autonomous precision farming. ACA separates cognitive reasoning from domain-specific agents, physical runtimes, and deployment environments.

---

## Repository Structure

```
agentic_agriculture/
│
├── aca/                          # Core ACA package
│   ├── config.py                 # Centralised configuration (frozen dataclasses)
│   ├── logging_config.py         # Structured logging with trace-ID propagation
│   │
│   ├── cognition/                # Cognitive layers (Goal, Perception, Reasoning, Planning, Execution, Learning, Meta-Cognition)
│   │   ├── goal_management/
│   │   ├── perception/
│   │   ├── reasoning/
│   │   ├── planning/
│   │   ├── execution/
│   │   ├── learning/
│   │   └── meta_cognition/
│   │
│   ├── memory/                   # Standalone memory subsystem
│   │   ├── working_memory.py     # Transient context (bounded, FIFO eviction)
│   │   ├── episodic_memory.py    # Append-only intervention history
│   │   ├── semantic_memory.py    # Internal agronomic knowledge (freezable)
│   │   └── farm_memory.py        # Spatial-temporal farm topology
│   │
│   ├── knowledge/                # External knowledge layer (RAG, KG, policies)
│   │   ├── interfaces.py         # Vector store & embedder abstractions
│   │   ├── local_store.py        # Local vector database & RAG engine
│   │   └── schemas.py            # Knowledge document dataclasses
│   │
│   ├── world_model/              # Dynamic farm state representation
│   │   ├── graph_engine.py       # Graph-based farm topology & state snapshots
│   │   ├── interfaces.py         # WorldModel gateway contracts
│   │   └── schemas.py            # Graph node and edge schemas
│   │
│   ├── digital_twin/             # Predictive simulation using World Model
│   │   ├── engine.py             # Biophysical dynamic simulation engine
│   │   ├── interfaces.py         # Digital twin contracts
│   │   └── schemas.py            # State trajectories and simulation parameters
│   │
│   ├── tools/                    # Tool interfaces and registry
│   │   ├── base_tool.py          # Abstract BaseTool with schema and validation
│   │   ├── registry.py           # Centralised ToolRegistry
│   │   ├── agronomy_tool.py      # Standard agronomic calculation tools
│   │   └── actuator_tools.py     # Physical valve & treatment alert tools
│   │
│   ├── skills/                   # Reusable agricultural competencies
│   │   ├── base_skill.py         # Abstract BaseSkill (composes tools & models)
│   │   ├── registry.py           # Centralised SkillRegistry
│   │   └── tomato_diagnosis_skill.py # CondConViT_V2 vision diagnosis skill
│   │
│   ├── orchestration/            # Workflow coordination
│   │   ├── schemas.py            # ACA Communication Protocol (10 message types)
│   │   ├── message_bus.py        # Pub/sub MessageBus with priority queuing
│   │   ├── workflow_engine.py    # Task DAG management and replanning
│   │   ├── scheduler.py          # Runtime assignment (Edge/Cloud)
│   │   └── supervisor.py         # Mission management interface
│   │
│   ├── agents/                   # Specialised contract-enforced layer agents
│   │   ├── base_agent.py         # BaseAgent with MemoryGateway & ToolGateway
│   │   ├── perception_agent.py   # Multi-modal telemetry & vision perception
│   │   ├── reasoning_agent.py    # Sensor fusion & Bayesian etiology reasoning
│   │   ├── planning_agent.py     # Agronomic action matrix planning
│   │   └── execution_agent.py    # Actuator dispatch & feedback logging
│   │
│   ├── edge/                     # Edge runtime shell
│   └── cloud/                    # Cloud runtime shell
│
├── simulation/                   # Telemetry streaming & farm simulation
│   └── telemetry_streamer.py     # 8-channel synchronized IoT microclimate streamer
│
├── evaluation/                   # Experimental evaluation framework
│   ├── run_experiment.py         # 100-step closed-loop simulation runner
│   └── metrics_logger.py         # Causal cognitive metrics logger (CSV/JSONL)
│
├── datasets/                     # IoT telemetry, leaf images, and benchmark logs
│   ├── experiment_results.csv    # Evaluated cycle metrics output
│   └── experiment_traces.jsonl   # Full causal message traces output
│
├── tests/unit/                   # Comprehensive unit test suite (263 tests passing)
│   ├── test_milestone1.py        # M1: Core infrastructure & MessageBus (69 tests)
│   ├── test_milestone2.py        # M2: Cognitive core (20 tests)
│   ├── test_milestone3_world_model.py # M3: World Model graph engine (56 tests)
│   ├── test_milestone3_digital_twin.py # M3: Digital Twin simulation (44 tests)
│   ├── test_milestone4_knowledge.py   # M4: Knowledge layer & RAG (46 tests)
│   ├── test_milestone5_perception.py  # M5: Multi-modal vision perception (15 tests)
│   ├── test_milestone6_execution.py   # M6: Planning & actuator execution (10 tests)
│   └── test_milestone7_evaluation.py  # M7: Metrics logging & causal tracing (3 tests)
│
├── docs/                         # Scientific and research documentation
│   ├── aca_architectural_spec.md # Formal scientific design specification
│   └── methology.md              # Research methodology & design principles
│
└── paper/                        # Academic paper manuscripts
```

---

## Separation of Concerns

| Subsystem | Mandate | Principle |
|:---|:---|:---|
| **Cognition** | Evaluates, reasons, plans, decides | _Decides_ |
| **Skills** | Pre-packaged agricultural workflows (e.g. CondConViT_V2) | _Knows How_ |
| **Tools** | Standardised environment & actuator interactions | _Interacts_ |
| **Memory** | Working, episodic, semantic, farm topology | _Remembers_ |
| **Knowledge** | External references, agronomic guidelines, RAG | _Informs_ |
| **World Model** | Dynamic physical farm state graph | _Represents Reality_ |
| **Digital Twin** | Future trajectory biophysical simulation | _Predicts Reality_ |
| **Orchestration** | Mission → workflow → task scheduling | _Coordinates_ |
| **Agents** | Perception, Reasoning, Planning, Execution | _Executes Arc_ |
| **Edge / Cloud** | Deployment shells | _Deploys_ |

---

## Quick Start

### 1. Run Unit Tests (263 Passing)
```bash
# Run all unit tests across all milestones (M1–M7)
python -m pytest tests/unit -v
```

### 2. Run the 100-Step Experimental Simulation & Evaluation
```bash
# Execute the full closed-loop cognitive pipeline
python evaluation/run_experiment.py

# Optional: Run with local Ollama LLM reasoning
python evaluation/run_experiment.py --model gemma4:4b-q4_K_M --timeout 3.0 --zone greenhouse_bay_1
```

### 3. Test Components Interactively
```python
from aca.config import ACAConfig
from aca.orchestration.message_bus import MessageBus
from aca.orchestration.schemas import create_message, MessageType, MissionPayload

config = ACAConfig()
bus = MessageBus(config.message_bus)

msg = create_message(
    source="supervisor",
    destination="BROADCAST",
    message_type=MessageType.MISSION,
    payload=MissionPayload(mission_id="m1", objective="Optimize tomato yield & suppress late blight"),
)
print(f"Published to {bus.publish(msg)} subscribers")
```

---

## Architecture Principles

1. **Dependency Injection** — All components receive dependencies via constructors, never global singletons.
2. **Abstract Base Classes** — `BaseTool`, `BaseSkill`, `BaseAgent` define contracts; implementations are pluggable.
3. **Agent Contracts** — Every agent declares inputs, outputs, memory permissions, tool allowlists, latency budgets, and failure modes. Violations raise `AgentContractViolation`.
4. **SOLID & Proxy Pattern** — Single responsibility per module; `MemoryGateway` and `ToolGateway` act as permission-checked proxies.
5. **Epistemic Traceability** — Every intervention carries an immutable causal provenance chain: `Observation` $\rightarrow$ `Hypothesis` $\rightarrow$ `Decision` $\rightarrow$ `Action` $\rightarrow$ `Feedback`.

---

## ACA Communication Protocol

All messages use the `ACAMessage` envelope:

| Field | Type | Description |
|:---|:---|:---|
| `uuid` | `str` | UUIDv4 |
| `timestamp` | `str` | ISO-8601 |
| `source` | `str` | Originating component |
| `destination` | `str` | Target or `BROADCAST` |
| `message_type` | `MessageType` | Category enum |
| `confidence` | `float` | [0.0, 1.0] |
| `priority` | `int` | [1 (low) … 5 (critical)] |
| `payload` | Typed dataclass | Category-specific content |
| `metadata` | `dict` | Trace IDs, tags |

**10 Message Categories:** Mission · Goal · Task · Observation · Evidence · Hypothesis · Belief · Decision · Explanation · Feedback

---

## Milestone Status

| Milestone | Subsystem / Capability | Status | Unit Tests |
|:---|:---|:---:|:---:|
| **M1: Core Infrastructure** | MessageBus, Pub/Sub, Schemas, Config, Tools & Skills Registries | ✅ Complete | 69/69 passing |
| **M2: Cognitive Core** | Perception Processor, Reasoning Engine, Planning Engine | ✅ Complete | 20/20 passing |
| **M3: World Model & Digital Twin** | Farm Topology Graph, Snapshots, Biophysical Simulation | ✅ Complete | 100/100 passing |
| **M4: Knowledge Layer & RAG** | Vector Database, Embedder, Agronomic Reference Retrieval | ✅ Complete | 46/46 passing |
| **M5: Perception & Multi-Modal Fusion** | CondConViT_V2 Vision Model, IoT Telemetry Streamer, PerceptionAgent | ✅ Complete | 15/15 passing |
| **M6: Planning & Actuator Execution** | Action Matrix Planning, Actuators (Valves/Alerts), ExecutionAgent | ✅ Complete | 10/10 passing |
| **M7: Evaluation & Closed-Loop Simulation** | 100-step simulation runner, Causal Metrics Logger (CSV/JSONL) | ✅ Complete | 3/3 passing |
| **Total** | **Full Agricultural Cognitive Architecture (ACA v1.0)** | ✅ **Complete** | **263/263 passing** |

