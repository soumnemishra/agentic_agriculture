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
│   ├── world_model/              # Dynamic farm state representation
│   ├── digital_twin/             # Predictive simulation using World Model
│   │
│   ├── tools/                    # Tool interfaces and registry
│   │   ├── base_tool.py          # Abstract BaseTool with schema and validation
│   │   └── registry.py           # Centralised ToolRegistry
│   │
│   ├── skills/                   # Reusable agricultural competencies
│   │   ├── base_skill.py         # Abstract BaseSkill (composes tools)
│   │   └── registry.py           # Centralised SkillRegistry
│   │
│   ├── orchestration/            # Workflow coordination
│   │   ├── schemas.py            # ACA Communication Protocol (10 message types)
│   │   ├── message_bus.py        # Pub/sub MessageBus with priority queuing
│   │   ├── workflow_engine.py    # Task DAG management and replanning
│   │   ├── scheduler.py          # Runtime assignment (Edge/Cloud)
│   │   └── supervisor.py         # Mission management interface
│   │
│   ├── agents/                   # Specialised layer agents
│   │   └── base_agent.py         # Contract-enforced BaseAgent
│   │
│   ├── edge/                     # Edge runtime shell
│   └── cloud/                    # Cloud runtime shell
│
├── tests/unit/                   # Unit tests (69 tests, all passing)
│   └── test_milestone1.py
│
├── simulation/                   # Crop growth simulation (future)
├── evaluation/                   # Comparative evaluation (future)
├── docs/                         # Research documentation
├── paper/                        # Academic paper drafts
└── datasets/                     # Farm topology and knowledge data
```

---

## Separation of Concerns

| Subsystem | Mandate | Principle |
|:---|:---|:---|
| **Cognition** | Evaluates, reasons, decides | _Decides_ |
| **Skills** | Pre-packaged agricultural workflows | _Knows How_ |
| **Tools** | Standardised environment interactions | _Interacts_ |
| **Memory** | Working, episodic, semantic, farm state | _Remembers_ |
| **Knowledge** | External references, policies, RAG | _Informs_ |
| **World Model** | Dynamic physical farm state | _Represents Reality_ |
| **Digital Twin** | Future trajectory simulation | _Predicts Reality_ |
| **Orchestration** | Mission → workflow → task scheduling | _Coordinates_ |
| **Edge / Cloud** | Deployment shells | _Deploys_ |

---

## Quick Start

```bash
# Run all unit tests
python -m pytest tests/unit/test_milestone1.py -v

# Import and use components
python -c "
from aca.config import ACAConfig
from aca.orchestration.message_bus import MessageBus
from aca.orchestration.schemas import create_message, MessageType, MissionPayload

config = ACAConfig()
bus = MessageBus(config.message_bus)

msg = create_message(
    source='supervisor',
    destination='BROADCAST',
    message_type=MessageType.MISSION,
    payload=MissionPayload(mission_id='m1', objective='Optimize yield'),
)
print(f'Published to {bus.publish(msg)} subscribers')
"
```

---

## Architecture Principles

1. **Dependency Injection** — All components receive dependencies via constructors, never global singletons.
2. **Abstract Base Classes** — `BaseTool`, `BaseSkill`, `BaseAgent` define contracts; implementations are pluggable.
3. **Agent Contracts** — Every agent declares inputs, outputs, memory permissions, tool allowlists, latency budgets, and failure modes. Violations raise `AgentContractViolation`.
4. **SOLID** — Single responsibility per module; open for extension via registries; Liskov-safe ABCs; interface segregation via gateways; dependency inversion throughout.

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

| Milestone | Status | Tests |
|:---|:---|:---|
| **M1: Core Infrastructure** | ✅ Complete | 69/69 passing |
| **M2: Cognitive Core** | ✅ Complete | 20/20 passing |
| M3: World Model & Digital Twin | 🔲 Planned | — |
| M4: Knowledge Layer & RAG | 🔲 Planned | — |
| M5: Simulation & Evaluation | 🔲 Planned | — |
