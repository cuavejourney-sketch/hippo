# Hippo: Dual-AI Cascade Memory System

A neuromorphic dual-agent AI system inspired by the hippocampus-neocortex memory architecture in human cognition.

## Overview

Hippo implements a two-agent system that mirrors the complementary learning systems (CLS) theory:

- **Volatile Agent (Hippocampal)**: Fast encoding of "nowness", maintains working context, rapid adaptation
- **Neocortex Agent**: Gradual abstraction and consolidation, uses toki pona for simple conceptual representations
- **Semantic Bridge**: Shared latent embedding space for cross-agent communication

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Dual-AI Memory Architecture                │
│                                                             │
│   +-----------------+           Shared Semantic Bridge      │
│   | Volatile Agent  |<──────concept embeddings / summaries──┤
│   | ("Hippocampal") |                                     ▲  │
│   +-----------------+                                     │  │
│         ↑ forward pass / replay                             │
│         │ reconstruct current state                         │
│         │                                                     │
│         └──────────────┐                                      │
│                        │                                      │
│                +-------------------+                         │
│                | Neocortex Agent   | (Toki Pona)               │
│                +-------------------+                         │
│                 ↑ compress / abstract                         │
│                 ↓ feedback (pattern summary)                  │
│                                                             │
│     Latent Cache / Swap Storage  <──  Shared Memory Pool     │
│            (concept embeddings, compressed ideas)           │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

- **Complementary Learning Systems**: Fast hippocampal-like learning + slow neocortical consolidation
- **Memory Consolidation**: Periodic replay cycles compress and abstract experiences
- **Semantic Bridge**: Shared concept embedding space for agent communication
- **Toki Pona Abstraction**: Simple 120-word language for conceptual representations
- **Efficient Memory**: Lossy but semantically meaningful latent state
- **Fast Inference**: Target 50-60ms per cycle

## Scientific Foundations

Based on latest neuroscience and AI research:
- Complementary Learning Systems (CLS) theory
- Artificial Hippocampus Networks (AHN)
- Memory consolidation and replay mechanisms
- Catastrophic interference mitigation
- Continual learning architectures

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from hippo import VolatileAgent, NeocortexAgent, SemanticBridge, run_dual_cascade

# Initialize agents
volatile = VolatileAgent(window_size=128)
neocortex = NeocortexAgent()
bridge = SemanticBridge()

# Run the dual cascade
for output in run_dual_cascade(volatile, neocortex, bridge, input_stream):
    print("Neocortex (toki pona):", output)
```

## Components

- `hippo/volatile_agent.py`: Hippocampal-inspired rapid encoding agent
- `hippo/neocortex_agent.py`: Abstraction and consolidation agent
- `hippo/semantic_bridge.py`: Shared concept embedding space
- `hippo/memory.py`: Memory management and consolidation utilities
- `hippo/compression.py`: Embedding compression and clustering
- `hippo/toki_pona.py`: Toki pona language renderer
- `hippo/loop.py`: Main inference loop

## Usage Examples

See `examples/` directory for detailed examples.

## References

- [Complementary Learning Systems](https://pubmed.ncbi.nlm.nih.gov/39595874/)
- [Artificial Hippocampus Networks](https://www.emergentmind.com/papers/2510.07318)
- [Transformer as Hippocampal Memory Consolidation](https://proceedings.neurips.cc/paper_files/paper/2023/file/2f1eb4c897e63870eee9a0a0f7a10332-Paper-Conference.pdf)

## License

MIT License
