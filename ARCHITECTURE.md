# Architecture Documentation

## Dual-AI Cascade Memory System

This document provides detailed information about the architecture and implementation of the Hippo dual-agent memory system.

## Overview

The system implements a neuromorphic architecture inspired by the hippocampus-neocortex memory loop in human cognition. It consists of three main components working in concert:

1. **Volatile Agent** (Hippocampal-inspired)
2. **Neocortex Agent** (Abstraction and consolidation)
3. **Semantic Bridge** (Shared concept space)

## Component Details

### 1. Volatile Agent (`hippo/volatile_agent.py`)

**Purpose**: Fast encoding of current experiences and short-term memory.

**Key Features**:
- Sliding window context buffer (default 128 tokens)
- Rapid encoding using transformer-based models
- Summary vector generation
- Novelty detection through comparison with semantic bridge
- Replay vector production

**Architecture**:
```
Input Tokens → Context Buffer → Encoder → Embeddings → Summarizer → Summary Vector
                                                              ↓
                                               Compare with Semantic Bridge
                                                              ↓
                                               Generate Replay Vector
```

**Parameters**:
- `window_size`: Size of sliding context window (default: 128)
- `embedding_dim`: Dimension of concept embeddings (default: 768)
- `device`: Computation device ('cpu' or 'cuda')

### 2. Neocortex Agent (`hippo/neocortex_agent.py`)

**Purpose**: Gradual abstraction, pattern recognition, and long-term memory consolidation.

**Key Features**:
- Latent memory storage with configurable capacity
- Clustering-based generalization (MiniBatchKMeans)
- Pattern-based prediction
- Toki pona language output for conceptual abstraction
- Memory pruning strategies

**Architecture**:
```
Replay Vector → Compress → Latent Memory → Cluster → Abstract Concepts
                                               ↓
                                          Predictions → Toki Pona Output
```

**Parameters**:
- `embedding_dim`: Dimension of concept embeddings (default: 768)
- `max_memory_size`: Maximum memory capacity (default: 1000)
- `n_clusters`: Number of concept clusters (default: 50)
- `consolidation_threshold`: Minimum vectors before clustering (default: 10)

### 3. Semantic Bridge (`hippo/semantic_bridge.py`)

**Purpose**: Shared latent space for cross-agent communication and concept retrieval.

**Key Features**:
- FAISS-based similarity search
- Efficient nearest neighbor queries
- Concept ID tracking
- Importance weighting
- Training for IVF indexes

**Architecture**:
```
Concept Vectors → FAISS Index → Similarity Search
                       ↓
                  Query Interface → Retrieve Concepts
```

**Parameters**:
- `embedding_dim`: Dimension of embeddings (default: 768)
- `index_type`: FAISS index type ('flat' or 'ivf')

## Inference Cycle

The main inference loop (`hippo/loop.py`) orchestrates the interaction between agents:

### Cycle Steps (Target: 50-60ms)

1. **Observation**: Volatile agent receives new input tokens
2. **Encoding**: Current context is encoded into embeddings
3. **Summarization**: Embeddings compressed into summary vector
4. **Comparison**: Summary compared with semantic bridge for novelty
5. **Replay Generation**: Replay vector created blending summary and novelty
6. **Bridge Update**: Summary added to semantic bridge
7. **Neocortex Processing**: Replay vector received and compressed
8. **Consolidation**: Periodic memory consolidation (every N cycles)
9. **Generalization**: Cluster formation and pattern abstraction
10. **Prediction**: Forward predictions generated
11. **Output**: Toki pona sentence produced

### Timing Considerations

- Target cycle time: 50-60ms
- Internal processing target: ~7ms per component
- Optimization strategies:
  - Model quantization
  - Batch processing
  - Efficient indexing (FAISS)
  - Periodic vs. continuous consolidation

## Memory Mechanisms

### Consolidation

Memory consolidation happens periodically:
- Clustering of latent memory vectors
- Merging similar concepts
- Pruning redundant memories
- Stabilizing abstract patterns

### Forgetting Strategies

Three forgetting strategies are implemented:
1. **Age-based**: Remove oldest memories
2. **Frequency-based**: Remove least accessed memories
3. **Importance-based**: Remove least important memories

### Replay

Replay mechanism simulates "sleep" consolidation:
- Sample memories from replay buffer
- Send to neocortex for re-processing
- Strengthen important patterns
- Integrate new knowledge

## Toki Pona Abstraction

### Why Toki Pona?

Toki pona is a minimal constructed language with ~120 words, making it ideal for:
- Forcing simple conceptual representations
- Reducing semantic drift
- Ensuring stable abstractions
- Efficient concept communication

### Vector-to-Word Mapping

Concept vectors are mapped to toki pona words based on:
- **Magnitude**: Strong signals → positive/negative words
- **Variance**: High variance → action/change words
- **Positive ratio**: Cognitive/knowledge words
- **Distribution**: Temporal/existence words

### Sentence Construction

Simple grammatical patterns:
- `[subject] li [predicate]`
- `[subject] li [predicate] [modifiers]`

## Compression Techniques

### Embedding Compression (`hippo/compression.py`)

Multiple compression methods:
1. **Mean pooling**: Average across sequence
2. **Max pooling**: Maximum activation
3. **Weighted average**: Recent tokens weighted higher

### Concept Merging

Similar concepts merged based on:
- Cosine similarity threshold (default: 0.9)
- Clustering algorithms (k-means, hierarchical)
- Novelty detection

## Performance Optimization

### Hardware Considerations

- **CPU**: Suitable for small models (embedding_dim ≤ 256)
- **GPU**: Required for larger models (embedding_dim ≥ 512)
- **Memory**: ~2GB RAM for default configuration

### Model Size Recommendations

| Use Case | Embedding Dim | Window Size | Memory Size |
|----------|--------------|-------------|-------------|
| Demo     | 128          | 32          | 100         |
| Standard | 768          | 128         | 1000        |
| Large    | 1024         | 256         | 5000        |

### Optimization Techniques

1. **Model Quantization**: Int8/Float16 for faster inference
2. **Batch Processing**: Process multiple inputs together
3. **Lazy Consolidation**: Consolidate only when threshold reached
4. **Efficient Indexing**: Use IVF indexes for large concept spaces

## Scientific Foundations

### Key Research Areas

1. **Complementary Learning Systems (CLS)**
   - Fast hippocampal learning
   - Slow neocortical consolidation
   - Avoiding catastrophic interference

2. **Memory Consolidation**
   - Replay mechanisms
   - Pattern abstraction
   - Long-term stability

3. **Artificial Hippocampus Networks**
   - Compressed KV cache
   - Recurrent long-term modules
   - Efficient context modeling

### Related Papers

- "Neuroplasticity Meets Artificial Intelligence" (PubMed, 2024)
- "Artificial Hippocampus Networks for Long-Context Modeling" (Emergent Mind)
- "Transformer as Hippocampal Memory Consolidation" (NeurIPS 2023)

## Extension Points

### Adding New Compression Methods

```python
def custom_compress(embeddings, **kwargs):
    # Your compression logic
    return compressed_vector

# Register in compression.py
```

### Custom Toki Pona Mappings

```python
# Extend TOKI_PONA_VOCAB in toki_pona.py
TOKI_PONA_VOCAB["custom_category"] = ["word1", "word2"]
```

### Alternative Clustering

```python
from sklearn.cluster import DBSCAN

# Use in neocortex_agent.py
clusterer = DBSCAN(eps=0.5)
```

## Future Enhancements

1. **Multimodal Support**: Images, audio, video
2. **Attention Mechanisms**: Learned importance weighting
3. **Online Learning**: Continuous model adaptation
4. **Distributed Processing**: Multi-agent systems
5. **Advanced Replay**: Prioritized experience replay
6. **Meta-Learning**: Learn-to-consolidate strategies

## Troubleshooting

### Common Issues

**Slow Cycle Times**
- Reduce embedding dimensions
- Use smaller window size
- Enable GPU acceleration
- Use quantized models

**Memory Overflow**
- Reduce max_memory_size
- More aggressive consolidation
- Implement forgetting strategies

**Poor Abstraction**
- Increase consolidation_threshold
- Use more clusters
- Adjust similarity thresholds

## References

See README.md for full list of scientific references.
