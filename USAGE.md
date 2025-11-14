# Usage Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/cuavejourney-sketch/hippo.git
cd hippo

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### Basic Usage

```python
from hippo import VolatileAgent, NeocortexAgent, SemanticBridge, run_dual_cascade
from hippo.loop import create_text_stream

# Initialize agents
volatile = VolatileAgent(window_size=128, embedding_dim=768)
neocortex = NeocortexAgent(embedding_dim=768, max_memory_size=1000)
bridge = SemanticBridge(embedding_dim=768)

# Create input stream
texts = ["learning patterns", "recognizing concepts", "forming memories"]
input_stream = create_text_stream(texts)

# Run the cascade
for result in run_dual_cascade(volatile, neocortex, bridge, input_stream):
    print(f"Cycle {result['cycle']}: {result['toki_output']}")
    print(f"Time: {result['cycle_time_ms']:.2f}ms")
```

### Custom Configuration

```python
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize with config
volatile = VolatileAgent(
    window_size=config['volatile']['window_size'],
    embedding_dim=config['volatile']['embedding_dim'],
    device=config['volatile']['device']
)
```

## Advanced Features

### Memory Consolidation

```python
from hippo.memory import MemoryManager, simulate_sleep_replay

# Create memory manager
memory_manager = MemoryManager(
    replay_buffer_size=100,
    forgetting_strategy="importance"
)

# Add memories during processing
memory_manager.add_to_replay_buffer(summary_vector, importance=0.8)

# Simulate sleep consolidation
simulate_sleep_replay(memory_manager, neocortex, n_replay_cycles=10)
```

### Novelty Detection

```python
# Compare with semantic bridge
summary, difference = volatile.reconstruct_and_compare(summary, bridge)

# Calculate novelty
from hippo.compression import calculate_novelty
novelty_score = calculate_novelty(summary, neocortex.latent_memory)

print(f"Novelty: {novelty_score:.2f}")  # 0-1 range
```

### Custom Toki Pona Mappings

```python
from hippo.toki_pona import TOKI_PONA_VOCAB

# Add custom category
TOKI_PONA_VOCAB["custom"] = ["word1", "word2", "word3"]

# Use in rendering
sentence = neocortex.speak_toki_pona(predictions)
```

### Inference Cycle Control

```python
from hippo.loop import InferenceCycle

# Create cycle with custom parameters
cycle = InferenceCycle(
    volatile, neocortex, bridge,
    target_cycle_ms=50.0,
    consolidation_interval=5
)

# Manual stepping
for i in range(100):
    result = cycle.step(input_tokens)
    
    # Check timing
    if result['cycle_time_ms'] > 60:
        print("Warning: Cycle time exceeded target")
    
    # Get statistics
    if i % 10 == 0:
        stats = cycle.get_stats()
        print(f"Avg time: {stats['avg_cycle_time_ms']:.2f}ms")
```

## Examples

### Example 1: Basic Demo

```bash
python examples/basic_demo.py
```

Demonstrates:
- Agent initialization
- Basic inference cycle
- Toki pona output
- Performance metrics

### Example 2: Advanced Demo

```bash
python examples/advanced_demo.py
```

Demonstrates:
- Memory consolidation
- Sleep replay simulation
- Novelty detection
- Multi-phase learning

## Configuration Options

### Volatile Agent

| Parameter | Default | Description |
|-----------|---------|-------------|
| window_size | 128 | Context window size |
| embedding_dim | 768 | Embedding dimension |
| device | "cpu" | Computation device |

### Neocortex Agent

| Parameter | Default | Description |
|-----------|---------|-------------|
| embedding_dim | 768 | Embedding dimension |
| max_memory_size | 1000 | Memory capacity |
| n_clusters | 50 | Number of clusters |
| consolidation_threshold | 10 | Min for clustering |

### Semantic Bridge

| Parameter | Default | Description |
|-----------|---------|-------------|
| embedding_dim | 768 | Embedding dimension |
| index_type | "flat" | FAISS index type |

## Performance Tuning

### Reducing Cycle Time

1. **Reduce embedding dimensions**
   ```python
   volatile = VolatileAgent(embedding_dim=256)  # Faster
   ```

2. **Use smaller window size**
   ```python
   volatile = VolatileAgent(window_size=64)  # Less processing
   ```

3. **Enable GPU acceleration**
   ```python
   volatile = VolatileAgent(device="cuda")  # Much faster
   ```

4. **Reduce cluster count**
   ```python
   neocortex = NeocortexAgent(n_clusters=10)  # Faster clustering
   ```

### Managing Memory

1. **Limit memory size**
   ```python
   neocortex = NeocortexAgent(max_memory_size=100)
   ```

2. **Aggressive consolidation**
   ```python
   neocortex = NeocortexAgent(consolidation_threshold=5)
   ```

3. **Forgetting strategies**
   ```python
   memory_manager = MemoryManager(forgetting_strategy="importance")
   ```

## Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_volatile_agent.py -v
```

Run with coverage:
```bash
pytest tests/ --cov=hippo --cov-report=html
```

## Troubleshooting

### Issue: Slow cycle times

**Solution**: Reduce model size or use GPU
```python
volatile = VolatileAgent(embedding_dim=128, device="cuda")
```

### Issue: Memory overflow

**Solution**: Reduce memory capacity and consolidate more frequently
```python
neocortex = NeocortexAgent(max_memory_size=100)
cycle = InferenceCycle(consolidation_interval=5)
```

### Issue: Poor abstraction quality

**Solution**: Increase consolidation threshold and cluster count
```python
neocortex = NeocortexAgent(
    consolidation_threshold=20,
    n_clusters=100
)
```

### Issue: High memory usage

**Solution**: Use smaller embeddings and clear bridge periodically
```python
bridge = SemanticBridge(embedding_dim=256)
# Periodically
bridge.clear()  # Clear old concepts
```

## API Reference

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed API documentation.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file.

## References

- [Complementary Learning Systems](https://pubmed.ncbi.nlm.nih.gov/39595874/)
- [Artificial Hippocampus Networks](https://www.emergentmind.com/papers/2510.07318)
- [Transformer as Memory Consolidation](https://proceedings.neurips.cc/paper_files/paper/2023/file/2f1eb4c897e63870eee9a0a0f7a10332-Paper-Conference.pdf)
