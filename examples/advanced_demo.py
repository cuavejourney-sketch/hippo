"""
Advanced example demonstrating memory consolidation and replay mechanisms.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hippo import VolatileAgent, NeocortexAgent, SemanticBridge
from hippo.loop import InferenceCycle
from hippo.memory import MemoryManager, simulate_sleep_replay
from hippo.toki_pona import translate_to_english
import numpy as np


def demonstrate_consolidation():
    """Demonstrate memory consolidation and replay."""
    
    print("=" * 70)
    print("Advanced Demo: Memory Consolidation & Replay")
    print("=" * 70)
    print()
    
    # Initialize agents
    print("Initializing system...")
    volatile = VolatileAgent(
        window_size=64,
        embedding_dim=128,
        device="cpu"
    )
    
    neocortex = NeocortexAgent(
        embedding_dim=128,
        max_memory_size=50,
        n_clusters=8,
        consolidation_threshold=5
    )
    
    bridge = SemanticBridge(embedding_dim=128)
    
    # Create inference cycle
    cycle = InferenceCycle(
        volatile, neocortex, bridge,
        target_cycle_ms=60.0,
        consolidation_interval=5
    )
    
    # Create memory manager
    memory_manager = MemoryManager(
        replay_buffer_size=50,
        forgetting_strategy="importance"
    )
    
    print("✓ System ready")
    print()
    
    # Phase 1: Learning phase
    print("Phase 1: Learning Phase")
    print("-" * 70)
    
    learning_texts = [
        "machine learning concepts",
        "neural network training",
        "deep learning models",
        "pattern recognition systems",
        "artificial intelligence research",
        "cognitive architecture design",
        "memory consolidation process",
        "knowledge representation methods",
        "semantic understanding models",
        "computational neuroscience",
    ]
    
    for i, text in enumerate(learning_texts):
        # Tokenize (simple hash-based)
        tokens = [hash(word) % 10000 for word in text.split()]
        
        # Run cycle
        result = cycle.step(tokens)
        
        # Add to replay buffer
        embeddings = volatile.encode_current()
        summary = volatile.summarize(embeddings)
        importance = np.linalg.norm(result['cycle'] / len(learning_texts))
        memory_manager.add_to_replay_buffer(summary, importance=importance)
        
        print(f"Cycle {result['cycle']:2d}: {result['toki_output']:<20} "
              f"({result['cycle_time_ms']:.2f}ms)")
    
    print()
    print(f"Learned {len(learning_texts)} concepts")
    print(f"Memory utilization: {neocortex.get_memory_stats()['utilization']:.1%}")
    print()
    
    # Phase 2: Consolidation phase
    print("Phase 2: Consolidation Phase")
    print("-" * 70)
    
    print("Performing memory consolidation...")
    neocortex.consolidate()
    
    stats = neocortex.get_memory_stats()
    print(f"✓ Consolidation complete")
    print(f"  Memories: {stats['total_memories']}")
    print(f"  Clusters: {stats['n_clusters']}")
    print()
    
    # Phase 3: Sleep replay simulation
    print("Phase 3: Sleep Replay Simulation")
    print("-" * 70)
    
    print("Simulating sleep-like replay...")
    simulate_sleep_replay(memory_manager, neocortex, n_replay_cycles=5)
    
    print("✓ Replay complete")
    mem_stats = memory_manager.get_stats()
    print(f"  Replayed samples: {mem_stats['total_accesses']}")
    print(f"  Consolidations: {mem_stats['consolidation_count']}")
    print()
    
    # Phase 4: Recall and prediction
    print("Phase 4: Recall and Prediction")
    print("-" * 70)
    
    recall_texts = [
        "neural networks",
        "memory systems",
        "intelligent agents",
    ]
    
    for text in recall_texts:
        tokens = [hash(word) % 10000 for word in text.split()]
        result = cycle.step(tokens)
        
        toki = result['toki_output']
        english = translate_to_english(toki)
        
        print(f"Input:    '{text}'")
        print(f"Output:   {toki}")
        print(f"English:  {english}")
        print()
    
    # Final statistics
    print("=" * 70)
    print("Final Statistics")
    print("=" * 70)
    
    cycle_stats = cycle.get_stats()
    print(f"Total cycles:       {cycle_stats['total_cycles']}")
    print(f"Average cycle time: {cycle_stats['avg_cycle_time_ms']:.2f}ms")
    print(f"Target cycle time:  {cycle_stats['target_cycle_ms']:.2f}ms")
    print()
    
    mem_stats = neocortex.get_memory_stats()
    print(f"Memory Statistics:")
    print(f"  Total memories:  {mem_stats['total_memories']}")
    print(f"  Clusters:        {mem_stats['n_clusters']}")
    print(f"  Utilization:     {mem_stats['utilization']:.1%}")
    print()
    
    bridge_stats = bridge.get_stats()
    print(f"Semantic Bridge:")
    print(f"  Total concepts:  {bridge_stats['total_concepts']}")
    print()
    
    print("=" * 70)
    print("Demo complete!")
    print()


def demonstrate_novelty_detection():
    """Demonstrate novelty detection and importance weighting."""
    
    print("=" * 70)
    print("Novelty Detection Demo")
    print("=" * 70)
    print()
    
    volatile = VolatileAgent(window_size=32, embedding_dim=128)
    neocortex = NeocortexAgent(embedding_dim=128)
    bridge = SemanticBridge(embedding_dim=128)
    
    cycle = InferenceCycle(volatile, neocortex, bridge)
    
    # Familiar concepts
    familiar = ["learning", "memory", "pattern"]
    
    print("Learning familiar concepts...")
    for text in familiar * 3:  # Repeat 3 times
        tokens = [hash(word) % 10000 for word in text.split()]
        result = cycle.step(tokens)
        print(f"  Familiar: '{text}' - {result['toki_output']}")
    
    print()
    
    # Novel concepts
    novel = ["quantum entanglement", "cryptographic primitives", "topological spaces"]
    
    print("Introducing novel concepts...")
    for text in novel:
        tokens = [hash(word) % 10000 for word in text.split()]
        result = cycle.step(tokens)
        print(f"  Novel:    '{text}' - {result['toki_output']}")
    
    print()
    print("Bridge contains", bridge.get_stats()['total_concepts'], "concepts")
    print("=" * 70)
    print()


if __name__ == "__main__":
    # Run main demo
    demonstrate_consolidation()
    
    # Run novelty demo
    print("\n\n")
    demonstrate_novelty_detection()
