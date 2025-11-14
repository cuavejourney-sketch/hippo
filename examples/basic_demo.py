"""
Example demonstrating the dual-agent cascade system.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hippo import VolatileAgent, NeocortexAgent, SemanticBridge, run_dual_cascade
from hippo.loop import create_text_stream
from hippo.toki_pona import translate_to_english


def main():
    """Run a simple demonstration of the dual-agent system."""
    
    print("=" * 70)
    print("Hippo: Dual-AI Cascade Memory System Demo")
    print("Inspired by hippocampus-neocortex architecture")
    print("=" * 70)
    print()
    
    # Initialize agents
    print("Initializing agents...")
    volatile = VolatileAgent(
        window_size=64,
        embedding_dim=128,
        device="cpu"
    )
    
    neocortex = NeocortexAgent(
        embedding_dim=128,
        max_memory_size=100,
        n_clusters=10
    )
    
    bridge = SemanticBridge(embedding_dim=128)
    
    print("✓ Volatile Agent (Hippocampal) ready")
    print("✓ Neocortex Agent ready")
    print("✓ Semantic Bridge ready")
    print()
    
    # Create sample input stream
    sample_texts = [
        "learning new patterns",
        "recognizing familiar concepts",
        "discovering novel information",
        "consolidating memories",
        "predicting future events",
        "understanding relationships",
        "abstracting core ideas",
        "integrating knowledge",
        "forming connections",
        "building mental models",
        "encoding experiences",
        "retrieving memories",
        "generalizing patterns",
        "adapting to change",
        "maintaining stability",
    ]
    
    print("Running dual-agent cascade...")
    print("-" * 70)
    print()
    
    # Run the cascade
    input_stream = create_text_stream(sample_texts)
    
    results = []
    for result in run_dual_cascade(volatile, neocortex, bridge, input_stream, verbose=False):
        cycle_num = result['cycle']
        toki = result['toki_output']
        time_ms = result['cycle_time_ms']
        
        # Translate toki pona to English
        english = translate_to_english(toki)
        
        print(f"Cycle {cycle_num:2d}:")
        print(f"  Toki Pona: {toki}")
        print(f"  English:   {english}")
        print(f"  Time:      {time_ms:.2f}ms")
        print()
        
        results.append(result)
    
    # Print summary statistics
    print("-" * 70)
    print("Summary Statistics:")
    print("-" * 70)
    
    if results:
        final_result = results[-1]
        
        avg_time = sum(r['cycle_time_ms'] for r in results) / len(results)
        print(f"Total cycles:       {len(results)}")
        print(f"Average cycle time: {avg_time:.2f}ms")
        print(f"Target cycle time:  60.00ms")
        print()
        
        mem_stats = final_result['memory_stats']
        print("Memory Statistics:")
        print(f"  Total memories:  {mem_stats['total_memories']}")
        print(f"  Clusters formed: {mem_stats['n_clusters']}")
        print(f"  Utilization:     {mem_stats['utilization']:.1%}")
        print()
        
        bridge_stats = final_result['bridge_stats']
        print("Semantic Bridge:")
        print(f"  Total concepts:  {bridge_stats['total_concepts']}")
        print(f"  Embedding dim:   {bridge_stats['embedding_dim']}")
        print()
    
    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
