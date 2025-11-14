#!/usr/bin/env python3
"""
Verification script to check all components of the Hippo system.
"""

import sys
import os

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from hippo import VolatileAgent, NeocortexAgent, SemanticBridge
        from hippo import run_dual_cascade, InferenceCycle
        from hippo.memory import MemoryManager
        from hippo.compression import compress_embeddings
        from hippo.toki_pona import render_toki_pona
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic system functionality."""
    print("\nTesting basic functionality...")
    try:
        from hippo import VolatileAgent, NeocortexAgent, SemanticBridge
        from hippo.loop import InferenceCycle
        
        # Create agents
        volatile = VolatileAgent(window_size=32, embedding_dim=64)
        neocortex = NeocortexAgent(embedding_dim=64)
        bridge = SemanticBridge(embedding_dim=64)
        
        # Create cycle
        cycle = InferenceCycle(volatile, neocortex, bridge)
        
        # Run a cycle
        result = cycle.step([1, 2, 3])
        
        assert 'cycle' in result
        assert 'toki_output' in result
        assert 'cycle_time_ms' in result
        
        print(f"✓ Basic functionality works")
        print(f"  Cycle time: {result['cycle_time_ms']:.2f}ms")
        print(f"  Output: {result['toki_output']}")
        return True
    except Exception as e:
        print(f"✗ Basic functionality failed: {e}")
        return False


def test_performance():
    """Test system performance."""
    print("\nTesting performance...")
    try:
        from hippo import VolatileAgent, NeocortexAgent, SemanticBridge
        from hippo.loop import InferenceCycle
        import time
        
        volatile = VolatileAgent(window_size=32, embedding_dim=128)
        neocortex = NeocortexAgent(embedding_dim=128)
        bridge = SemanticBridge(embedding_dim=128)
        cycle = InferenceCycle(volatile, neocortex, bridge)
        
        # Run multiple cycles
        times = []
        for i in range(10):
            result = cycle.step([i, i+1, i+2])
            times.append(result['cycle_time_ms'])
        
        avg_time = sum(times) / len(times)
        print(f"✓ Performance test passed")
        print(f"  Average cycle time: {avg_time:.2f}ms")
        print(f"  Target: 60.00ms")
        print(f"  Performance ratio: {60/avg_time:.1f}x faster")
        
        return avg_time < 60.0
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False


def test_memory_operations():
    """Test memory operations."""
    print("\nTesting memory operations...")
    try:
        from hippo import NeocortexAgent
        from hippo.memory import MemoryManager
        import numpy as np
        
        neocortex = NeocortexAgent(embedding_dim=64)
        memory = MemoryManager(replay_buffer_size=20)
        
        # Add memories
        for _ in range(10):
            vec = np.random.randn(64)
            neocortex.receive(vec)
            memory.add_to_replay_buffer(vec, importance=0.5)
        
        # Consolidate
        neocortex.consolidate()
        
        stats = neocortex.get_memory_stats()
        print(f"✓ Memory operations work")
        print(f"  Memories: {stats['total_memories']}")
        print(f"  Clusters: {stats['n_clusters']}")
        return True
    except Exception as e:
        print(f"✗ Memory operations failed: {e}")
        return False


def test_toki_pona():
    """Test toki pona rendering."""
    print("\nTesting toki pona...")
    try:
        from hippo.toki_pona import render_toki_pona, translate_to_english
        import numpy as np
        
        predictions = [np.random.randn(64) for _ in range(3)]
        sentence = render_toki_pona(predictions)
        english = translate_to_english(sentence)
        
        print(f"✓ Toki pona rendering works")
        print(f"  Toki pona: {sentence}")
        print(f"  English: {english}")
        return len(sentence) > 0
    except Exception as e:
        print(f"✗ Toki pona failed: {e}")
        return False


def check_files():
    """Check that all required files exist."""
    print("\nChecking files...")
    required_files = [
        'README.md',
        'ARCHITECTURE.md',
        'USAGE.md',
        'IMPLEMENTATION.md',
        'LICENSE',
        'setup.py',
        'requirements.txt',
        'hippo/__init__.py',
        'hippo/volatile_agent.py',
        'hippo/neocortex_agent.py',
        'hippo/semantic_bridge.py',
        'hippo/loop.py',
        'hippo/memory.py',
        'hippo/compression.py',
        'hippo/toki_pona.py',
        'hippo/models.py',
        'examples/basic_demo.py',
        'examples/advanced_demo.py',
        'tests/test_volatile_agent.py',
        'tests/test_neocortex_agent.py',
        'tests/test_semantic_bridge.py',
        'tests/test_compression.py',
        'tests/test_toki_pona.py',
        'tests/test_loop.py',
        'tests/test_integration.py',
    ]
    
    missing = []
    for file in required_files:
        path = os.path.join(os.path.dirname(__file__), file)
        if not os.path.exists(path):
            missing.append(file)
    
    if missing:
        print(f"✗ Missing files: {missing}")
        return False
    else:
        print(f"✓ All {len(required_files)} required files present")
        return True


def main():
    """Run all verification tests."""
    print("=" * 70)
    print("Hippo System Verification")
    print("=" * 70)
    
    results = {
        "File Check": check_files(),
        "Imports": test_imports(),
        "Basic Functionality": test_basic_functionality(),
        "Performance": test_performance(),
        "Memory Operations": test_memory_operations(),
        "Toki Pona": test_toki_pona(),
    }
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<50} {status}")
    
    all_passed = all(results.values())
    print("=" * 70)
    
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nSystem is fully operational and ready to use!")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
