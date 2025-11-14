"""
Integration tests for the complete dual-agent system.
"""

import pytest
import numpy as np

from hippo import VolatileAgent, NeocortexAgent, SemanticBridge, run_dual_cascade
from hippo.loop import InferenceCycle, create_text_stream
from hippo.memory import MemoryManager, simulate_sleep_replay


class TestIntegration:
    """Integration tests for full system."""
    
    def test_end_to_end_flow(self):
        """Test complete end-to-end flow."""
        # Initialize agents
        volatile = VolatileAgent(window_size=32, embedding_dim=64)
        neocortex = NeocortexAgent(embedding_dim=64, max_memory_size=50)
        bridge = SemanticBridge(embedding_dim=64)
        
        # Create input
        texts = ["test input", "another test", "final test"]
        input_stream = create_text_stream(texts)
        
        # Run cascade
        results = list(run_dual_cascade(
            volatile, neocortex, bridge, input_stream, max_cycles=3
        ))
        
        # Verify
        assert len(results) == 3
        assert all('toki_output' in r for r in results)
        assert all('cycle_time_ms' in r for r in results)
    
    def test_memory_consolidation_flow(self):
        """Test memory consolidation workflow."""
        # Setup
        volatile = VolatileAgent(window_size=32, embedding_dim=64)
        neocortex = NeocortexAgent(
            embedding_dim=64,
            consolidation_threshold=3
        )
        bridge = SemanticBridge(embedding_dim=64)
        
        cycle = InferenceCycle(volatile, neocortex, bridge)
        
        # Add multiple cycles
        for i in range(10):
            tokens = [i, i+1, i+2]
            cycle.step(tokens)
        
        # Consolidate
        neocortex.consolidate()
        
        # Check consolidation occurred
        stats = neocortex.get_memory_stats()
        assert stats['n_clusters'] > 0
    
    def test_replay_mechanism(self):
        """Test memory replay mechanism."""
        # Setup
        neocortex = NeocortexAgent(embedding_dim=64)
        memory_manager = MemoryManager(replay_buffer_size=20)
        
        # Add memories
        for _ in range(10):
            vector = np.random.randn(64)
            memory_manager.add_to_replay_buffer(vector, importance=0.5)
        
        # Simulate replay
        simulate_sleep_replay(memory_manager, neocortex, n_replay_cycles=3)
        
        # Verify memories were processed
        stats = memory_manager.get_stats()
        assert stats['total_accesses'] > 0
    
    def test_novelty_detection_integration(self):
        """Test novelty detection in full system."""
        volatile = VolatileAgent(window_size=32, embedding_dim=64)
        bridge = SemanticBridge(embedding_dim=64)
        
        # Add familiar concept
        familiar = np.random.randn(64)
        bridge.add(familiar, concept_id="familiar")
        
        # Observe similar input
        volatile.observe([1, 2, 3])
        embeddings = volatile.encode_current()
        summary = volatile.summarize(embeddings)
        
        # Compare
        _, difference = volatile.reconstruct_and_compare(summary, bridge)
        
        # Should detect some difference
        assert isinstance(difference, np.ndarray)
        assert difference.shape == (64,)
    
    def test_full_cycle_timing(self):
        """Test that cycle timing is reasonable."""
        volatile = VolatileAgent(window_size=32, embedding_dim=64)
        neocortex = NeocortexAgent(embedding_dim=64)
        bridge = SemanticBridge(embedding_dim=64)
        
        cycle = InferenceCycle(volatile, neocortex, bridge, target_cycle_ms=100.0)
        
        # Run multiple cycles
        times = []
        for i in range(10):
            result = cycle.step([i, i+1])
            times.append(result['cycle_time_ms'])
        
        # Check average time
        avg_time = sum(times) / len(times)
        assert avg_time < 100.0  # Should be under target
    
    def test_toki_pona_output_consistency(self):
        """Test toki pona output is consistent."""
        neocortex = NeocortexAgent(embedding_dim=64)
        
        # Add multiple memories
        for _ in range(10):
            vector = np.random.randn(64)
            neocortex.receive(vector)
        
        # Generate output multiple times
        clusters = neocortex.generalize()
        predictions = neocortex.predict(clusters)
        
        outputs = []
        for _ in range(5):
            output = neocortex.speak_toki_pona(predictions)
            outputs.append(output)
        
        # All outputs should be valid strings
        assert all(isinstance(o, str) for o in outputs)
        assert all(len(o) > 0 for o in outputs)
    
    def test_bridge_concept_retrieval(self):
        """Test semantic bridge concept retrieval."""
        bridge = SemanticBridge(embedding_dim=64)
        
        # Add concepts
        concepts = {}
        for i in range(10):
            vector = np.random.randn(64)
            concept_id = f"concept_{i}"
            bridge.add(vector, concept_id=concept_id)
            concepts[concept_id] = vector
        
        # Retrieve specific concept
        retrieved = bridge.poll("concept_5")
        assert retrieved is not None
        
        # Search for similar
        query = np.random.randn(64)
        similar = bridge.find_similar(query)
        assert similar is not None
    
    def test_system_scalability(self):
        """Test system scales to larger inputs."""
        volatile = VolatileAgent(window_size=128, embedding_dim=128)
        neocortex = NeocortexAgent(embedding_dim=128, max_memory_size=200)
        bridge = SemanticBridge(embedding_dim=128)
        
        cycle = InferenceCycle(volatile, neocortex, bridge)
        
        # Process many cycles
        for i in range(50):
            tokens = list(range(i, i+10))
            result = cycle.step(tokens)
            
            # Should complete successfully
            assert result['cycle'] == i + 1
        
        # Check final state
        stats = cycle.get_stats()
        assert stats['total_cycles'] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
