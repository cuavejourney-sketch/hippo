"""
Tests for the NeocortexAgent.
"""

import pytest
import numpy as np

from hippo.neocortex_agent import NeocortexAgent


def test_neocortex_agent_init():
    """Test NeocortexAgent initialization."""
    agent = NeocortexAgent(embedding_dim=128, max_memory_size=100)
    
    assert agent.embedding_dim == 128
    assert agent.max_memory_size == 100
    assert len(agent.latent_memory) == 0


def test_neocortex_agent_receive():
    """Test receiving replay vectors."""
    agent = NeocortexAgent(embedding_dim=128)
    
    replay_vector = np.random.randn(128)
    
    compressed = agent.receive(replay_vector)
    
    assert isinstance(compressed, np.ndarray)
    assert len(agent.latent_memory) == 1


def test_neocortex_agent_generalize():
    """Test generalizing patterns."""
    agent = NeocortexAgent(embedding_dim=128, consolidation_threshold=5)
    
    # Add multiple vectors
    for _ in range(10):
        vector = np.random.randn(128)
        agent.receive(vector)
    
    clusters = agent.generalize()
    
    assert clusters is not None
    assert len(clusters) > 0


def test_neocortex_agent_predict():
    """Test producing predictions."""
    agent = NeocortexAgent(embedding_dim=128, consolidation_threshold=5)
    
    # Add vectors and generalize
    for _ in range(10):
        vector = np.random.randn(128)
        agent.receive(vector)
    
    clusters = agent.generalize()
    predictions = agent.predict(clusters)
    
    assert len(predictions) > 0
    assert all(isinstance(p, np.ndarray) for p in predictions)


def test_neocortex_agent_speak_toki_pona():
    """Test toki pona output."""
    agent = NeocortexAgent(embedding_dim=128)
    
    predictions = [np.random.randn(128) for _ in range(3)]
    
    sentence = agent.speak_toki_pona(predictions)
    
    assert isinstance(sentence, str)
    assert len(sentence) > 0


def test_neocortex_agent_consolidate():
    """Test memory consolidation."""
    agent = NeocortexAgent(embedding_dim=128, consolidation_threshold=5)
    
    # Add vectors
    for _ in range(10):
        vector = np.random.randn(128)
        agent.receive(vector)
    
    initial_count = len(agent.latent_memory)
    
    agent.consolidate()
    
    # Consolidation may change memory count
    assert agent.memory_clusters is not None


def test_neocortex_agent_memory_limit():
    """Test that memory respects max size."""
    agent = NeocortexAgent(embedding_dim=128, max_memory_size=10)
    
    # Add more than max
    for _ in range(20):
        vector = np.random.randn(128)
        agent.receive(vector)
    
    assert len(agent.latent_memory) <= 10


def test_neocortex_agent_stats():
    """Test getting memory statistics."""
    agent = NeocortexAgent(embedding_dim=128)
    
    stats = agent.get_memory_stats()
    
    assert "total_memories" in stats
    assert "n_clusters" in stats
    assert "utilization" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
