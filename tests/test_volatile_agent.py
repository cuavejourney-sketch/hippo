"""
Tests for the VolatileAgent.
"""

import pytest
import numpy as np
import torch

from hippo.volatile_agent import VolatileAgent


def test_volatile_agent_init():
    """Test VolatileAgent initialization."""
    agent = VolatileAgent(window_size=64, embedding_dim=128)
    
    assert agent.window_size == 64
    assert agent.embedding_dim == 128
    assert len(agent.context_buffer) == 0


def test_volatile_agent_observe():
    """Test observing input tokens."""
    agent = VolatileAgent(window_size=10, embedding_dim=128)
    
    tokens = [1, 2, 3, 4, 5]
    agent.observe(tokens)
    
    assert agent.get_context_size() == 5
    assert list(agent.context_buffer) == tokens


def test_volatile_agent_window_overflow():
    """Test that context buffer respects window size."""
    agent = VolatileAgent(window_size=5, embedding_dim=128)
    
    # Add more tokens than window size
    tokens = list(range(10))
    agent.observe(tokens)
    
    # Should only keep last 5
    assert agent.get_context_size() == 5
    assert list(agent.context_buffer) == [5, 6, 7, 8, 9]


def test_volatile_agent_encode():
    """Test encoding current context."""
    agent = VolatileAgent(window_size=10, embedding_dim=128)
    
    tokens = [1, 2, 3]
    agent.observe(tokens)
    
    embeddings = agent.encode_current()
    
    assert embeddings.shape[-1] == 128
    assert embeddings.dim() == 2


def test_volatile_agent_summarize():
    """Test summarizing embeddings."""
    agent = VolatileAgent(window_size=10, embedding_dim=128)
    
    # Create dummy embeddings
    embeddings = torch.randn(5, 128)
    
    summary = agent.summarize(embeddings)
    
    assert isinstance(summary, np.ndarray)
    assert summary.shape == (128,)


def test_volatile_agent_produce_replay():
    """Test producing replay vectors."""
    agent = VolatileAgent(window_size=10, embedding_dim=128)
    
    summary = np.random.randn(128)
    difference = np.random.randn(128)
    
    replay = agent.produce_replay(summary, difference, novelty_weight=0.5)
    
    assert isinstance(replay, np.ndarray)
    assert replay.shape == (128,)
    
    # Check normalization
    norm = np.linalg.norm(replay)
    assert abs(norm - 1.0) < 0.01  # Should be normalized


def test_volatile_agent_reset():
    """Test resetting context."""
    agent = VolatileAgent(window_size=10, embedding_dim=128)
    
    agent.observe([1, 2, 3])
    assert agent.get_context_size() == 3
    
    agent.reset_context()
    assert agent.get_context_size() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
