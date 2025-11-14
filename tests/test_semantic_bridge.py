"""
Tests for the SemanticBridge.
"""

import pytest
import numpy as np

from hippo.semantic_bridge import SemanticBridge


def test_semantic_bridge_init():
    """Test SemanticBridge initialization."""
    bridge = SemanticBridge(embedding_dim=128)
    
    assert bridge.embedding_dim == 128
    assert len(bridge.vectors) == 0


def test_semantic_bridge_add():
    """Test adding concept vectors."""
    bridge = SemanticBridge(embedding_dim=128)
    
    vector = np.random.randn(128)
    bridge.add(vector, concept_id="test_concept")
    
    assert len(bridge.vectors) == 1
    assert "test_concept" in bridge.concept_ids


def test_semantic_bridge_find_similar():
    """Test finding similar concepts."""
    bridge = SemanticBridge(embedding_dim=128)
    
    # Add some vectors
    for i in range(5):
        vector = np.random.randn(128)
        bridge.add(vector, concept_id=f"concept_{i}")
    
    # Query
    query = np.random.randn(128)
    similar = bridge.find_similar(query)
    
    assert similar is not None
    assert similar.shape == (128,)


def test_semantic_bridge_find_similar_empty():
    """Test finding similar in empty bridge."""
    bridge = SemanticBridge(embedding_dim=128)
    
    query = np.random.randn(128)
    similar = bridge.find_similar(query)
    
    assert similar is None


def test_semantic_bridge_find_similar_with_ids():
    """Test finding similar concepts with IDs."""
    bridge = SemanticBridge(embedding_dim=128)
    
    # Add vectors
    for i in range(5):
        vector = np.random.randn(128)
        bridge.add(vector, concept_id=f"concept_{i}")
    
    query = np.random.randn(128)
    results = bridge.find_similar_with_ids(query, k=3)
    
    assert len(results) <= 3
    assert all(len(r) == 3 for r in results)  # (id, vector, score)


def test_semantic_bridge_poll():
    """Test polling for concept by ID."""
    bridge = SemanticBridge(embedding_dim=128)
    
    vector = np.random.randn(128)
    bridge.add(vector, concept_id="my_concept")
    
    retrieved = bridge.poll("my_concept")
    
    assert retrieved is not None
    np.testing.assert_array_almost_equal(retrieved, vector / np.linalg.norm(vector))


def test_semantic_bridge_poll_missing():
    """Test polling for non-existent concept."""
    bridge = SemanticBridge(embedding_dim=128)
    
    retrieved = bridge.poll("nonexistent")
    
    assert retrieved is None


def test_semantic_bridge_stats():
    """Test getting bridge statistics."""
    bridge = SemanticBridge(embedding_dim=128)
    
    stats = bridge.get_stats()
    
    assert "total_concepts" in stats
    assert "embedding_dim" in stats
    assert stats["embedding_dim"] == 128


def test_semantic_bridge_clear():
    """Test clearing the bridge."""
    bridge = SemanticBridge(embedding_dim=128)
    
    # Add vectors
    for i in range(5):
        vector = np.random.randn(128)
        bridge.add(vector)
    
    assert len(bridge.vectors) == 5
    
    bridge.clear()
    
    assert len(bridge.vectors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
