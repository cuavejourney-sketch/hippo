"""
Tests for compression utilities.
"""

import pytest
import numpy as np
import torch

from hippo.compression import (
    compress_embeddings,
    compress_vector,
    cluster_vectors,
    calculate_novelty,
    merge_similar_concepts
)


def test_compress_embeddings_mean():
    """Test mean pooling compression."""
    embeddings = torch.randn(10, 128)
    
    summary = compress_embeddings(embeddings, method="mean")
    
    assert isinstance(summary, np.ndarray)
    assert summary.shape == (128,)


def test_compress_embeddings_max():
    """Test max pooling compression."""
    embeddings = torch.randn(10, 128)
    
    summary = compress_embeddings(embeddings, method="max")
    
    assert isinstance(summary, np.ndarray)
    assert summary.shape == (128,)


def test_compress_embeddings_weighted():
    """Test weighted compression."""
    embeddings = torch.randn(10, 128)
    
    summary = compress_embeddings(embeddings, method="weighted")
    
    assert isinstance(summary, np.ndarray)
    assert summary.shape == (128,)


def test_compress_vector_identity():
    """Test identity compression."""
    vector = np.random.randn(128)
    
    compressed = compress_vector(vector, target_dim=64, method="identity")
    
    assert compressed.shape == (64,)


def test_compress_vector_random():
    """Test random projection compression."""
    vector = np.random.randn(128)
    
    compressed = compress_vector(vector, target_dim=64, method="random")
    
    assert compressed.shape == (64,)


def test_cluster_vectors_kmeans():
    """Test k-means clustering."""
    vectors = np.random.randn(50, 128)
    
    centers, labels = cluster_vectors(vectors, n_clusters=5, method="kmeans")
    
    assert centers.shape == (5, 128)
    assert labels.shape == (50,)


def test_cluster_vectors_hierarchical():
    """Test hierarchical clustering."""
    vectors = np.random.randn(50, 128)
    
    centers, labels = cluster_vectors(vectors, n_clusters=5, method="hierarchical")
    
    assert centers.shape == (5, 128)
    assert labels.shape == (50,)


def test_calculate_novelty():
    """Test novelty calculation."""
    vector = np.random.randn(128)
    memory = [np.random.randn(128) for _ in range(5)]
    
    novelty = calculate_novelty(vector, memory)
    
    assert 0.0 <= novelty <= 1.0


def test_calculate_novelty_empty():
    """Test novelty with empty memory."""
    vector = np.random.randn(128)
    
    novelty = calculate_novelty(vector, [])
    
    assert novelty == 1.0


def test_merge_similar_concepts():
    """Test merging similar concept vectors."""
    # Create some similar vectors
    base = np.random.randn(128)
    vectors = [base + np.random.randn(128) * 0.01 for _ in range(5)]
    
    merged = merge_similar_concepts(vectors, threshold=0.9)
    
    assert len(merged) <= len(vectors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
