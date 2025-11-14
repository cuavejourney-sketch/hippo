"""
Compression utilities for embedding vectors and concept abstraction.
"""

from typing import Optional
import numpy as np
import torch
from sklearn.decomposition import PCA


def compress_embeddings(
    embeddings: torch.Tensor,
    method: str = "mean"
) -> np.ndarray:
    """
    Compress a sequence of embeddings into a single summary vector.
    
    Args:
        embeddings: Tensor of shape (sequence_length, embedding_dim)
        method: Compression method ('mean', 'max', 'weighted')
    
    Returns:
        Compressed vector of shape (embedding_dim,)
    """
    if embeddings.numel() == 0:
        return np.zeros(embeddings.shape[-1] if embeddings.ndim > 1 else 1)
    
    embeddings_np = embeddings.detach().cpu().numpy()
    
    if method == "mean":
        # Mean pooling
        summary = np.mean(embeddings_np, axis=0)
    elif method == "max":
        # Max pooling
        summary = np.max(embeddings_np, axis=0)
    elif method == "weighted":
        # Weighted average (recent tokens get higher weight)
        weights = np.linspace(0.5, 1.0, len(embeddings_np))
        weights = weights / weights.sum()
        summary = np.average(embeddings_np, axis=0, weights=weights)
    else:
        raise ValueError(f"Unknown compression method: {method}")
    
    return summary


def compress_vector(
    vector: np.ndarray,
    target_dim: Optional[int] = None,
    method: str = "identity"
) -> np.ndarray:
    """
    Compress a vector to target dimension.
    
    Args:
        vector: Input vector
        target_dim: Target dimension (None means keep same)
        method: Compression method ('identity', 'pca', 'random')
    
    Returns:
        Compressed vector
    """
    if target_dim is None or target_dim == len(vector):
        return vector
    
    if method == "identity":
        # Just truncate or pad
        if len(vector) > target_dim:
            return vector[:target_dim]
        else:
            return np.pad(vector, (0, target_dim - len(vector)))
    elif method == "pca":
        # PCA compression (needs multiple vectors, fallback to identity)
        return compress_vector(vector, target_dim, method="identity")
    elif method == "random":
        # Random projection
        proj_matrix = np.random.randn(len(vector), target_dim)
        proj_matrix = proj_matrix / np.linalg.norm(proj_matrix, axis=0)
        return np.dot(vector, proj_matrix)
    else:
        raise ValueError(f"Unknown compression method: {method}")


def cluster_vectors(
    vectors: np.ndarray,
    n_clusters: int,
    method: str = "kmeans"
) -> tuple:
    """
    Cluster a set of vectors.
    
    Args:
        vectors: Array of shape (n_vectors, embedding_dim)
        n_clusters: Number of clusters
        method: Clustering method ('kmeans', 'hierarchical')
    
    Returns:
        Tuple of (cluster_centers, labels)
    """
    if method == "kmeans":
        from sklearn.cluster import MiniBatchKMeans
        clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        labels = clusterer.fit_predict(vectors)
        return clusterer.cluster_centers_, labels
    elif method == "hierarchical":
        from sklearn.cluster import AgglomerativeClustering
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clusterer.fit_predict(vectors)
        
        # Calculate cluster centers
        centers = []
        for i in range(n_clusters):
            cluster_vecs = vectors[labels == i]
            if len(cluster_vecs) > 0:
                centers.append(np.mean(cluster_vecs, axis=0))
            else:
                centers.append(np.zeros(vectors.shape[1]))
        
        return np.array(centers), labels
    else:
        raise ValueError(f"Unknown clustering method: {method}")


def calculate_novelty(
    vector: np.ndarray,
    memory_vectors: list
) -> float:
    """
    Calculate novelty score of a vector compared to memory.
    
    Args:
        vector: Query vector
        memory_vectors: List of memory vectors
    
    Returns:
        Novelty score (0-1, higher = more novel)
    """
    if not memory_vectors:
        return 1.0
    
    # Calculate minimum distance to existing memories
    distances = []
    for mem_vec in memory_vectors:
        dist = np.linalg.norm(vector - mem_vec)
        distances.append(dist)
    
    min_distance = min(distances)
    
    # Normalize to 0-1 range (using sigmoid)
    novelty = 1.0 / (1.0 + np.exp(-min_distance + 1))
    
    return novelty


def merge_similar_concepts(
    vectors: list,
    threshold: float = 0.9
) -> list:
    """
    Merge concept vectors that are very similar.
    
    Args:
        vectors: List of concept vectors
        threshold: Similarity threshold for merging (0-1)
    
    Returns:
        List of merged concept vectors
    """
    if len(vectors) <= 1:
        return vectors
    
    merged = []
    used = set()
    
    for i, vec_i in enumerate(vectors):
        if i in used:
            continue
        
        # Find similar vectors
        similar_group = [vec_i]
        for j, vec_j in enumerate(vectors[i+1:], start=i+1):
            if j in used:
                continue
            
            # Calculate cosine similarity
            sim = np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j))
            
            if sim >= threshold:
                similar_group.append(vec_j)
                used.add(j)
        
        # Merge similar vectors by averaging
        merged_vec = np.mean(similar_group, axis=0)
        merged.append(merged_vec)
        used.add(i)
    
    return merged
