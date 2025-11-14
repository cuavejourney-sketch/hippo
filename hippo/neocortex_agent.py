"""
Neocortex Agent

Gradual abstraction and consolidation agent that uses toki pona
for simple conceptual representations.
"""

from typing import List, Optional, Dict, Any
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import torch


class NeocortexAgent:
    """
    Neocortex-inspired agent for abstraction and long-term memory consolidation.
    
    Receives replay vectors from volatile agent, compresses them into stable
    concepts, and produces predictions in toki pona.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        max_memory_size: int = 1000,
        n_clusters: int = 50,
        consolidation_threshold: int = 10
    ):
        """
        Initialize the Neocortex Agent.
        
        Args:
            embedding_dim: Dimension of concept embeddings
            max_memory_size: Maximum number of concept vectors to maintain
            n_clusters: Number of clusters for generalization
            consolidation_threshold: Min vectors before clustering
        """
        self.embedding_dim = embedding_dim
        self.max_memory_size = max_memory_size
        self.n_clusters = n_clusters
        self.consolidation_threshold = consolidation_threshold
        
        # Latent memory storage
        self.latent_memory: List[np.ndarray] = []
        self.memory_metadata: List[Dict[str, Any]] = []
        
        # Clustering model for generalization
        self.clusterer = MiniBatchKMeans(
            n_clusters=min(n_clusters, max_memory_size),
            random_state=42
        )
        
        # Memory clusters (abstract concepts)
        self.memory_clusters: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
    
    def receive(self, replay_vector: np.ndarray, metadata: Optional[Dict] = None) -> np.ndarray:
        """
        Receive and compress a replay vector from the volatile agent.
        
        Args:
            replay_vector: Vector from volatile agent
            metadata: Optional metadata about the experience
        
        Returns:
            Compressed concept vector
        """
        from hippo.compression import compress_vector
        
        # Compress the replay vector
        compressed = compress_vector(replay_vector, target_dim=self.embedding_dim)
        
        # Add to latent memory
        self.latent_memory.append(compressed)
        self.memory_metadata.append(metadata or {})
        
        # Manage memory size
        if len(self.latent_memory) > self.max_memory_size:
            self._evict_old_memories()
        
        return compressed
    
    def generalize(self) -> Optional[np.ndarray]:
        """
        Cluster and abstract common patterns in latent memory.
        
        Returns:
            Cluster centers (abstract concepts) or None if insufficient data
        """
        if len(self.latent_memory) < self.consolidation_threshold:
            return None
        
        # Stack memory vectors
        memory_matrix = np.stack(self.latent_memory)
        
        # Update clustering
        n_clusters = min(self.n_clusters, len(self.latent_memory))
        self.clusterer.n_clusters = n_clusters
        self.cluster_labels = self.clusterer.fit_predict(memory_matrix)
        self.memory_clusters = self.clusterer.cluster_centers_
        
        return self.memory_clusters
    
    def predict(self, memory_clusters: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Produce forward predictions from memory clusters.
        
        Args:
            memory_clusters: Cluster centers (uses self.memory_clusters if None)
        
        Returns:
            List of prediction vectors
        """
        if memory_clusters is None:
            memory_clusters = self.memory_clusters
        
        if memory_clusters is None or len(memory_clusters) == 0:
            return []
        
        # Generate predictions based on cluster patterns
        predictions = []
        for cluster_center in memory_clusters:
            # Simple prediction: project cluster forward with small perturbation
            prediction = cluster_center + np.random.normal(0, 0.01, cluster_center.shape)
            predictions.append(prediction)
        
        return predictions
    
    def speak_toki_pona(self, predictions: List[np.ndarray]) -> str:
        """
        Convert conceptual predictions to toki pona language.
        
        Args:
            predictions: List of prediction vectors
        
        Returns:
            Toki pona sentence describing the concepts
        """
        from hippo.toki_pona import render_toki_pona
        
        # Convert prediction vectors to toki pona
        sentence = render_toki_pona(predictions, self.memory_clusters)
        return sentence
    
    def consolidate(self) -> None:
        """
        Perform memory consolidation: cluster, prune, and stabilize.
        """
        if len(self.latent_memory) < self.consolidation_threshold:
            return
        
        # Generalize patterns
        self.generalize()
        
        # Prune redundant memories
        self._prune_redundant_memories()
    
    def _evict_old_memories(self) -> None:
        """Evict oldest memories when capacity is exceeded."""
        # Simple FIFO eviction
        excess = len(self.latent_memory) - self.max_memory_size
        if excess > 0:
            self.latent_memory = self.latent_memory[excess:]
            self.memory_metadata = self.memory_metadata[excess:]
    
    def _prune_redundant_memories(self) -> None:
        """
        Prune memories that are too similar to cluster centers.
        Keep only representative and novel memories.
        """
        if self.memory_clusters is None or len(self.latent_memory) == 0:
            return
        
        # Calculate distances to nearest cluster
        memory_matrix = np.stack(self.latent_memory)
        distances = []
        
        for memory in memory_matrix:
            # Find distance to nearest cluster center
            dists = np.linalg.norm(self.memory_clusters - memory, axis=1)
            min_dist = np.min(dists)
            distances.append(min_dist)
        
        # Keep memories that are either very close (representative) or far (novel)
        distances = np.array(distances)
        threshold_low = np.percentile(distances, 20)
        threshold_high = np.percentile(distances, 80)
        
        keep_indices = np.where(
            (distances <= threshold_low) | (distances >= threshold_high)
        )[0]
        
        # Keep selected memories
        self.latent_memory = [self.latent_memory[i] for i in keep_indices]
        self.memory_metadata = [self.memory_metadata[i] for i in keep_indices]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about current memory state."""
        return {
            "total_memories": len(self.latent_memory),
            "n_clusters": len(self.memory_clusters) if self.memory_clusters is not None else 0,
            "memory_capacity": self.max_memory_size,
            "utilization": len(self.latent_memory) / self.max_memory_size
        }
