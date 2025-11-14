"""
Semantic Bridge

Shared latent embedding space for cross-agent communication
and concept retrieval.
"""

from typing import List, Optional, Tuple
import numpy as np
import faiss


class SemanticBridge:
    """
    Shared semantic bridge for concept embeddings and communication
    between volatile and neocortex agents.
    
    Uses FAISS for efficient similarity search.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        index_type: str = "flat"
    ):
        """
        Initialize the Semantic Bridge.
        
        Args:
            embedding_dim: Dimension of concept embeddings
            index_type: Type of FAISS index ('flat' or 'ivf')
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        
        # Initialize FAISS index
        if index_type == "flat":
            self.index = faiss.IndexFlatL2(embedding_dim)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatL2(embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, 100)
        else:
            raise ValueError(f"Unknown index_type: {index_type}")
        
        # Storage for concept vectors
        self.vectors: List[np.ndarray] = []
        self.concept_ids: List[str] = []
        self.importance_weights: List[float] = []
        
        # Training flag for IVF index
        self.trained = (index_type == "flat")
    
    def add(
        self,
        concept_vector: np.ndarray,
        concept_id: Optional[str] = None,
        importance: float = 1.0
    ) -> None:
        """
        Add a concept vector to the bridge.
        
        Args:
            concept_vector: Concept embedding to add
            concept_id: Optional identifier for the concept
            importance: Importance weight (higher = more important)
        """
        # Ensure correct shape
        if concept_vector.ndim == 1:
            concept_vector = concept_vector.reshape(1, -1)
        
        # Normalize vector
        norm = np.linalg.norm(concept_vector)
        if norm > 0:
            concept_vector = concept_vector / norm
        
        # Train IVF index if needed
        if not self.trained and len(self.vectors) >= 100:
            self._train_index()
        
        # Add to index
        if self.trained:
            self.index.add(concept_vector.astype(np.float32))
        
        # Store vector and metadata
        self.vectors.append(concept_vector.squeeze())
        self.concept_ids.append(concept_id or f"concept_{len(self.vectors)}")
        self.importance_weights.append(importance)
    
    def find_similar(
        self,
        query_vector: np.ndarray,
        k: int = 1
    ) -> Optional[np.ndarray]:
        """
        Find the most similar concept vector.
        
        Args:
            query_vector: Query embedding
            k: Number of nearest neighbors to return
        
        Returns:
            Most similar concept vector or None if bridge is empty
        """
        if len(self.vectors) == 0:
            return None
        
        # Ensure correct shape
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Normalize query
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
        
        if not self.trained:
            # Fallback to brute force search
            similarities = []
            for vec in self.vectors:
                sim = np.dot(query_vector.squeeze(), vec)
                similarities.append(sim)
            
            best_idx = np.argmax(similarities)
            return self.vectors[best_idx]
        
        # Search using FAISS
        distances, indices = self.index.search(
            query_vector.astype(np.float32), k
        )
        
        best_idx = indices[0, 0]
        return self.vectors[best_idx]
    
    def find_similar_with_ids(
        self,
        query_vector: np.ndarray,
        k: int = 5
    ) -> List[Tuple[str, np.ndarray, float]]:
        """
        Find similar concepts with their IDs and similarity scores.
        
        Args:
            query_vector: Query embedding
            k: Number of nearest neighbors to return
        
        Returns:
            List of (concept_id, vector, similarity) tuples
        """
        if len(self.vectors) == 0:
            return []
        
        # Ensure correct shape
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Normalize query
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
        
        results = []
        
        if not self.trained:
            # Brute force search
            for i, vec in enumerate(self.vectors):
                sim = np.dot(query_vector.squeeze(), vec)
                results.append((self.concept_ids[i], vec, float(sim)))
            
            results.sort(key=lambda x: x[2], reverse=True)
            return results[:k]
        
        # FAISS search
        distances, indices = self.index.search(
            query_vector.astype(np.float32), min(k, len(self.vectors))
        )
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.vectors):
                similarity = 1.0 / (1.0 + dist)  # Convert distance to similarity
                results.append((
                    self.concept_ids[idx],
                    self.vectors[idx],
                    similarity
                ))
        
        return results
    
    def poll(self, concept_query: str) -> Optional[np.ndarray]:
        """
        Poll for a concept by ID.
        
        Args:
            concept_query: Concept ID to retrieve
        
        Returns:
            Concept vector or None if not found
        """
        try:
            idx = self.concept_ids.index(concept_query)
            return self.vectors[idx]
        except ValueError:
            return None
    
    def _train_index(self) -> None:
        """Train the FAISS index (for IVF indexes)."""
        if self.index_type == "ivf" and not self.trained:
            vectors_matrix = np.stack(self.vectors).astype(np.float32)
            self.index.train(vectors_matrix)
            self.trained = True
    
    def get_stats(self) -> dict:
        """Get statistics about the semantic bridge."""
        return {
            "total_concepts": len(self.vectors),
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "trained": self.trained
        }
    
    def clear(self) -> None:
        """Clear all concepts from the bridge."""
        self.index.reset()
        self.vectors = []
        self.concept_ids = []
        self.importance_weights = []
        self.trained = (self.index_type == "flat")
