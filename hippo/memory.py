"""
Memory management utilities for consolidation and replay.
"""

from typing import List, Optional, Tuple
import numpy as np
from collections import defaultdict


class MemoryManager:
    """
    Manages memory consolidation, replay, and forgetting strategies.
    """
    
    def __init__(
        self,
        replay_buffer_size: int = 100,
        forgetting_strategy: str = "age"
    ):
        """
        Initialize memory manager.
        
        Args:
            replay_buffer_size: Size of replay buffer
            forgetting_strategy: Strategy for forgetting ('age', 'frequency', 'importance')
        """
        self.replay_buffer_size = replay_buffer_size
        self.forgetting_strategy = forgetting_strategy
        
        self.replay_buffer: List[np.ndarray] = []
        self.access_counts = defaultdict(int)
        self.ages = defaultdict(int)
        self.importance_scores = defaultdict(float)
        
        self.consolidation_count = 0
    
    def add_to_replay_buffer(
        self,
        vector: np.ndarray,
        importance: float = 1.0
    ) -> None:
        """
        Add vector to replay buffer.
        
        Args:
            vector: Memory vector to add
            importance: Importance score
        """
        vector_id = len(self.replay_buffer)
        
        self.replay_buffer.append(vector)
        self.importance_scores[vector_id] = importance
        self.ages[vector_id] = 0
        
        # Update ages
        for vid in self.ages:
            self.ages[vid] += 1
        
        # Evict if buffer full
        if len(self.replay_buffer) > self.replay_buffer_size:
            self._evict_memory()
    
    def sample_for_replay(self, n_samples: int = 10) -> List[np.ndarray]:
        """
        Sample memories for replay.
        
        Args:
            n_samples: Number of samples to return
        
        Returns:
            List of sampled memory vectors
        """
        if len(self.replay_buffer) == 0:
            return []
        
        # Sample based on importance
        importance_values = [
            self.importance_scores[i] for i in range(len(self.replay_buffer))
        ]
        
        # Normalize to probabilities
        total = sum(importance_values)
        if total > 0:
            probs = [imp / total for imp in importance_values]
        else:
            probs = [1.0 / len(self.replay_buffer)] * len(self.replay_buffer)
        
        # Sample
        n_samples = min(n_samples, len(self.replay_buffer))
        indices = np.random.choice(
            len(self.replay_buffer),
            size=n_samples,
            replace=False,
            p=probs
        )
        
        # Update access counts
        for idx in indices:
            self.access_counts[idx] += 1
        
        return [self.replay_buffer[i] for i in indices]
    
    def consolidate(self, memories: List[np.ndarray]) -> List[np.ndarray]:
        """
        Perform memory consolidation.
        
        Args:
            memories: List of memory vectors to consolidate
        
        Returns:
            Consolidated memory vectors
        """
        if not memories:
            return []
        
        from hippo.compression import merge_similar_concepts
        
        # Merge similar memories
        consolidated = merge_similar_concepts(memories, threshold=0.85)
        
        self.consolidation_count += 1
        
        return consolidated
    
    def _evict_memory(self) -> None:
        """Evict memory based on forgetting strategy."""
        if len(self.replay_buffer) == 0:
            return
        
        if self.forgetting_strategy == "age":
            # Evict oldest
            oldest_idx = max(self.ages.items(), key=lambda x: x[1])[0]
            evict_idx = oldest_idx
        
        elif self.forgetting_strategy == "frequency":
            # Evict least accessed
            least_accessed_idx = min(
                self.access_counts.items(),
                key=lambda x: x[1],
                default=(0, 0)
            )[0]
            evict_idx = least_accessed_idx
        
        elif self.forgetting_strategy == "importance":
            # Evict least important
            least_important_idx = min(
                self.importance_scores.items(),
                key=lambda x: x[1]
            )[0]
            evict_idx = least_important_idx
        
        else:
            # Default: FIFO
            evict_idx = 0
        
        # Remove memory
        if evict_idx < len(self.replay_buffer):
            del self.replay_buffer[evict_idx]
            del self.ages[evict_idx]
            if evict_idx in self.access_counts:
                del self.access_counts[evict_idx]
            if evict_idx in self.importance_scores:
                del self.importance_scores[evict_idx]
    
    def get_stats(self) -> dict:
        """Get memory statistics."""
        return {
            "buffer_size": len(self.replay_buffer),
            "buffer_capacity": self.replay_buffer_size,
            "consolidation_count": self.consolidation_count,
            "total_accesses": sum(self.access_counts.values()),
            "avg_importance": np.mean(list(self.importance_scores.values())) if self.importance_scores else 0
        }


def simulate_sleep_replay(
    memory_manager: MemoryManager,
    neocortex,
    n_replay_cycles: int = 10
) -> None:
    """
    Simulate sleep-like replay for memory consolidation.
    
    Args:
        memory_manager: MemoryManager instance
        neocortex: Neocortex agent
        n_replay_cycles: Number of replay cycles
    """
    for _ in range(n_replay_cycles):
        # Sample memories for replay
        replay_samples = memory_manager.sample_for_replay(n_samples=5)
        
        # Send to neocortex for consolidation
        for memory in replay_samples:
            neocortex.receive(memory)
        
        # Consolidate
        neocortex.consolidate()
