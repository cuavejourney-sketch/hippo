"""
Volatile Agent (Hippocampal-inspired)

Rapid encoding of "nowness", maintains working context buffer,
and generates summary vectors for the neocortex agent.
"""

from collections import deque
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn


class VolatileAgent:
    """
    Hippocampal-inspired agent for rapid encoding and short-term memory.
    
    Maintains a sliding window context buffer and generates compressed
    summaries of current experiences.
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        window_size: int = 128,
        embedding_dim: int = 768,
        device: str = "cpu"
    ):
        """
        Initialize the Volatile Agent.
        
        Args:
            model: Transformer model for encoding (optional, will use default if None)
            window_size: Size of the sliding context window
            embedding_dim: Dimension of concept embeddings
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Context buffer for recent observations
        self.context_buffer = deque(maxlen=window_size)
        
        # Initialize model
        if model is None:
            from hippo.models import get_default_encoder
            self.model = get_default_encoder(embedding_dim).to(device)
        else:
            self.model = model.to(device)
        
        self.model.eval()
    
    def observe(self, input_tokens: List[int]) -> None:
        """
        Add new input tokens to the context buffer.
        
        Args:
            input_tokens: List of token IDs to add to context
        """
        self.context_buffer.extend(input_tokens)
    
    def encode_current(self) -> torch.Tensor:
        """
        Encode the current context buffer into embeddings.
        
        Returns:
            Tensor of shape (sequence_length, embedding_dim)
        """
        if len(self.context_buffer) == 0:
            return torch.zeros(1, self.embedding_dim, device=self.device)
        
        # Convert context buffer to tensor
        context_tensor = torch.tensor(
            list(self.context_buffer),
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            embeddings = self.model(context_tensor)
        
        return embeddings.squeeze(0)  # Remove batch dimension
    
    def summarize(self, embeddings: torch.Tensor) -> np.ndarray:
        """
        Compress embeddings into a summary vector.
        
        Args:
            embeddings: Tensor of shape (sequence_length, embedding_dim)
        
        Returns:
            Summary vector of shape (embedding_dim,)
        """
        from hippo.compression import compress_embeddings
        
        # Mean pooling as default compression
        summary = compress_embeddings(embeddings)
        return summary
    
    def reconstruct_and_compare(
        self,
        summary: np.ndarray,
        shared_bridge
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compare summary to existing concepts in the semantic bridge.
        
        Args:
            summary: Current summary vector
            shared_bridge: SemanticBridge instance
        
        Returns:
            Tuple of (summary, difference) vectors
        """
        # Find most similar concept in bridge
        similar = shared_bridge.find_similar(summary)
        
        # Calculate novelty/difference
        if similar is not None:
            difference = summary - similar
        else:
            difference = summary  # All is novel if nothing similar found
        
        return summary, difference
    
    def produce_replay(
        self,
        summary: np.ndarray,
        difference: np.ndarray,
        novelty_weight: float = 0.5
    ) -> np.ndarray:
        """
        Generate a replay vector combining summary and novelty.
        
        Args:
            summary: Summary vector
            difference: Novelty/difference vector
            novelty_weight: Weight for novelty component (0-1)
        
        Returns:
            Replay vector for neocortex processing
        """
        # Blend summary and novelty
        replay_vector = (1 - novelty_weight) * summary + novelty_weight * difference
        
        # Normalize
        norm = np.linalg.norm(replay_vector)
        if norm > 0:
            replay_vector = replay_vector / norm
        
        return replay_vector
    
    def reset_context(self) -> None:
        """Clear the context buffer."""
        self.context_buffer.clear()
    
    def get_context_size(self) -> int:
        """Get current context buffer size."""
        return len(self.context_buffer)
