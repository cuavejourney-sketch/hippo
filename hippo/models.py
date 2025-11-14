"""
Simple transformer encoder models for embedding generation.
"""

import torch
import torch.nn as nn


class SimpleTransformerEncoder(nn.Module):
    """
    Simple transformer encoder for generating embeddings.
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        embedding_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 6,
        max_seq_length: int = 512
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.max_seq_length = max_seq_length
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_length)
        
        Returns:
            Embeddings of shape (batch_size, seq_length, embedding_dim)
        """
        batch_size, seq_length = input_ids.shape
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Position embeddings
        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(positions)
        
        # Combine
        embeddings = token_embeds + position_embeds
        
        # Transform
        output = self.transformer(embeddings)
        
        return output


class RandomEmbeddingModel(nn.Module):
    """
    Simple random embedding model for testing/demo purposes.
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        embedding_dim: int = 768
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize with small random values
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - just return embeddings.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_length)
        
        Returns:
            Embeddings of shape (batch_size, seq_length, embedding_dim)
        """
        return self.embedding(input_ids)


def get_default_encoder(embedding_dim: int = 768, model_type: str = "simple") -> nn.Module:
    """
    Get a default encoder model.
    
    Args:
        embedding_dim: Embedding dimension
        model_type: Type of model ('simple', 'random')
    
    Returns:
        Encoder model
    """
    if model_type == "simple":
        return SimpleTransformerEncoder(embedding_dim=embedding_dim)
    elif model_type == "random":
        return RandomEmbeddingModel(embedding_dim=embedding_dim)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
