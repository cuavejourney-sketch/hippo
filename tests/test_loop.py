"""
Tests for the main inference loop.
"""

import pytest
import numpy as np

from hippo import VolatileAgent, NeocortexAgent, SemanticBridge, run_dual_cascade
from hippo.loop import InferenceCycle, tokenize_text, create_text_stream


def test_inference_cycle_init():
    """Test InferenceCycle initialization."""
    volatile = VolatileAgent(window_size=32, embedding_dim=64)
    neocortex = NeocortexAgent(embedding_dim=64)
    bridge = SemanticBridge(embedding_dim=64)
    
    cycle = InferenceCycle(volatile, neocortex, bridge)
    
    assert cycle.cycle_count == 0
    assert cycle.target_cycle_ms == 60.0


def test_inference_cycle_step():
    """Test single inference cycle step."""
    volatile = VolatileAgent(window_size=32, embedding_dim=64)
    neocortex = NeocortexAgent(embedding_dim=64)
    bridge = SemanticBridge(embedding_dim=64)
    
    cycle = InferenceCycle(volatile, neocortex, bridge)
    
    tokens = [1, 2, 3, 4, 5]
    result = cycle.step(tokens)
    
    assert "cycle" in result
    assert "toki_output" in result
    assert "cycle_time_ms" in result
    assert result["cycle"] == 1


def test_inference_cycle_multiple_steps():
    """Test multiple inference cycles."""
    volatile = VolatileAgent(window_size=32, embedding_dim=64)
    neocortex = NeocortexAgent(embedding_dim=64)
    bridge = SemanticBridge(embedding_dim=64)
    
    cycle = InferenceCycle(volatile, neocortex, bridge)
    
    for i in range(5):
        tokens = list(range(i, i+5))
        result = cycle.step(tokens)
        assert result["cycle"] == i + 1


def test_run_dual_cascade():
    """Test running the dual cascade loop."""
    volatile = VolatileAgent(window_size=32, embedding_dim=64)
    neocortex = NeocortexAgent(embedding_dim=64)
    bridge = SemanticBridge(embedding_dim=64)
    
    # Create simple input stream
    def input_gen():
        for i in range(5):
            yield [i, i+1, i+2]
    
    results = list(run_dual_cascade(volatile, neocortex, bridge, input_gen(), max_cycles=5))
    
    assert len(results) == 5
    assert all("toki_output" in r for r in results)


def test_tokenize_text():
    """Test text tokenization."""
    text = "hello world test"
    
    tokens = tokenize_text(text)
    
    assert isinstance(tokens, list)
    assert len(tokens) == 3


def test_create_text_stream():
    """Test creating text stream."""
    texts = ["hello world", "test stream"]
    
    stream = create_text_stream(texts)
    tokens_list = list(stream)
    
    assert len(tokens_list) == 2
    assert all(isinstance(tokens, list) for tokens in tokens_list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
