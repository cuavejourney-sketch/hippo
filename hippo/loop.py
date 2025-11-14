"""
Main inference loop for the dual-agent cascade system.
"""

from typing import Iterator, Optional, Generator, Any
import time
import numpy as np

from hippo.volatile_agent import VolatileAgent
from hippo.neocortex_agent import NeocortexAgent
from hippo.semantic_bridge import SemanticBridge


class InferenceCycle:
    """
    Manages the recurrent inference cycle between agents.
    
    Target: 50-60ms per cycle (adjustable)
    """
    
    def __init__(
        self,
        volatile: VolatileAgent,
        neocortex: NeocortexAgent,
        bridge: SemanticBridge,
        target_cycle_ms: float = 60.0,
        consolidation_interval: int = 10
    ):
        """
        Initialize the inference cycle.
        
        Args:
            volatile: Volatile agent instance
            neocortex: Neocortex agent instance
            bridge: Semantic bridge instance
            target_cycle_ms: Target cycle time in milliseconds
            consolidation_interval: Cycles between consolidation
        """
        self.volatile = volatile
        self.neocortex = neocortex
        self.bridge = bridge
        self.target_cycle_ms = target_cycle_ms
        self.consolidation_interval = consolidation_interval
        
        self.cycle_count = 0
        self.total_time = 0.0
    
    def step(self, input_tokens: list) -> dict:
        """
        Perform one inference cycle.
        
        Args:
            input_tokens: New input tokens
        
        Returns:
            Dict with cycle results
        """
        start_time = time.time()
        
        # 1. Volatile agent observes input
        self.volatile.observe(input_tokens)
        
        # 2. Encode current context
        embeddings = self.volatile.encode_current()
        
        # 3. Summarize
        summary = self.volatile.summarize(embeddings)
        
        # 4. Compare with bridge
        summary, difference = self.volatile.reconstruct_and_compare(summary, self.bridge)
        
        # 5. Generate replay vector
        replay = self.volatile.produce_replay(summary, difference)
        
        # 6. Add to semantic bridge
        self.bridge.add(summary, importance=np.linalg.norm(difference))
        
        # 7. Neocortex receives and compresses
        compressed = self.neocortex.receive(replay, metadata={"cycle": self.cycle_count})
        
        # 8. Periodically consolidate
        if self.cycle_count % self.consolidation_interval == 0:
            self.neocortex.consolidate()
        
        # 9. Generate predictions
        clusters = self.neocortex.generalize()
        predictions = self.neocortex.predict(clusters) if clusters is not None else []
        
        # 10. Speak in toki pona
        toki_sentence = self.neocortex.speak_toki_pona(predictions)
        
        # Timing
        cycle_time = (time.time() - start_time) * 1000  # Convert to ms
        self.cycle_count += 1
        self.total_time += cycle_time
        
        return {
            "cycle": self.cycle_count,
            "toki_output": toki_sentence,
            "predictions": predictions,
            "cycle_time_ms": cycle_time,
            "avg_time_ms": self.total_time / self.cycle_count,
            "context_size": self.volatile.get_context_size(),
            "memory_stats": self.neocortex.get_memory_stats(),
            "bridge_stats": self.bridge.get_stats()
        }
    
    def get_stats(self) -> dict:
        """Get overall statistics."""
        return {
            "total_cycles": self.cycle_count,
            "total_time_s": self.total_time / 1000,
            "avg_cycle_time_ms": self.total_time / self.cycle_count if self.cycle_count > 0 else 0,
            "target_cycle_ms": self.target_cycle_ms,
            "memory_stats": self.neocortex.get_memory_stats(),
            "bridge_stats": self.bridge.get_stats()
        }


def run_dual_cascade(
    volatile: VolatileAgent,
    neocortex: NeocortexAgent,
    bridge: SemanticBridge,
    input_stream: Iterator[list],
    max_cycles: Optional[int] = None,
    verbose: bool = False
) -> Generator[dict, None, None]:
    """
    Run the dual-agent cascade loop.
    
    Args:
        volatile: Volatile agent instance
        neocortex: Neocortex agent instance
        bridge: Semantic bridge instance
        input_stream: Iterator yielding token lists
        max_cycles: Maximum number of cycles (None = unlimited)
        verbose: Print detailed output
    
    Yields:
        Dict with cycle results
    """
    cycle = InferenceCycle(volatile, neocortex, bridge)
    
    for i, tokens in enumerate(input_stream):
        if max_cycles is not None and i >= max_cycles:
            break
        
        result = cycle.step(tokens)
        
        if verbose:
            print(f"Cycle {result['cycle']}: {result['toki_output']} "
                  f"({result['cycle_time_ms']:.2f}ms)")
        
        yield result


def tokenize_text(text: str, vocab_size: int = 50000) -> list:
    """
    Simple tokenizer for demo purposes.
    
    Args:
        text: Input text
        vocab_size: Vocabulary size
    
    Returns:
        List of token IDs
    """
    # Very simple: hash each character/word to token ID
    tokens = []
    for word in text.split():
        token_id = hash(word) % vocab_size
        tokens.append(token_id)
    
    return tokens


def create_text_stream(texts: list, vocab_size: int = 50000) -> Generator[list, None, None]:
    """
    Create a token stream from text list.
    
    Args:
        texts: List of text strings
        vocab_size: Vocabulary size
    
    Yields:
        Token lists
    """
    for text in texts:
        tokens = tokenize_text(text, vocab_size)
        yield tokens
