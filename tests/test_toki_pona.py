"""
Tests for toki pona rendering.
"""

import pytest
import numpy as np

from hippo.toki_pona import (
    render_toki_pona,
    translate_to_english,
    concepts_to_toki_pona
)


def test_render_toki_pona():
    """Test rendering toki pona from predictions."""
    predictions = [np.random.randn(128) for _ in range(3)]
    
    sentence = render_toki_pona(predictions)
    
    assert isinstance(sentence, str)
    assert len(sentence) > 0


def test_render_toki_pona_empty():
    """Test rendering with no predictions."""
    sentence = render_toki_pona([])
    
    assert sentence == "ala"


def test_translate_to_english():
    """Test translating toki pona to English."""
    toki_sentence = "ijo li pona"
    
    english = translate_to_english(toki_sentence)
    
    assert isinstance(english, str)
    assert "thing" in english
    assert "good" in english


def test_concepts_to_toki_pona():
    """Test converting concepts to toki pona."""
    concepts = ["memory", "learning", "change"]
    
    sentence = concepts_to_toki_pona(concepts)
    
    assert isinstance(sentence, str)
    assert len(sentence) > 0


def test_concepts_to_toki_pona_unknown():
    """Test with unknown concepts."""
    concepts = ["unknown_concept_xyz"]
    
    sentence = concepts_to_toki_pona(concepts)
    
    assert isinstance(sentence, str)
    # Should fall back to generic word
    assert "ijo" in sentence


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
