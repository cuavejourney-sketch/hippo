"""
Toki Pona language renderer for conceptual abstraction.

Toki pona is a minimal constructed language with ~120 words,
perfect for representing abstract concepts.
"""

from typing import List, Optional
import numpy as np


# Core toki pona vocabulary organized by semantic categories
TOKI_PONA_VOCAB = {
    # Existence and being
    "existence": ["ijo", "ali", "wan"],  # thing, all, one
    
    # Action and change
    "action": ["pali", "kama", "tawa", "awen"],  # do, become, move, stay
    
    # Qualities
    "positive": ["pona", "suli", "wawa"],  # good, big, strong
    "negative": ["ike", "lili", "weka"],  # bad, small, away
    
    # Knowledge
    "cognitive": ["sona", "lukin", "kute"],  # know, see, hear
    
    # Time
    "temporal": ["tenpo", "sin", "pini"],  # time, new, past
    
    # Social
    "social": ["jan", "toki", "pana"],  # person, speak, give
    
    # Abstract
    "abstract": ["nasin", "lawa", "pilin"],  # way, head/guide, feel
}


def render_toki_pona(
    predictions: List[np.ndarray],
    memory_clusters: Optional[np.ndarray] = None,
    max_words: int = 10
) -> str:
    """
    Convert prediction vectors to toki pona sentence.
    
    Args:
        predictions: List of prediction vectors
        memory_clusters: Optional cluster centers for context
        max_words: Maximum words in output sentence
    
    Returns:
        Toki pona sentence
    """
    if not predictions:
        return "ala"  # nothing
    
    words = []
    
    # Analyze prediction characteristics
    for pred_vec in predictions[:max_words]:
        word = _vector_to_toki_word(pred_vec)
        if word and word not in words:
            words.append(word)
    
    if not words:
        return "ala"
    
    # Construct simple sentence
    sentence = _construct_toki_sentence(words)
    
    return sentence


def _vector_to_toki_word(vector: np.ndarray) -> str:
    """
    Map a vector to a toki pona word based on its characteristics.
    
    Args:
        vector: Concept vector
    
    Returns:
        Toki pona word
    """
    # Analyze vector properties
    magnitude = np.linalg.norm(vector)
    mean_val = np.mean(vector)
    std_val = np.std(vector)
    positive_ratio = np.sum(vector > 0) / len(vector)
    
    # Map to semantic categories
    if magnitude > 1.5:
        # Strong signal
        if mean_val > 0:
            return np.random.choice(TOKI_PONA_VOCAB["positive"])
        else:
            return np.random.choice(TOKI_PONA_VOCAB["negative"])
    
    elif std_val > 0.5:
        # High variance - action/change
        return np.random.choice(TOKI_PONA_VOCAB["action"])
    
    elif positive_ratio > 0.7:
        # Mostly positive - knowledge/cognitive
        return np.random.choice(TOKI_PONA_VOCAB["cognitive"])
    
    elif positive_ratio < 0.3:
        # Mostly negative - temporal
        return np.random.choice(TOKI_PONA_VOCAB["temporal"])
    
    else:
        # Neutral - existence
        return np.random.choice(TOKI_PONA_VOCAB["existence"])


def _construct_toki_sentence(words: List[str]) -> str:
    """
    Construct a grammatical toki pona sentence from words.
    
    Args:
        words: List of toki pona words
    
    Returns:
        Grammatical sentence
    """
    if not words:
        return "ala"
    
    if len(words) == 1:
        return words[0]
    
    # Simple sentence patterns
    # Pattern: [subject] [li] [predicate]
    # Pattern: [subject] [predicate] (for mi/sina)
    
    subject = words[0]
    
    if len(words) == 2:
        # Simple: subject + predicate
        return f"{subject} li {words[1]}"
    
    else:
        # Complex: subject + predicate + modifiers
        predicate = words[1]
        modifiers = " ".join(words[2:])
        return f"{subject} li {predicate} {modifiers}"


def translate_to_english(toki_sentence: str) -> str:
    """
    Provide rough English translation of toki pona sentence.
    
    Args:
        toki_sentence: Toki pona sentence
    
    Returns:
        English approximation
    """
    # Simple word-by-word translation dictionary
    translations = {
        "ijo": "thing",
        "ali": "everything",
        "wan": "one",
        "pali": "work/do",
        "kama": "come/become",
        "tawa": "go/move",
        "awen": "stay/keep",
        "pona": "good",
        "suli": "big/important",
        "wawa": "strong/powerful",
        "ike": "bad",
        "lili": "small",
        "weka": "away/absent",
        "sona": "know/knowledge",
        "lukin": "see/look",
        "kute": "hear/listen",
        "tenpo": "time",
        "sin": "new",
        "pini": "end/past",
        "jan": "person",
        "toki": "speak/language",
        "pana": "give",
        "nasin": "way/path",
        "lawa": "head/lead",
        "pilin": "feel/heart",
        "li": "is/does",
        "ala": "nothing/no",
    }
    
    words = toki_sentence.split()
    english_words = [translations.get(word, word) for word in words]
    
    return " ".join(english_words)


def concepts_to_toki_pona(concepts: List[str]) -> str:
    """
    Convert abstract concepts to toki pona representation.
    
    Args:
        concepts: List of concept strings
    
    Returns:
        Toki pona sentence
    """
    # Map concepts to toki pona
    concept_map = {
        "memory": "sona",
        "learning": "kama sona",
        "understanding": "sona pona",
        "change": "ante",
        "stability": "awen",
        "pattern": "nasin",
        "new": "sin",
        "old": "pini",
        "important": "suli",
        "connection": "kulupu",
    }
    
    toki_words = []
    for concept in concepts:
        concept_lower = concept.lower()
        if concept_lower in concept_map:
            toki_words.append(concept_map[concept_lower])
        else:
            # Default to generic words
            toki_words.append("ijo")
    
    return _construct_toki_sentence(toki_words[:5])
