"""Heuristics that mark Whisper-style segments as noise rather than speech."""

from .constants import (
    HALLUCINATION_PATTERNS,
    MIN_HALLUCINATION_TEXT_LENGTH,
    MIN_REPETITION_WORDS,
    REPETITION_RATIO_THRESHOLD,
)


def is_hallucination(text: str) -> bool:
    text_lower = text.lower().strip()
    if len(text_lower) < MIN_HALLUCINATION_TEXT_LENGTH:
        return True
    words = text_lower.split()
    repeated_word_ratio = 0.0
    if words:
        repeated_word_ratio = words.count(max(set(words), key=words.count)) / len(words)
    if len(words) >= MIN_REPETITION_WORDS and repeated_word_ratio > REPETITION_RATIO_THRESHOLD:
        return True
    return any(pattern in text_lower for pattern in HALLUCINATION_PATTERNS)
