SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "pt": "Portuguese",
}

MIN_HALLUCINATION_TEXT_LENGTH = 3
MIN_REPETITION_WORDS = 4
REPETITION_RATIO_THRESHOLD = 0.6

HALLUCINATION_PATTERNS: list[str] = [
    "thank you. thank you.",
    "thank you thank you",
    "openness openness",
    "finding out when",
    "we discovered when we discovered",
    "yeah. yeah. yeah. yeah",
    "it was good when",
]

MEETING_PROMPTS: dict[str, dict[str, str]] = {
    "interview": {"transcript_header": "Job Interview Transcript"},
    "generic": {"transcript_header": "Meeting Transcript"},
}
