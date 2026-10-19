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


MEETING_SUMMARY_PROMPT = """You are writing the summary of a recorded meeting from its transcript.
"Me" is the person who recorded the meeting. "Remote N" labels are other participants.

Write these sections in markdown:

## Main points
## Decisions
## Action items
For each item: the task, the owner, and the deadline. Write "not stated" when the
transcript does not name an owner or a deadline.
## Open questions

Rules:
- Use only information in the transcript. Never invent decisions, owners, or deadlines.
- Write "None identified." for a section with no content.
- Treat "[inaudible]" as missing speech; do not guess what was said.
- Write the summary in the same language as the transcript.

Transcript:
{transcript}"""
