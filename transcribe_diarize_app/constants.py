MEETING_PROMPTS: dict[str, dict[str, str]] = {
    "interview": {
        "initial_prompt": "Job interview conversation.",
        "analysis_prompt": """Analyze this job interview transcript:

1. Strengths: What did the candidate do well?
2. Areas for Improvement: Where could they improve?
3. Communication Style: Clarity, confidence, professionalism
4. Technical Answers: Depth and accuracy assessment
5. Actionable Recommendations: Tips for future interviews

Transcript:
{transcript}""",
        "transcript_header": "Job Interview Transcript",
        "analysis_header": "Interview Analysis",
    },
    "generic": {
        "initial_prompt": "Professional meeting conversation.",
        "analysis_prompt": """Analyze this meeting transcript:

1. Key Topics: Main subjects discussed
2. Decisions Made: Any conclusions or agreements
3. Action Items: Tasks or follow-ups assigned
4. Open Questions: Unresolved items needing attention

Transcript:
{transcript}""",
        "transcript_header": "Meeting Transcript",
        "analysis_header": "Meeting Analysis",
    },
}
