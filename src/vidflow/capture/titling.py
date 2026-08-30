"""AI-powered title generation using the local quick slot (via aikit)."""

import logging
from dataclasses import dataclass

from vidflow.capture.utils import sanitize_title

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You generate concise, informative titles for YouTube video notes.
Output ONLY the title text, nothing else.

Rules:
1. Maximum 10 words
2. Begin with the key person or organization:
   - Single-person content (vlog, lecture): use the host's name
   - Interview with a guest: use the guest's name
   - Institutional/corporate content: use the organization's name
   - If unclear: use the channel name
3. After the name, use " - " (space dash space) then a concise descriptive phrase
4. Use Title Case for the entire title
5. The descriptive part should capture the core topic or thesis
6. Do not use quotes, colons, or special punctuation in the title
7. Use only ASCII characters - no unicode dashes, quotes, or accents

Example: "Ilya Sutskever - Moving from Scaling to Research"
"""

_MODEL = "quick"  # resident gateway slot; titling is a small local task
# Local models reason before answering and reasoning tokens count against
# max_tokens -- a tight budget truncates the visible title mid-word.
_MAX_TOKENS = 500
_TEMPERATURE = 0.3
_TIMEOUT = 15.0  # keep capture snappy; fall back rather than wait on a JIT load


@dataclass
class TitleResult:
    """Result of AI title generation."""

    ai_title: str
    original_title: str
    used_ai: bool


def is_ai_titling_available() -> bool:
    """Check if AI titling is available.

    The local gateway needs no API key; any failure (host asleep, slot
    unavailable) is handled by generate_ai_title's catch-all fallback, so
    this is a cheap import check only.
    """
    try:
        import aikit  # noqa: F401

        return True
    except ImportError:
        return False


def _validate_title(title: str) -> bool:
    """Validate that a generated title meets basic criteria."""
    words = title.split()
    if len(words) < 2 or len(words) > 12:
        return False
    if len(title) < 10 or len(title) > 150:
        return False
    return True


def _clean_title(raw: str) -> str:
    """Clean up raw LLM output to extract the title."""
    title = raw.strip()
    if len(title) >= 2 and title[0] in ('"', "'") and title[-1] == title[0]:
        title = title[1:-1].strip()
    title = title.lstrip("#").strip()
    return title


def generate_ai_title(
    title: str,
    channel: str,
    description: str,
) -> TitleResult:
    """Generate an AI-powered title for a YouTube video.

    Falls back to the original title on any failure -- an unreachable
    gateway, an unavailable slot, or an invalid generation.
    """
    try:
        import aikit
    except ImportError:
        logger.debug("aikit not installed, falling back to original title")
        return TitleResult(ai_title=title, original_title=title, used_ai=False)

    user_message = f"Title: {title}\n" f"Channel: {channel}\n" f"Description: {description[:500]}"

    try:
        client = aikit.local_client(timeout=_TIMEOUT)
        response = client.chat.completions.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            temperature=_TEMPERATURE,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        raw_title = response.choices[0].message.content or ""
        cleaned = _clean_title(raw_title)

        if not _validate_title(cleaned):
            logger.debug("AI title failed validation: %r", cleaned)
            return TitleResult(ai_title=title, original_title=title, used_ai=False)

        sanitized = sanitize_title(cleaned)
        if not sanitized:
            logger.debug("AI title empty after sanitization: %r", cleaned)
            return TitleResult(ai_title=title, original_title=title, used_ai=False)

        return TitleResult(ai_title=sanitized, original_title=title, used_ai=True)

    except Exception as e:
        logger.debug("AI title generation failed: %s", e)
        return TitleResult(ai_title=title, original_title=title, used_ai=False)
