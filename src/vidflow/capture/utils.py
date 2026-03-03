"""Utility functions for video capture."""

import re
from urllib.parse import parse_qs, urlparse

from dateutil import parser as date_parser

# Regex to find YouTube URLs in arbitrary text
_YOUTUBE_URL_RE = re.compile(
    r'https?://(?:www\.|m\.)?(?:youtube\.com|youtu\.be)'
    r'[^\s\)\]"\'>,]*'
)


def sanitize_title(title: str, max_length: int = 100) -> str:
    """Sanitize a title for use as a filename."""
    sanitized = re.sub(r'[<>:"/\\|?*]', ' ', title)
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rsplit(' ', 1)[0]
    return sanitized


def truncate_title_words(title: str, max_words: int = 6) -> str:
    """Truncate title to first N words."""
    words = title.split()
    if len(words) <= max_words:
        return title
    return ' '.join(words[:max_words])


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_date(date_str: str | None) -> str:
    """Parse a date string and return YYYY-MM-DD format."""
    if not date_str:
        return ""
    try:
        if re.match(r'^\d{8}$', date_str):
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        parsed = date_parser.parse(date_str)
        return parsed.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return ""


def extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from various URL formats."""
    parsed = urlparse(url)
    if parsed.netloc in ('youtu.be', 'www.youtu.be'):
        return parsed.path.lstrip('/')
    if parsed.netloc in ('youtube.com', 'www.youtube.com', 'm.youtube.com'):
        if parsed.path == '/watch':
            query = parse_qs(parsed.query)
            if 'v' in query:
                return query['v'][0]
        if parsed.path.startswith(('/embed/', '/v/')):
            parts = parsed.path.split('/')
            if len(parts) >= 3:
                return parts[2]
    return None


def extract_playlist_id(url: str) -> str | None:
    """Extract YouTube playlist ID from various URL formats."""
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if 'list' in query:
        return query['list'][0]
    return None


# YouTube video IDs are 11 characters: alphanumeric, hyphens, underscores
_VIDEO_ID_RE = re.compile(r'^[A-Za-z0-9_-]{11}$')


def is_video_id(text: str) -> bool:
    """Check if text is a bare YouTube video ID (11 chars, base64url alphabet)."""
    return bool(_VIDEO_ID_RE.match(text))


def video_id_to_url(video_id: str) -> str:
    """Convert a bare YouTube video ID to a full watch URL."""
    return f"https://www.youtube.com/watch?v={video_id}"


def is_video_url(url: str) -> bool:
    """Check if URL is a YouTube video URL."""
    return extract_video_id(url) is not None


def is_playlist_url(url: str) -> bool:
    """Check if URL is a YouTube playlist URL (without a video ID)."""
    if extract_video_id(url) is not None:
        return False
    return extract_playlist_id(url) is not None


def extract_youtube_urls(text: str) -> list[str]:
    """Extract all YouTube video and playlist URLs from arbitrary text."""
    seen: set[str] = set()
    urls: list[str] = []
    for match in _YOUTUBE_URL_RE.finditer(text):
        url = match.group(0)
        if url in seen:
            continue
        if is_video_url(url) or is_playlist_url(url):
            seen.add(url)
            urls.append(url)
    return urls
