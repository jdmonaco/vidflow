"""Data models for video frame transcription."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class TimestampSection:
    """Represents a single timestamp section from a vidcapture markdown file."""

    timestamp: str  # "00:05:30"
    image_embed: str  # "![[images/id/frame-0010.jpg]]"
    image_path: Path  # Resolved absolute path
    existing_text: str = ""  # Pre-existing transcript text (e.g., YouTube auto-captions)
    content: str = ""  # Filled transcript


@dataclass
class VidcaptureDocument:
    """Represents a parsed vidcapture markdown file."""

    source_path: Path
    frontmatter: str  # Raw YAML block to preserve
    title: str
    sections: List[TimestampSection] = field(default_factory=list)
