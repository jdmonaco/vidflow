"""Local video metadata extraction using ffprobe."""

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from vidflow.capture.utils import sanitize_title


class LocalVideoError(Exception):
    """Exception raised for local video errors."""
    pass


@dataclass
class LocalVideoMetadata:
    """Metadata extracted from a local video file."""

    file_path: Path
    _base_title: str
    duration: float
    creation_date: str  # YYYYMMDD format
    _identifier_suffix: int = 0

    @property
    def identifier(self) -> str:
        base = self.file_path.stem
        if self._identifier_suffix > 0:
            return f"{base}-{self._identifier_suffix}"
        return base

    @property
    def title(self) -> str:
        if self._identifier_suffix > 0:
            return f"{self._base_title} ({self._identifier_suffix})"
        return self._base_title

    @property
    def author(self) -> str | None:
        return None

    @property
    def source_date(self) -> str:
        return self.creation_date

    @property
    def description(self) -> str:
        return ""

    @property
    def source_type(self) -> str:
        return 'video'


def check_ffprobe() -> bool:
    """Check if ffprobe is available in PATH."""
    import shutil
    return shutil.which('ffprobe') is not None


def get_local_video_metadata(video_path: Path) -> LocalVideoMetadata:
    """Extract metadata from a local video file using ffprobe."""
    if not video_path.exists():
        raise LocalVideoError(f"Video file not found: {video_path}")

    if not check_ffprobe():
        raise LocalVideoError(
            "ffprobe not found. Please install ffmpeg:\n"
            "  macOS: brew install ffmpeg\n"
            "  Ubuntu: sudo apt install ffmpeg\n"
            "  Windows: https://ffmpeg.org/download.html"
        )

    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        str(video_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            raise LocalVideoError(f"ffprobe failed: {result.stderr}")

        probe_data = json.loads(result.stdout)
        format_info = probe_data.get('format', {})

        duration_str = format_info.get('duration', '0')
        try:
            duration = float(duration_str)
        except (ValueError, TypeError):
            duration = 0.0

    except subprocess.TimeoutExpired:
        raise LocalVideoError("ffprobe timed out")
    except json.JSONDecodeError as e:
        raise LocalVideoError(f"Failed to parse ffprobe output: {e}") from e
    except FileNotFoundError:
        raise LocalVideoError("ffprobe not found")
    except LocalVideoError:
        raise
    except Exception as e:
        raise LocalVideoError(f"Unexpected error: {e}") from e

    base_title = sanitize_title(video_path.stem)

    try:
        mtime = video_path.stat().st_mtime
        creation_date = datetime.fromtimestamp(mtime).strftime('%Y%m%d')
    except Exception:
        creation_date = datetime.now().strftime('%Y%m%d')

    return LocalVideoMetadata(
        file_path=video_path,
        _base_title=base_title,
        duration=duration,
        creation_date=creation_date,
    )
