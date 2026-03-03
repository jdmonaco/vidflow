"""Video metadata extraction and download using yt-dlp."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from vidflow.capture.utils import extract_video_id


@dataclass
class VideoMetadata:
    """Metadata extracted from a YouTube video."""

    video_id: str
    title: str
    channel: str
    upload_date: str
    description: str
    duration: float
    _original_title: str = ""

    @property
    def identifier(self) -> str:
        return self.video_id

    @property
    def author(self) -> str:
        return self.channel

    @property
    def source_date(self) -> str:
        return self.upload_date

    @property
    def source_type(self) -> str:
        return 'youtube'


class VideoError(Exception):
    """Exception raised for video-related errors."""
    pass


def get_video_metadata(url: str) -> VideoMetadata:
    """Extract metadata from a YouTube video."""
    cmd = [
        'yt-dlp',
        '--dump-json',
        '--skip-download',
        '--no-warnings',
        '--remote-components', 'ejs:github',
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            error_msg = result.stderr
            if 'Private video' in error_msg:
                raise VideoError(f"Video is private: {url}")
            if 'Video unavailable' in error_msg:
                raise VideoError(f"Video is unavailable: {url}")
            if 'Sign in' in error_msg:
                raise VideoError(f"Video requires authentication: {url}")
            raise VideoError(f"Failed to get video metadata: {error_msg}")

        info = json.loads(result.stdout)
        video_id = extract_video_id(url) or info.get('id', '')

        return VideoMetadata(
            video_id=video_id,
            title=info.get('title', 'Untitled'),
            channel=info.get('channel', info.get('uploader', 'Unknown')),
            upload_date=info.get('upload_date', ''),
            description=info.get('description', ''),
            duration=float(info.get('duration', 0)),
        )

    except subprocess.TimeoutExpired:
        raise VideoError("Metadata extraction timed out")
    except json.JSONDecodeError as e:
        raise VideoError(f"Failed to parse video metadata: {e}") from e
    except FileNotFoundError:
        raise VideoError(
            "yt-dlp not found. Please install yt-dlp:\n"
            "  pip install yt-dlp\n"
            "  or: brew install yt-dlp"
        )
    except VideoError:
        raise
    except Exception as e:
        raise VideoError(f"Unexpected error getting metadata: {e}") from e


def get_stream_url(url: str) -> str:
    """Get the direct stream URL for a YouTube video."""
    format_spec = 'bestvideo[height<=480][ext=mp4]/bestvideo[height<=480]/18/best'
    cmd = [
        'yt-dlp',
        '--format', format_spec,
        '--get-url',
        '--no-warnings',
        '--remote-components', 'ejs:github',
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise VideoError(f"Failed to get stream URL: {result.stderr}")
        stream_url = result.stdout.strip()
        if not stream_url:
            raise VideoError("No stream URL returned")
        return stream_url
    except subprocess.TimeoutExpired:
        raise VideoError("Stream URL extraction timed out")
    except FileNotFoundError:
        raise VideoError("yt-dlp not found. Please install yt-dlp.")
    except VideoError:
        raise
    except Exception as e:
        raise VideoError(f"Unexpected error getting stream URL: {e}") from e


def download_video(url: str, output_dir: Path) -> Path:
    """Download video to local file."""
    format_spec = (
        'bestvideo[height<=1080][height>=720][ext=mp4]+bestaudio[ext=m4a]/'
        'bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/'
        '18/best'
    )
    output_template = str(output_dir / '%(id)s.mp4')

    cmd = [
        'yt-dlp',
        '--format', format_spec,
        '--output', output_template,
        '--no-warnings',
        '--remote-components', 'ejs:github',
        '--merge-output-format', 'mp4',
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            error_msg = result.stderr
            if 'Private video' in error_msg:
                raise VideoError(f"Video is private: {url}")
            if 'Video unavailable' in error_msg:
                raise VideoError(f"Video is unavailable: {url}")
            if 'Sign in' in error_msg:
                raise VideoError(f"Video requires authentication: {url}")
            raise VideoError(f"Failed to download video: {error_msg}")

        video_files = list(output_dir.glob('*.mp4'))
        if not video_files:
            video_files = list(output_dir.glob('*.webm'))
        if not video_files:
            video_files = list(output_dir.glob('*.mkv'))

        if not video_files:
            raise VideoError("Download completed but no video file found")

        return video_files[0]

    except subprocess.TimeoutExpired:
        raise VideoError("Video download timed out (10 minute limit)")
    except FileNotFoundError:
        raise VideoError(
            "yt-dlp not found. Please install yt-dlp:\n"
            "  pip install yt-dlp\n"
            "  or: brew install yt-dlp"
        )
    except VideoError:
        raise
    except Exception as e:
        raise VideoError(f"Unexpected error downloading video: {e}") from e


def expand_playlist(url: str) -> list[str]:
    """Expand a YouTube playlist URL to a list of video URLs."""
    cmd = [
        'yt-dlp',
        '--dump-json',
        '--flat-playlist',
        '--skip-download',
        '--no-warnings',
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            error_msg = result.stderr
            if 'Private' in error_msg:
                raise VideoError(f"Playlist is private: {url}")
            if 'not exist' in error_msg or 'unavailable' in error_msg.lower():
                raise VideoError(f"Playlist not found: {url}")
            raise VideoError(f"Failed to expand playlist: {error_msg}")

        video_urls: list[str] = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            try:
                entry = json.loads(line)
                video_id = entry.get('id')
                if video_id:
                    video_urls.append(f"https://www.youtube.com/watch?v={video_id}")
            except json.JSONDecodeError:
                continue

        return video_urls

    except subprocess.TimeoutExpired:
        raise VideoError("Playlist expansion timed out")
    except FileNotFoundError:
        raise VideoError(
            "yt-dlp not found. Please install yt-dlp:\n"
            "  pip install yt-dlp\n"
            "  or: brew install yt-dlp"
        )
    except VideoError:
        raise
    except Exception as e:
        raise VideoError(f"Unexpected error expanding playlist: {e}") from e
