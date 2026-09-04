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
        return "youtube"


class VideoError(Exception):
    """Exception raised for video-related errors."""

    pass


class VideoBlocked(VideoError):
    """YouTube is bot-gating page requests from this IP.

    yt-dlp reports this as "Sign in to confirm you're not a bot". Like a
    caption 429 it is an IP-level condition, not a property of the video:
    every further request will fail the same way and deepen the block.
    """

    pass


def is_bot_gate(stderr: str) -> bool:
    """True when yt-dlp output carries YouTube's sign-in bot challenge."""
    return "not a bot" in stderr


def get_video_metadata(url: str) -> VideoMetadata:
    """Extract metadata from a YouTube video."""
    cmd = [
        "yt-dlp",
        "--dump-json",
        "--skip-download",
        "--no-playlist",
        "--no-warnings",
        "--remote-components",
        "ejs:github",
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            error_msg = result.stderr
            if "Private video" in error_msg:
                raise VideoError(f"Video is private: {url}")
            if "Video unavailable" in error_msg:
                raise VideoError(f"Video is unavailable: {url}")
            if is_bot_gate(error_msg):
                raise VideoBlocked(
                    "YouTube is bot-gating requests from this IP (sign-in challenge)"
                )
            if "Sign in" in error_msg:
                raise VideoError(f"Video requires authentication: {url}")
            raise VideoError(f"Failed to get video metadata: {error_msg}")

        info = json.loads(result.stdout)
        video_id = extract_video_id(url) or info.get("id", "")

        return VideoMetadata(
            video_id=video_id,
            title=info.get("title", "Untitled"),
            channel=info.get("channel", info.get("uploader", "Unknown")),
            upload_date=info.get("upload_date", ""),
            description=info.get("description", ""),
            duration=float(info.get("duration", 0)),
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
    format_spec = "bestvideo[height<=480][ext=mp4]/bestvideo[height<=480]/18/best"
    cmd = [
        "yt-dlp",
        "--format",
        format_spec,
        "--get-url",
        "--no-playlist",
        "--no-warnings",
        "--remote-components",
        "ejs:github",
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


@dataclass
class VideoFetch:
    """Everything one yt-dlp extraction yields for a video.

    Captions are whatever yt-dlp wrote alongside the download as json3
    files (manual subtitles take precedence over automatic captions for
    the same language key, so at most one file per language variant).
    """

    metadata: VideoMetadata
    video_path: Path
    caption_files: list[Path]
    manual_langs: frozenset[str]  # languages with manual (non-generated) subtitles
    captions_blocked: bool  # a caption download was refused with 429


def _classify_ytdlp_failure(error_msg: str, url: str, what: str) -> VideoError:
    """Map yt-dlp stderr to the exception a failed call should raise."""
    if "Private video" in error_msg:
        return VideoError(f"Video is private: {url}")
    if "Video unavailable" in error_msg:
        return VideoError(f"Video is unavailable: {url}")
    if is_bot_gate(error_msg):
        return VideoBlocked("YouTube is bot-gating requests from this IP (sign-in challenge)")
    if "Sign in" in error_msg:
        return VideoError(f"Video requires authentication: {url}")
    return VideoError(f"Failed to {what}: {error_msg}")


def _video_files(output_dir: Path, video_id: str) -> list[Path]:
    """Downloaded media files for a video id (merged output is mp4)."""
    return sorted(
        f for f in output_dir.glob(f"{video_id}.*") if f.suffix in (".mp4", ".webm", ".mkv")
    )


def fetch_video(url: str, output_dir: Path, language: str = "en") -> VideoFetch:
    """Fetch metadata, captions, and the video file in one yt-dlp call.

    YouTube's bot gate budgets page extractions per IP, and one bulk run
    was gated after twelve videos when metadata, captions, and download
    were separate calls. A single extraction writes the info JSON, any
    json3 captions matching ``language`` (exact and variants such as
    en-orig), and the video into ``output_dir``; the info JSON is consumed
    and removed here, caption files are left for the transcript stage.
    Runs with --ignore-errors so a refused caption download (429) is
    reported but does not abandon the video download; success is judged
    by the video file's presence, and warnings are kept in stderr so the
    429 is still visible.
    """
    format_spec = (
        "bestvideo[height<=1080][height>=720][ext=mp4]+bestaudio[ext=m4a]/"
        "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/"
        "18/best"
    )
    output_template = str(output_dir / "%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--format",
        format_spec,
        "--output",
        output_template,
        "--merge-output-format",
        "mp4",
        "--write-info-json",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs",
        f"{language}.*,{language}",
        "--sub-format",
        "json3",
        "--ignore-errors",
        "--no-playlist",
        "--remote-components",
        "ejs:github",
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        raise VideoError("Video download timed out (10 minute limit)") from None
    except FileNotFoundError:
        raise VideoError(
            "yt-dlp not found. Please install yt-dlp:\n"
            "  pip install yt-dlp\n"
            "  or: brew install yt-dlp"
        ) from None

    video_id = extract_video_id(url)
    info_files = sorted(output_dir.glob(f"{video_id or '*'}.info.json"))
    video_files = _video_files(output_dir, video_id) if video_id else []
    if result.returncode != 0 and not (info_files and video_files):
        raise _classify_ytdlp_failure(result.stderr, url, "fetch video")
    if not info_files:
        raise VideoError("Download completed but no info JSON found")
    info_path = info_files[0]
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise VideoError(f"Failed to parse video metadata: {e}") from e
    info_path.unlink(missing_ok=True)
    video_id = video_id or info.get("id", "")

    metadata = VideoMetadata(
        video_id=video_id,
        title=info.get("title", "Untitled"),
        channel=info.get("channel", info.get("uploader", "Unknown")),
        upload_date=info.get("upload_date", ""),
        description=info.get("description", ""),
        duration=float(info.get("duration", 0)),
    )

    video_files = video_files or _video_files(output_dir, video_id)
    if not video_files:
        raise VideoError("Download completed but no video file found")
    caption_files = sorted(output_dir.glob(f"{video_id}.*.json3"))

    stderr = result.stderr
    captions_blocked = "HTTP Error 429" in stderr or "Too Many Requests" in stderr
    manual_langs = frozenset((info.get("subtitles") or {}).keys())

    return VideoFetch(
        metadata=metadata,
        video_path=video_files[0],
        caption_files=caption_files,
        manual_langs=manual_langs,
        captions_blocked=captions_blocked,
    )


def normalize_video_urls(urls: list[str], log=None) -> list[str]:
    """Classify and expand capture inputs into unique video watch URLs.

    Accepts bare video IDs (normalized to watch URLs), playlist URLs
    (expanded via yt-dlp), and plain video URLs; anything else is skipped
    with a note. Duplicates are dropped, order preserved. Watch URLs that
    also carry a list= parameter pass through unchanged — the single-video
    yt-dlp calls pin them with --no-playlist. `log` receives plain-text
    progress messages when provided.

    Shared by the ytcapture standalone CLI and vidflow youtube, including
    both clipboard fallbacks.
    """
    from vidflow.capture.utils import (
        is_playlist_url,
        is_video_id,
        is_video_url,
        video_id_to_url,
    )

    def _log(msg: str) -> None:
        if log:
            log(msg)

    video_urls: list[str] = []
    for url in urls:
        if is_video_id(url):
            full_url = video_id_to_url(url)
            _log(f"Video ID: {url} -> {full_url}")
            video_urls.append(full_url)
        elif is_playlist_url(url):
            _log(f"Expanding playlist: {url}")
            try:
                playlist_videos = expand_playlist(url)
            except VideoError as e:
                _log(f"! Failed to expand playlist: {e}")
                continue
            _log(f"+ Found {len(playlist_videos)} videos in playlist")
            video_urls.extend(playlist_videos)
        elif is_video_url(url):
            video_urls.append(url)
        else:
            _log(f"! Skipping invalid URL: {url}")

    seen: set[str] = set()
    unique_urls: list[str] = []
    for url in video_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    return unique_urls


def expand_playlist(url: str) -> list[str]:
    """Expand a YouTube playlist URL to a list of video URLs."""
    cmd = [
        "yt-dlp",
        "--dump-json",
        "--flat-playlist",
        "--skip-download",
        "--no-warnings",
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            error_msg = result.stderr
            if "Private" in error_msg:
                raise VideoError(f"Playlist is private: {url}")
            if "not exist" in error_msg or "unavailable" in error_msg.lower():
                raise VideoError(f"Playlist not found: {url}")
            raise VideoError(f"Failed to expand playlist: {error_msg}")

        video_urls: list[str] = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            try:
                entry = json.loads(line)
                video_id = entry.get("id")
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
