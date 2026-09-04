"""Transcript fetching and parsing.

Two lanes: youtube-transcript-api (primary — fast, no full page
extraction) and yt-dlp subtitle extraction (fallback). YouTube blocks the
transcript API's raw caption endpoint for some videos — podcast-classified
and auto-dubbed uploads consistently, arbitrary IPs intermittently — while
yt-dlp's full extractor still reaches the same captions.
"""

import json
import random
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from youtube_transcript_api import (
    IpBlocked,
    RequestBlocked,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

from vidflow.capture.video import is_bot_gate


@dataclass
class TranscriptSegment:
    """A single segment of a transcript."""

    text: str
    start: float
    duration: float


class TranscriptError(Exception):
    """Exception raised for transcript-related errors."""

    pass


class TranscriptBlocked(TranscriptError):
    """YouTube is rate-limiting caption requests from this IP.

    Distinct from a missing transcript: the video may well have captions,
    but the caption endpoints refuse to serve them right now. Blocks on
    residential IPs are temporary (community reports: hours to ~48h).
    """

    pass


# Once a block is detected, further fetch attempts in this process are
# pointless (every video will fail the same way) — fail fast instead so
# bulk runs report the block rather than hammering the endpoint.
_block_detected = False

# Monotonic timestamp of the last caption request made by this process;
# None until the first one. Drives pacing between caption requests.
_last_caption_request: float | None = None

# Jittered spacing (seconds) enforced between consecutive caption requests
PACE_SPAN: tuple[float, float] = (2.0, 5.0)


def reset_block_state() -> None:
    """Clear the sticky block flag and pacing clock (tests, long-lived processes)."""
    global _block_detected, _last_caption_request
    _block_detected = False
    _last_caption_request = None


def pace_caption_request(log=None, span: tuple[float, float] = PACE_SPAN) -> float:
    """Sleep as needed so consecutive caption requests are spaced apart.

    YouTube's caption endpoints rate-limit bursts from one IP; thresholds
    are undocumented, but one observed trip (2026-08-29): ~40 caption
    requests (2 per video) at ~2s spacing blocked after video ~20. The
    pause is tied to caption requests only: nothing sleeps before the
    first request of the process, and time already spent since the last
    one (a video download, a failed metadata call) counts toward the
    jittered interval, so it mainly protects runs with tiny downloads
    (e.g., Shorts). Returns the seconds actually slept.
    """
    global _last_caption_request
    now = time.monotonic()
    slept = 0.0
    if _last_caption_request is not None:
        remaining = random.uniform(*span) - (now - _last_caption_request)
        if remaining > 0:
            if log:
                log(f"Pacing {remaining:.1f}s before next caption request (rate-limit courtesy)")
            time.sleep(remaining)
            slept = remaining
    _last_caption_request = time.monotonic()
    return slept


def get_transcript(
    video_id: str,
    language: str = "en",
    prefer_manual: bool = True,
    log=None,
) -> list[TranscriptSegment] | None:
    """Fetch transcript for a YouTube video, falling back to yt-dlp.

    Returns None when no transcript exists; raises TranscriptBlocked when
    YouTube is rate-limiting caption requests from this IP (sticky for the
    rest of the process, so bulk runs fail fast after the first block).
    Consecutive calls are paced (see pace_caption_request); ``log`` receives
    the pacing notice when a sleep is needed.
    """
    global _block_detected
    if _block_detected:
        raise TranscriptBlocked("caption requests from this IP are rate-limited")

    pace_caption_request(log=log)

    api_blocked = False
    try:
        segments = get_transcript_api(video_id, language, prefer_manual)
    except TranscriptBlocked:
        api_blocked = True
        segments = None
    if segments:
        return segments

    try:
        segments = get_transcript_ytdlp(video_id, language, prefer_manual)
    except TranscriptBlocked:
        _block_detected = True
        raise
    if segments:
        return segments

    if api_blocked:
        # The API lane was explicitly blocked and yt-dlp found nothing:
        # report the block, not a missing transcript
        _block_detected = True
        raise TranscriptBlocked("YouTube is rate-limiting caption requests from this IP")
    return None


def get_transcript_api(
    video_id: str,
    language: str = "en",
    prefer_manual: bool = True,
) -> list[TranscriptSegment] | None:
    """Fetch transcript via youtube-transcript-api (primary lane).

    Raises TranscriptBlocked when YouTube blocks the request; returns None
    for missing/disabled transcripts or any other failure.
    """
    try:
        ytt = YouTubeTranscriptApi()
        transcript_list = list(ytt.list(video_id))
    except TranscriptsDisabled:
        return None
    except (RequestBlocked, IpBlocked) as e:
        raise TranscriptBlocked(str(e)) from e
    except Exception:
        return None

    if not transcript_list:
        return None

    manual_transcripts = [t for t in transcript_list if not t.is_generated]
    generated_transcripts = [t for t in transcript_list if t.is_generated]

    transcript = None

    if prefer_manual:
        for t in manual_transcripts:
            if t.language_code.startswith(language):
                transcript = t
                break

    if transcript is None:
        for t in generated_transcripts:
            if t.language_code.startswith(language):
                transcript = t
                break

    if transcript is None and manual_transcripts:
        transcript = manual_transcripts[0]

    if transcript is None and generated_transcripts:
        transcript = generated_transcripts[0]

    if transcript is None and transcript_list:
        transcript = transcript_list[0]

    if transcript is None:
        return None

    try:
        raw_transcript = transcript.fetch()
        return [
            TranscriptSegment(
                text=segment.text,
                start=float(segment.start),
                duration=float(segment.duration),
            )
            for segment in raw_transcript
        ]
    except (RequestBlocked, IpBlocked) as e:
        raise TranscriptBlocked(str(e)) from e
    except Exception:
        return None


def parse_json3_captions(path: Path) -> list[TranscriptSegment]:
    """Parse a yt-dlp json3 caption file into transcript segments.

    json3 carries one event per caption cue with millisecond timing;
    filler events (newline-only or without text segments) are dropped.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    segments: list[TranscriptSegment] = []
    for event in data.get("events", []):
        segs = event.get("segs")
        if not segs:
            continue
        text = "".join(s.get("utf8", "") for s in segs).replace("\n", " ").strip()
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                text=text,
                start=event.get("tStartMs", 0) / 1000.0,
                duration=event.get("dDurationMs", 0) / 1000.0,
            )
        )
    return segments


def get_transcript_ytdlp(
    video_id: str,
    language: str = "en",
    prefer_manual: bool = True,
) -> list[TranscriptSegment] | None:
    """Fetch transcript via yt-dlp subtitle extraction (fallback lane).

    Requests json3 captions matching the language (including variants
    like en-orig on auto-dubbed videos). Manual subtitles and automatic
    captions are tried as separate passes in preference order. Raises
    TranscriptBlocked when the caption download is rate-limited (429)
    and no captions could be retrieved.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    passes = ["--write-subs", "--write-auto-subs"]
    if not prefer_manual:
        passes.reverse()

    saw_rate_limit = False
    for subs_flag in passes:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cmd = [
                "yt-dlp",
                "--skip-download",
                "--no-playlist",
                subs_flag,
                "--sub-langs",
                f"{language}.*,{language}",
                "--sub-format",
                "json3",
                "--no-warnings",
                "--remote-components",
                "ejs:github",
                "-o",
                str(tmp_path / "%(id)s"),
                url,
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            except Exception:
                continue

            if (
                "429" in result.stderr
                or "Too Many Requests" in result.stderr
                or is_bot_gate(result.stderr)
            ):
                saw_rate_limit = True

            # Exact language match first, then variants (e.g. en-orig on
            # auto-dubbed videos) in name order
            exact_name = f"{video_id}.{language}.json3"
            caption_files = sorted(
                tmp_path.glob(f"{video_id}.*.json3"),
                key=lambda p: (p.name != exact_name, p.name),
            )
            for caption_file in caption_files:
                try:
                    segments = parse_json3_captions(caption_file)
                except (json.JSONDecodeError, OSError):
                    continue
                if segments:
                    return segments

    if saw_rate_limit:
        raise TranscriptBlocked("YouTube is rate-limiting or bot-gating caption downloads")
    return None


def save_transcript_json(
    transcript: list[TranscriptSegment],
    path: Path,
) -> None:
    """Save transcript to a JSON file."""
    data = [asdict(segment) for segment in transcript]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
