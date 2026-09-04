"""Transcript selection and parsing.

Two lanes: json3 caption files that yt-dlp wrote alongside the video
download (primary — no extra request, see ``fetch_video``) and
youtube-transcript-api (fallback, paced). YouTube blocks the transcript
API's raw caption endpoint for some videos — podcast-classified and
auto-dubbed uploads consistently, arbitrary IPs intermittently — while
yt-dlp's full extractor still reaches the same captions, which is one
more reason the download's captions come first.
"""

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from youtube_transcript_api import (
    IpBlocked,
    RequestBlocked,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)


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


def transcript_from_caption_files(
    caption_files: list[Path],
    video_id: str,
    language: str = "en",
    prefer_manual: bool = True,
    manual_langs: frozenset[str] = frozenset(),
) -> list[TranscriptSegment] | None:
    """Pick and parse the best json3 caption file yt-dlp wrote for a video.

    Files are named ``<video_id>.<lang>.json3``. The exact language match
    wins over variants (e.g. en-orig on auto-dubbed videos); with
    ``prefer_manual`` languages listed in ``manual_langs`` are tried first.
    Returns None when no file yields segments.
    """
    prefix = f"{video_id}."

    def lang_of(path: Path) -> str:
        return path.name[len(prefix) : -len(".json3")]

    candidates = [f for f in caption_files if f.name.startswith(prefix) and f.suffix == ".json3"]
    candidates.sort(
        key=lambda f: (
            prefer_manual and lang_of(f) not in manual_langs,
            lang_of(f) != language,
            f.name,
        )
    )
    for caption_file in candidates:
        try:
            segments = parse_json3_captions(caption_file)
        except (json.JSONDecodeError, OSError):
            continue
        if segments:
            return segments
    return None


def get_transcript(
    video_id: str,
    language: str = "en",
    prefer_manual: bool = True,
    log=None,
    caption_files: list[Path] | None = None,
    manual_langs: frozenset[str] = frozenset(),
    captions_blocked: bool = False,
) -> list[TranscriptSegment] | None:
    """Resolve a video's transcript: downloaded captions first, then the API.

    ``caption_files`` are the json3 files from ``fetch_video``; when one
    parses, no further request is made. Otherwise the transcript API lane
    runs, paced against the previous API request (see
    pace_caption_request; ``log`` receives the pacing notice). Returns
    None when no transcript exists. Raises TranscriptBlocked when YouTube
    is rate-limiting caption requests — either the download's caption
    fetch saw a 429 (``captions_blocked``) or the API lane was refused.
    The block is sticky for the rest of the process so bulk runs fail
    fast after the first one.
    """
    global _block_detected
    if _block_detected:
        raise TranscriptBlocked("caption requests from this IP are rate-limited")

    if caption_files:
        segments = transcript_from_caption_files(
            caption_files, video_id, language, prefer_manual, manual_langs
        )
        if segments:
            return segments

    if captions_blocked:
        _block_detected = True
        raise TranscriptBlocked("YouTube returned 429 (Too Many Requests) for caption downloads")

    pace_caption_request(log=log)
    try:
        segments = get_transcript_api(video_id, language, prefer_manual)
    except TranscriptBlocked:
        _block_detected = True
        raise
    return segments or None


def get_transcript_api(
    video_id: str,
    language: str = "en",
    prefer_manual: bool = True,
) -> list[TranscriptSegment] | None:
    """Fetch transcript via youtube-transcript-api (fallback lane).

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


def save_transcript_json(
    transcript: list[TranscriptSegment],
    path: Path,
) -> None:
    """Save transcript to a JSON file."""
    data = [asdict(segment) for segment in transcript]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
