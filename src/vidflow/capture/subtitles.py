"""Subtitle discovery and extraction for local video files.

A sidecar WebVTT file next to the video (``<stem>.vtt``, ``<stem>.en.vtt``,
``<stem>-en-US.vtt`` — the last is how Microsoft Teams/Stream name a
downloaded transcript) is preferred over embedded tracks. WebVTT voice tags
(``<v Speaker Name>``) carry speaker diarization into
``TranscriptSegment.speaker``.

For embedded tracks, detects text-based subtitle streams (mov_text, subrip, webvtt, ass/ssa) via
ffprobe, extracts the selected track via ffmpeg as WebVTT, parses it into
TranscriptSegment objects, and sanitizes cue text (strips HTML-like tags,
inline timestamp tags, ASS override codes, decodes HTML entities).

Bitmap codecs (hdmv_pgs_subtitle, dvd_subtitle) and CEA-608/708 closed
captions embedded in video streams are not supported in this phase.
"""

import html
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from vidflow.capture.transcript import TranscriptSegment

# Text-based subtitle codecs we can extract to WebVTT
TEXT_SUB_CODECS = frozenset(
    {
        "mov_text",  # MP4/MOV
        "subrip",  # SRT in Matroska
        "srt",
        "webvtt",
        "ass",  # Advanced SubStation Alpha
        "ssa",  # SubStation Alpha
        "text",  # generic text
    }
)

DEFAULT_LANGUAGE = "en"


class SubtitleError(Exception):
    """Exception raised for subtitle extraction errors."""

    pass


@dataclass
class SubtitleStream:
    """Descriptor for an embedded subtitle stream."""

    index: int  # Absolute stream index in the file
    subtitle_index: int  # Index among subtitle streams only (for -map 0:s:N)
    codec: str
    language: str | None
    title: str | None
    is_default: bool
    is_forced: bool
    is_hearing_impaired: bool

    @property
    def is_text_based(self) -> bool:
        return self.codec in TEXT_SUB_CODECS

    def describe(self) -> str:
        parts = [f"#{self.subtitle_index}", self.codec]
        if self.language:
            parts.append(self.language)
        if self.title:
            parts.append(f'"{self.title}"')
        flags = []
        if self.is_default:
            flags.append("default")
        if self.is_forced:
            flags.append("forced")
        if self.is_hearing_impaired:
            flags.append("hearing-impaired")
        if flags:
            parts.append(f"[{','.join(flags)}]")
        return " ".join(parts)


def probe_subtitle_streams(video_path: Path) -> list[SubtitleStream]:
    """Enumerate embedded subtitle streams via ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-select_streams",
        "s",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired as e:
        raise SubtitleError("ffprobe timed out while inspecting subtitle streams") from e
    except FileNotFoundError as e:
        raise SubtitleError("ffprobe not found") from e

    if result.returncode != 0:
        raise SubtitleError(f"ffprobe failed: {result.stderr.strip()}")

    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as e:
        raise SubtitleError(f"Failed to parse ffprobe output: {e}") from e

    streams: list[SubtitleStream] = []
    for sub_idx, stream in enumerate(data.get("streams", [])):
        tags = stream.get("tags") or {}
        disposition = stream.get("disposition") or {}
        streams.append(
            SubtitleStream(
                index=stream.get("index", -1),
                subtitle_index=sub_idx,
                codec=stream.get("codec_name", "unknown"),
                language=tags.get("language"),
                title=tags.get("title"),
                is_default=bool(disposition.get("default")),
                is_forced=bool(disposition.get("forced")),
                is_hearing_impaired=bool(disposition.get("hearing_impaired")),
            )
        )
    return streams


def _matches_language(stream: SubtitleStream, language: str) -> bool:
    if not stream.language:
        return False
    return stream.language.lower().startswith(language.lower())


def select_subtitle_stream(
    streams: list[SubtitleStream],
    *,
    track: int | None = None,
    language: str = DEFAULT_LANGUAGE,
) -> SubtitleStream | None:
    """Pick a subtitle stream per policy.

    Policy (when `track` is None):
      1. Default-flagged track matching `language`
      2. Any track matching `language` (prefer non-forced)
      3. Default-flagged track of any language
      4. First text-based track
      5. None
    """
    if not streams:
        return None

    if track is not None:
        matching = [s for s in streams if s.subtitle_index == track]
        if not matching:
            raise SubtitleError(
                f"Subtitle track #{track} not found " f"(available: 0..{len(streams) - 1})"
            )
        return matching[0]

    text_streams = [s for s in streams if s.is_text_based]
    if not text_streams:
        return None

    lang_matches = [s for s in text_streams if _matches_language(s, language)]

    default_lang = [s for s in lang_matches if s.is_default and not s.is_forced]
    if default_lang:
        return default_lang[0]

    non_forced_lang = [s for s in lang_matches if not s.is_forced]
    if non_forced_lang:
        return non_forced_lang[0]

    if lang_matches:
        return lang_matches[0]

    default_any = [s for s in text_streams if s.is_default and not s.is_forced]
    if default_any:
        return default_any[0]

    return text_streams[0]


# Opening WebVTT voice tag: <v Name> or <v.class1.class2 Name>; the
# annotation (everything up to ">") is the speaker name
_VOICE_RE = re.compile(r"<v(?:\.[^\s>]*)?\s+([^>]+)>")

# Matches HTML/XML-style tags (<b>, <font color="...">, <c.classname>, <v Speaker>, etc.)
_TAG_RE = re.compile(r"<[^>]+>")
# Matches inline WebVTT timestamp tags like <00:00:05.000>
_TS_TAG_RE = re.compile(r"<\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?>")
# Matches ASS override blocks like {\an8} or {\pos(100,200)}
_ASS_OVERRIDE_RE = re.compile(r"\{[^}]*\}")
# Collapses internal whitespace runs (including newlines) to single spaces
_WS_RE = re.compile(r"\s+")


def sanitize_cue_text(text: str) -> str:
    """Strip markup from a subtitle cue while preserving speaker labels."""
    text = _TS_TAG_RE.sub("", text)
    text = _ASS_OVERRIDE_RE.sub("", text)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    text = _WS_RE.sub(" ", text).strip()
    return text


# WebVTT cue timing line: "00:00:05.000 --> 00:00:08.000 ..."
_VTT_TIMING_RE = re.compile(
    r"^(?P<start>\d{1,2}:\d{2}:\d{2}\.\d{3}|\d{1,2}:\d{2}\.\d{3})"
    r"\s+-->\s+"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}\.\d{3}|\d{1,2}:\d{2}\.\d{3})"
    r"(?:\s.*)?$"
)


def _parse_vtt_timestamp(ts: str) -> float:
    """Parse HH:MM:SS.mmm or MM:SS.mmm to seconds."""
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, rest = parts
        return int(h) * 3600 + int(m) * 60 + float(rest)
    if len(parts) == 2:
        m, rest = parts
        return int(m) * 60 + float(rest)
    raise ValueError(f"Bad timestamp: {ts}")


def parse_webvtt(vtt_text: str) -> list[TranscriptSegment]:
    """Parse a WebVTT document into TranscriptSegment objects."""
    segments: list[TranscriptSegment] = []
    lines = vtt_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        match = _VTT_TIMING_RE.match(line.strip())
        if not match:
            i += 1
            continue

        try:
            start = _parse_vtt_timestamp(match.group("start"))
            end = _parse_vtt_timestamp(match.group("end"))
        except ValueError:
            i += 1
            continue

        i += 1
        cue_lines: list[str] = []
        while i < n and lines[i].strip() != "":
            cue_lines.append(lines[i])
            i += 1

        for speaker, raw in split_voice_spans("\n".join(cue_lines)):
            text = sanitize_cue_text(raw)
            if text:
                segments.append(
                    TranscriptSegment(
                        text=text,
                        start=start,
                        duration=max(0.0, end - start),
                        speaker=speaker,
                    )
                )
        i += 1

    return segments


def split_voice_spans(raw: str) -> list[tuple[str | None, str]]:
    """Split raw cue text into (speaker, text) spans at each WebVTT voice tag.

    A cue normally carries at most one voice tag (Teams writes one per cue),
    but the spec allows several; each starts a new span. Text before the
    first tag, or a cue without tags, has no speaker.
    """
    matches = list(_VOICE_RE.finditer(raw))
    if not matches:
        return [(None, raw)]
    spans: list[tuple[str | None, str]] = []
    if raw[: matches[0].start()].strip():
        spans.append((None, raw[: matches[0].start()]))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        speaker = _WS_RE.sub(" ", html.unescape(m.group(1))).strip() or None
        spans.append((speaker, raw[m.end() : end]))
    return spans


# Language suffix on a sidecar name: "en", "en-US", "en_GB". Restricted to
# two-letter primaries so arbitrary suffixes ("talk-old.vtt") don't match.
_SIDECAR_LANG_RE = re.compile(r"^[a-z]{2}(?:[-_][A-Z]{2})?$")


def find_sidecar_vtt(video_path: Path, language: str = DEFAULT_LANGUAGE) -> Path | None:
    """Locate a sidecar WebVTT transcript next to a video file.

    Candidates, best first: ``<stem>.vtt``; ``<stem>.<lang>.vtt`` or
    ``<stem>-<lang>.vtt`` whose language matches ``language``; the same
    with any other language. Ties break by name for determinism.
    """
    stem = video_path.stem
    ranked: list[tuple[int, str, Path]] = []
    try:
        entries = list(video_path.parent.iterdir())
    except OSError:
        return None
    for p in entries:
        if p.suffix.lower() != ".vtt" or p.name.startswith(".") or not p.is_file():
            continue
        name = p.name[: -len(p.suffix)]
        if name == stem:
            rank = 0
        elif len(name) > len(stem) + 1 and name.startswith(stem) and name[len(stem)] in ".-":
            tag = name[len(stem) + 1 :]
            if not _SIDECAR_LANG_RE.match(tag):
                continue
            rank = 1 if tag.lower().startswith(language.lower()) else 2
        else:
            continue
        ranked.append((rank, p.name, p))
    if not ranked:
        return None
    return min(ranked)[2]


def load_sidecar_vtt(path: Path) -> list[TranscriptSegment]:
    """Read and parse a sidecar WebVTT file (BOM and CRLF tolerant)."""
    try:
        vtt_text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as e:
        raise SubtitleError(f"Cannot read sidecar {path.name}: {e}") from e
    return parse_webvtt(vtt_text)


def list_speakers(segments: list[TranscriptSegment]) -> list[str]:
    """Distinct speaker names in order of first appearance."""
    return list(dict.fromkeys(s.speaker for s in segments if s.speaker))


def extract_subtitle_track(
    video_path: Path,
    stream: SubtitleStream,
) -> list[TranscriptSegment]:
    """Extract the selected subtitle track and parse it into segments."""
    if not stream.is_text_based:
        raise SubtitleError(
            f"Subtitle codec '{stream.codec}' is not text-based; "
            "bitmap and CEA-608/708 streams are not supported"
        )

    with tempfile.NamedTemporaryFile(suffix=".vtt", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-map",
            f"0:s:{stream.subtitle_index}",
            "-c:s",
            "webvtt",
            str(tmp_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise SubtitleError(f"ffmpeg subtitle extraction failed: {result.stderr.strip()}")

        vtt_text = tmp_path.read_text(encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired as e:
        raise SubtitleError("ffmpeg timed out during subtitle extraction") from e
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass

    return parse_webvtt(vtt_text)
