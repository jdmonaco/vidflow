"""Frame extraction from local video files using ffmpeg."""

import functools
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import imagehash
from PIL import Image

from vidflow.capture.config import DEFAULT_DEDUP_THRESHOLD


@dataclass
class FrameInfo:
    """Information about an extracted frame."""

    path: Path
    timestamp: float


class FrameExtractionError(Exception):
    """Exception raised for frame extraction errors."""

    pass


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available in PATH."""
    return shutil.which("ffmpeg") is not None


@functools.lru_cache(maxsize=1)
def ffmpeg_version() -> tuple[int, int]:
    """Return the installed ffmpeg (major, minor) version, (0, 0) if unknown."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10)
        match = re.search(r"ffmpeg version n?(\d+)\.(\d+)", result.stdout)
        if match:
            return int(match.group(1)), int(match.group(2))
    except Exception:
        pass
    return (0, 0)


def vfr_output_args() -> list[str]:
    """Return the variable-frame-rate output flag for the installed ffmpeg.

    ffmpeg 5.1 replaced -vsync with -fps_mode, and ffmpeg 9 removed -vsync
    outright. Older installs (e.g. Ubuntu 22.04's 4.4) only know -vsync.
    An unparseable version defaults to the modern flag.
    """
    if ffmpeg_version() >= (5, 1) or ffmpeg_version() == (0, 0):
        return ["-fps_mode", "vfr"]
    return ["-vsync", "vfr"]


def compute_phash(image_path: Path) -> imagehash.ImageHash:
    """Compute perceptual hash for an image."""
    with Image.open(image_path) as img:
        return imagehash.phash(img)


def hash_similarity(hash1: imagehash.ImageHash, hash2: imagehash.ImageHash) -> float:
    """Similarity in [0, 1] between two 64-bit perceptual hashes.

    1.0 means identical hashes; each differing bit subtracts 1/64. A frame is
    a duplicate when its similarity to the first frame of the current run is
    >= the dedup threshold (see FrameCollector), so a *lower* threshold
    removes more frames. phash distances are always even (each hash has
    exactly 32 set bits), so thresholds are effectively quantized in steps
    of 2/64.
    """
    distance = hash1 - hash2
    return 1.0 - (distance / 64.0)


class FrameCollector:
    """Move extracted frames into ``output_dir``, collapsing runs of near-duplicates.

    Consecutive frames whose phash similarity to the *first* frame of the
    current run is >= ``threshold`` belong to that run. Each run yields one
    output frame that carries the run's first timestamp (where its caption
    span begins) and its *last* image, so a slide that builds up over
    several samples is represented by its completed state rather than its
    first bullet. With ``threshold`` None every frame is kept.
    """

    def __init__(self, output_dir: Path, frame_format: str, threshold: float | None):
        self.output_dir = output_dir
        self.frame_format = frame_format
        self.threshold = threshold
        self.frames: list[FrameInfo] = []
        self._anchor: imagehash.ImageHash | None = None

    def add(self, source: Path, timestamp: float) -> bool:
        """Consume ``source`` (moved, never copied). True if it started a new run."""
        if self.threshold is not None:
            try:
                current = compute_phash(source)
            except Exception:
                current = None

            if current is not None and self._anchor is not None and self.frames:
                if hash_similarity(self._anchor, current) >= self.threshold:
                    # Same run: newer image replaces the kept file, timestamp stays
                    shutil.move(str(source), str(self.frames[-1].path))
                    return False

            self._anchor = current

        final_path = self.output_dir / f"frame-{len(self.frames):04d}.{self.frame_format}"
        shutil.move(str(source), str(final_path))
        self.frames.append(FrameInfo(path=final_path, timestamp=timestamp))
        return True


def extract_frames_fast(
    video_path: Path,
    output_dir: Path,
    duration: float,
    interval: int = 15,
    max_frames: int | None = None,
    frame_format: str = "jpg",
    dedup_threshold: float | None = DEFAULT_DEDUP_THRESHOLD,
) -> list[FrameInfo]:
    """Extract frames using fast keyframe seeking."""
    if not check_ffmpeg():
        raise FrameExtractionError(
            "ffmpeg not found. Please install ffmpeg:\n"
            "  macOS: brew install ffmpeg\n"
            "  Ubuntu: sudo apt install ffmpeg\n"
            "  Windows: https://ffmpeg.org/download.html"
        )

    if not video_path.exists():
        raise FrameExtractionError(f"Video file not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamps = []
    t = 0.0
    while t < duration:
        timestamps.append(t)
        t += interval
        if max_frames and len(timestamps) >= max_frames:
            break

    collector = FrameCollector(output_dir, frame_format, dedup_threshold)

    for timestamp in timestamps:
        temp_path = output_dir / f"_temp_frame.{frame_format}"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(timestamp),
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(temp_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0 or not temp_path.exists():
                continue
        except subprocess.TimeoutExpired:
            continue
        except Exception:
            continue

        collector.add(temp_path, timestamp)

    temp_path = output_dir / f"_temp_frame.{frame_format}"
    temp_path.unlink(missing_ok=True)

    return collector.frames


def extract_frames_from_file(
    video_path: Path,
    output_dir: Path,
    interval: int = 15,
    max_frames: int | None = None,
    frame_format: str = "jpg",
    dedup_threshold: float | None = DEFAULT_DEDUP_THRESHOLD,
) -> list[FrameInfo]:
    """Extract frames from a local video file with integrated deduplication."""
    if not check_ffmpeg():
        raise FrameExtractionError(
            "ffmpeg not found. Please install ffmpeg:\n"
            "  macOS: brew install ffmpeg\n"
            "  Ubuntu: sudo apt install ffmpeg\n"
            "  Windows: https://ffmpeg.org/download.html"
        )

    if not video_path.exists():
        raise FrameExtractionError(f"Video file not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        temp_pattern = temp_path / f"frame-%04d.{frame_format}"

        cmd = ["ffmpeg", "-y", "-i", str(video_path)]

        vf_parts = [f"fps=1/{interval}"]

        if max_frames and dedup_threshold is None:
            vf_parts.append(f"select='lt(n,{max_frames})'")

        if vf_parts:
            cmd.extend(["-vf", ",".join(vf_parts)])

        cmd.extend(
            [
                *vfr_output_args(),
                "-frame_pts",
                "1",
                str(temp_pattern),
            ]
        )

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                raise FrameExtractionError(
                    f"ffmpeg failed with code {result.returncode}:\n{result.stderr}"
                )

        except subprocess.TimeoutExpired:
            raise FrameExtractionError("Frame extraction timed out (10 minutes)")
        except FileNotFoundError:
            raise FrameExtractionError("ffmpeg not found")
        except FrameExtractionError:
            raise
        except Exception as e:
            raise FrameExtractionError(f"Frame extraction failed: {e}") from e

        temp_frames = sorted(temp_path.glob(f"frame-*.{frame_format}"))

        if not temp_frames:
            raise FrameExtractionError("No frames were extracted from video")

        collector = FrameCollector(output_dir, frame_format, dedup_threshold)

        for i, temp_frame in enumerate(temp_frames):
            if max_frames and len(collector.frames) >= max_frames:
                break
            collector.add(temp_frame, float(i * interval))

    return collector.frames
