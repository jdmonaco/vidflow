"""Frame extraction from local video files using ffmpeg."""

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import imagehash
from PIL import Image


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
    return shutil.which('ffmpeg') is not None


def compute_phash(image_path: Path) -> imagehash.ImageHash:
    """Compute perceptual hash for an image."""
    with Image.open(image_path) as img:
        return imagehash.phash(img)


def hash_similarity(hash1: imagehash.ImageHash, hash2: imagehash.ImageHash) -> float:
    """Compute similarity between two perceptual hashes."""
    distance = hash1 - hash2
    return 1.0 - (distance / 64.0)


def extract_frames_fast(
    video_path: Path,
    output_dir: Path,
    duration: float,
    interval: int = 15,
    max_frames: int | None = None,
    frame_format: str = 'jpg',
    dedup_threshold: float | None = 0.85,
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

    frames: list[FrameInfo] = []
    prev_hash: imagehash.ImageHash | None = None
    frame_index = 0

    for timestamp in timestamps:
        temp_path = output_dir / f'_temp_frame.{frame_format}'
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(timestamp),
            '-i', str(video_path),
            '-frames:v', '1',
            '-q:v', '2',
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

        if dedup_threshold is not None:
            try:
                current_hash = compute_phash(temp_path)
            except Exception:
                current_hash = None

            if current_hash is not None and prev_hash is not None:
                similarity = hash_similarity(prev_hash, current_hash)
                if similarity >= dedup_threshold:
                    temp_path.unlink(missing_ok=True)
                    continue

            prev_hash = current_hash

        final_name = f'frame-{frame_index:04d}.{frame_format}'
        final_path = output_dir / final_name
        shutil.move(str(temp_path), str(final_path))

        frames.append(FrameInfo(path=final_path, timestamp=timestamp))
        frame_index += 1

    temp_path = output_dir / f'_temp_frame.{frame_format}'
    temp_path.unlink(missing_ok=True)

    return frames


def extract_frames_from_file(
    video_path: Path,
    output_dir: Path,
    interval: int = 15,
    max_frames: int | None = None,
    frame_format: str = 'jpg',
    dedup_threshold: float | None = 0.85,
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
        temp_pattern = temp_path / f'frame-%04d.{frame_format}'

        cmd = ['ffmpeg', '-y', '-i', str(video_path)]

        vf_parts = [f'fps=1/{interval}']

        if max_frames and dedup_threshold is None:
            vf_parts.append(f"select='lt(n,{max_frames})'")

        if vf_parts:
            cmd.extend(['-vf', ','.join(vf_parts)])

        cmd.extend([
            '-vsync', 'vfr',
            '-frame_pts', '1',
            str(temp_pattern),
        ])

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

        temp_frames = sorted(temp_path.glob(f'frame-*.{frame_format}'))

        if not temp_frames:
            raise FrameExtractionError("No frames were extracted from video")

        frames: list[FrameInfo] = []
        prev_hash: imagehash.ImageHash | None = None
        frame_index = 0

        for i, temp_frame in enumerate(temp_frames):
            timestamp = float(i * interval)

            if max_frames and len(frames) >= max_frames:
                break

            if dedup_threshold is not None:
                try:
                    current_hash = compute_phash(temp_frame)
                except Exception:
                    current_hash = None

                if current_hash is not None and prev_hash is not None:
                    similarity = hash_similarity(prev_hash, current_hash)
                    if similarity >= dedup_threshold:
                        continue

                prev_hash = current_hash

            final_name = f'frame-{frame_index:04d}.{frame_format}'
            final_path = output_dir / final_name
            shutil.move(str(temp_frame), str(final_path))

            frames.append(FrameInfo(path=final_path, timestamp=timestamp))
            frame_index += 1

    return frames
