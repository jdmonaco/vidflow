"""Video capture module for vidflow (formerly ytcapture).

Provides YouTube and local video frame extraction into
Obsidian-compatible markdown notes.
"""

from pathlib import Path
from typing import Optional

__all__ = [
    "process_video",
    "process_local_video",
    "capture_youtube",
    "capture_local",
]


def process_video(*args, **kwargs):
    """Capture frames from a YouTube video. See capture.core for details."""
    from vidflow.capture.core import process_video as _process_video

    return _process_video(*args, **kwargs)


def process_local_video(*args, **kwargs):
    """Capture frames from a local video. See capture.core for details."""
    from vidflow.capture.core import process_local_video as _process_local_video

    return _process_local_video(*args, **kwargs)


def capture_youtube(
    url: str,
    output_dir: Path,
    interval: int = 15,
    max_frames: Optional[int] = None,
    frame_format: str = "jpg",
    language: str = "en",
    prefer_manual: bool = False,
    dedup_threshold: float = 0.95,
    no_dedup: bool = False,
    keep_video: bool = False,
    no_ai_title: bool = False,
):
    """Capture YouTube video with OperationResult output.

    Wraps process_video() for use by the vidflow CLI layer.
    """
    from vidflow.cli_common import OperationResult

    try:
        md_path = process_video(
            url=url,
            output_dir=output_dir,
            interval=interval,
            max_frames=max_frames,
            frame_format=frame_format,
            language=language,
            prefer_manual=prefer_manual,
            dedup_threshold=dedup_threshold,
            no_dedup=no_dedup,
            keep_video=keep_video,
            no_ai_title=no_ai_title,
        )
        return OperationResult(
            success=True,
            message=f"Captured YouTube video to {md_path}",
            data={"output_path": str(md_path)},
        )
    except Exception as e:
        return OperationResult(
            success=False,
            message=f"YouTube capture failed: {e}",
            errors=[str(e)],
        )


def capture_local(
    video_path: Path,
    output_dir: Path,
    interval: int = 15,
    max_frames: Optional[int] = None,
    frame_format: str = "jpg",
    dedup_threshold: float = 0.95,
    no_dedup: bool = False,
    fast: bool = False,
    force: bool = False,
    json_output: bool = False,
):
    """Capture local video with OperationResult output.

    Wraps process_local_video() for use by the vidflow CLI layer.
    """
    from vidflow.cli_common import OperationResult

    try:
        result = process_local_video(
            video_path=video_path,
            output_dir=output_dir,
            interval=interval,
            max_frames=max_frames,
            frame_format=frame_format,
            dedup_threshold=dedup_threshold,
            no_dedup=no_dedup,
            fast=fast,
            json_output=json_output,
            force=force,
        )
        if isinstance(result, dict):
            md_path = result.get("output_file", str(video_path))
            return OperationResult(
                success=result.get("status") != "error",
                message=result.get("message", f"Captured local video to {md_path}"),
                data=result,
            )
        else:
            return OperationResult(
                success=True,
                message=f"Captured local video to {result}",
                data={"output_path": str(result)},
            )
    except Exception as e:
        return OperationResult(
            success=False,
            message=f"Local capture failed: {e}",
            errors=[str(e)],
        )
