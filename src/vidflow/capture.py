"""Wrappers around ytcapture capture APIs.

Provides OperationResult-wrapped interfaces to ytcapture's
process_video() and process_local_video() functions.
"""

from pathlib import Path
from typing import Optional

from .cli_common import OperationResult


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
) -> OperationResult:
    """Capture frames from a YouTube video.

    Wraps ytcapture.cli.process_video() with OperationResult output.

    Returns:
        OperationResult with data.output_path set to the generated markdown file.
    """
    from ytcapture.cli import process_video

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
) -> OperationResult:
    """Capture frames from a local video file.

    Wraps ytcapture.cli.process_local_video() with OperationResult output.

    Returns:
        OperationResult with data.output_path set to the generated markdown file.
    """
    from ytcapture.cli import process_local_video

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
            # JSON mode returns a dict
            md_path = result.get("output_file", str(video_path))
            return OperationResult(
                success=result.get("status") != "error",
                message=result.get("message", f"Captured local video to {md_path}"),
                data=result,
            )
        else:
            # Normal mode returns a Path
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
