"""Core capture functions — library API for ytcapture and vidcapture.

These functions are Click-free and can be called programmatically.
"""

import shutil
import subprocess
from pathlib import Path

from rich.console import Console

from vidflow.capture.config import resolve_output_path
from vidflow.capture.frames import FrameExtractionError, extract_frames_fast, extract_frames_from_file
from vidflow.capture.local import LocalVideoError, LocalVideoMetadata, get_local_video_metadata
from vidflow.capture.markdown import (
    generate_local_markdown_filename,
    generate_markdown_file,
    generate_markdown_filename,
)
from vidflow.capture.transcript import TranscriptSegment, get_transcript, save_transcript_json
from vidflow.capture.utils import format_timestamp
from vidflow.capture.video import (
    VideoError,
    VideoMetadata,
    download_video,
    get_video_metadata,
)

console = Console()


def format_markdown(filepath: Path) -> bool:
    """Format markdown file with mdformat if available."""
    if shutil.which('mdformat') is None:
        return False
    try:
        subprocess.run(
            ['mdformat', '--wrap', 'no', filepath],
            capture_output=True,
            timeout=30,
        )
        return True
    except Exception:
        return False


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / 1024 / 1024:.1f} MB"


def shorten_path(path: str) -> str:
    """Shorten paths for display: $HOME -> ~, OneDrive CloudStorage -> ~/OneDrive."""
    home = str(Path.home())
    onedrive_prefix = f"{home}/Library/CloudStorage/OneDrive-"
    if path.startswith(onedrive_prefix):
        rest = path[len(onedrive_prefix):]
        if "/" in rest:
            _, subpath = rest.split("/", 1)
            return f"~/OneDrive/{subpath}"
        return f"~/OneDrive/{rest}"
    if path.startswith(home + "/"):
        return "~" + path[len(home):]
    elif path == home:
        return "~"
    return path


def process_video(
    url: str,
    output_dir: Path,
    interval: int,
    max_frames: int | None,
    frame_format: str,
    language: str,
    prefer_manual: bool,
    dedup_threshold: float,
    no_dedup: bool,
    keep_video: bool,
    no_ai_title: bool = False,
) -> Path:
    """Process a single YouTube video URL.

    Returns:
        Path to the generated markdown file.

    Raises:
        VideoError: If video processing fails.
        FrameExtractionError: If frame extraction fails.
    """
    # 1. Get video metadata
    with console.status("[bold blue]Fetching video metadata...", spinner="dots"):
        metadata: VideoMetadata = get_video_metadata(url)

    console.print("[green]+[/] Fetched video metadata")
    console.print(f"  [dim]Title:[/] {metadata.title}")
    console.print(f"  [dim]Channel:[/] {metadata.channel}")

    # 1b. AI title generation
    if not no_ai_title:
        from vidflow.capture.titling import generate_ai_title, is_ai_titling_available

        if is_ai_titling_available():
            with console.status("[bold blue]Generating AI title...", spinner="dots"):
                result = generate_ai_title(
                    title=metadata.title,
                    channel=metadata.channel,
                    description=metadata.description,
                )
            if result.used_ai:
                metadata._original_title = result.original_title
                metadata.title = result.ai_title
                console.print(f"[green]+[/] AI title: {metadata.title}")

    # 2. Create directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / 'images' / metadata.video_id
    frames_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir = output_dir / 'transcripts'
    transcripts_dir.mkdir(exist_ok=True)
    videos_dir = output_dir / 'videos'
    videos_dir.mkdir(exist_ok=True)

    # 3. Get transcript
    with console.status("[bold blue]Fetching transcript...", spinner="dots"):
        transcript: list[TranscriptSegment] | None = get_transcript(
            metadata.video_id,
            language=language,
            prefer_manual=prefer_manual,
        )

    if transcript:
        console.print(f"[green]+[/] Found {len(transcript)} transcript segments")
        save_transcript_json(
            transcript,
            transcripts_dir / f'raw-transcript-{metadata.video_id}.json',
        )
    else:
        console.print("[yellow]![/] No transcript available, proceeding with frames only")

    # 4. Download video
    with console.status("[bold blue]Downloading video...", spinner="dots"):
        video_path = download_video(url, videos_dir)

    video_size = format_size(video_path.stat().st_size)
    console.print(f"[green]+[/] Downloaded video ({video_size})")

    # 5. Extract frames (with integrated dedup)
    with console.status("[bold blue]Extracting frames...", spinner="dots"):
        frames = extract_frames_from_file(
            video_path,
            frames_dir,
            interval=interval,
            max_frames=max_frames,
            frame_format=frame_format,
            dedup_threshold=None if no_dedup else dedup_threshold,
        )

    dedup_msg = "" if no_dedup else " (deduplicated)"
    console.print(f"[green]+[/] Extracted {len(frames)} frames{dedup_msg}")

    # 6. Handle video file (keep or delete)
    final_video_path: Path | None = None
    if keep_video:
        md_filename = generate_markdown_filename(metadata)
        md_basename = md_filename.rsplit('.', 1)[0]
        video_ext = video_path.suffix
        final_video_path = videos_dir / f'{md_basename}{video_ext}'

        if video_path != final_video_path:
            video_path.rename(final_video_path)

        console.print(f"  [dim]Video saved:[/] {final_video_path}")
    else:
        try:
            video_path.unlink()
            videos_dir.rmdir()
        except Exception:
            pass

    # 7. Generate markdown
    with console.status("[bold blue]Generating markdown...", spinner="dots"):
        md_file = generate_markdown_file(
            metadata,
            url,
            transcript,
            frames,
            output_dir,
            video_path=final_video_path,
        )

    console.print("[green]+[/] Generated markdown")

    # 8. Format markdown (if mdformat available)
    if format_markdown(md_file):
        console.print("  [dim]Formatted with mdformat[/]")

    return md_file


def process_local_video(
    video_path: Path,
    output_dir: Path,
    interval: int,
    max_frames: int | None,
    frame_format: str,
    dedup_threshold: float,
    no_dedup: bool,
    fast: bool = False,
    json_output: bool = False,
    force: bool = False,
) -> dict | Path:
    """Process a single local video file.

    Returns:
        Path to the generated markdown file, or dict if json_output.
    """
    out_console = Console(quiet=True) if json_output else console

    # 1. Get video metadata
    with out_console.status("[bold blue]Extracting video metadata...", spinner="dots"):
        metadata: LocalVideoMetadata = get_local_video_metadata(video_path)

    out_console.print("[green]+[/] Extracted video metadata")
    out_console.print(f"  [dim]Title:[/] {metadata.title}")
    out_console.print(f"  [dim]Duration:[/] {metadata.duration:.1f}s")

    # Check if output file already exists
    md_filename = generate_local_markdown_filename(metadata)
    md_filepath = output_dir / md_filename
    if md_filepath.exists() and not force:
        if json_output:
            return {
                "status": "error",
                "video": str(video_path.resolve()),
                "error": f"Output file exists: {md_filename}. Use -f/--force to overwrite.",
            }
        response = input(f"Output file exists: {md_filename}\nOverwrite? [y/N]: ").strip().lower()
        if response not in ["y", "yes"]:
            raise LocalVideoError("Output file exists. Use -f/--force to overwrite.")

    # 2. Create directory structure (with collision handling)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / 'images' / metadata.identifier
    if frames_dir.exists():
        suffix = 2
        while (output_dir / 'images' / f"{metadata.file_path.stem}-{suffix}").exists():
            suffix += 1
        metadata._identifier_suffix = suffix
        frames_dir = output_dir / 'images' / metadata.identifier
        out_console.print(f"  [dim]Using identifier:[/] {metadata.identifier} (collision avoided)")
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 3. Extract frames (with integrated dedup)
    extraction_mode = "fast seek" if fast else "full decode"
    with out_console.status(f"[bold blue]Extracting frames ({extraction_mode})...", spinner="dots"):
        if fast:
            frames = extract_frames_fast(
                video_path,
                frames_dir,
                duration=metadata.duration,
                interval=interval,
                max_frames=max_frames,
                frame_format=frame_format,
                dedup_threshold=None if no_dedup else dedup_threshold,
            )
        else:
            frames = extract_frames_from_file(
                video_path,
                frames_dir,
                interval=interval,
                max_frames=max_frames,
                frame_format=frame_format,
                dedup_threshold=None if no_dedup else dedup_threshold,
            )

    dedup_msg = "" if no_dedup else " (deduplicated)"
    out_console.print(f"[green]+[/] Extracted {len(frames)} frames{dedup_msg}")

    # 4. Generate markdown (no transcript for local videos)
    with out_console.status("[bold blue]Generating markdown...", spinner="dots"):
        md_file = generate_markdown_file(
            metadata,
            url=None,
            transcript=None,
            frames=frames,
            output_dir=output_dir,
            video_path=None,
            filename=md_filename,
        )

    out_console.print("[green]+[/] Generated markdown")

    # 5. Format markdown (if mdformat available)
    if format_markdown(md_file):
        out_console.print("  [dim]Formatted with mdformat[/]")

    if json_output:
        return {
            "status": "success",
            "video": str(video_path.resolve()),
            "frames_dir": str(frames_dir.resolve()),
            "frame_count": len(frames),
            "markdown": str(md_file.resolve()),
        }
    return md_file
