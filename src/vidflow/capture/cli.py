"""Standalone CLI entry points for ytcapture and vidcapture (argparse-based).

Provides backward-compatible `ytcapture` and `vidcapture` commands.
"""

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from vidflow import __version__
from vidflow.capture.config import (
    config_was_auto_created,
    get_config_for_defaults,
    get_config_path,
    resolve_output_path,
)
from vidflow.capture.core import (
    format_markdown,
    process_local_video,
    process_video,
    shorten_path,
)
from vidflow.capture.frames import FrameExtractionError
from vidflow.capture.local import LocalVideoError
from vidflow.capture.utils import (
    extract_youtube_urls,
    is_playlist_url,
    is_video_id,
    is_video_url,
    video_id_to_url,
)
from vidflow.capture.video import VideoError, expand_playlist, get_video_metadata

# Load config at module level for CLI option defaults
_cfg = get_config_for_defaults()

console = Console()


def get_clipboard_urls() -> list[str]:
    """Check clipboard for YouTube URLs (macOS only)."""
    if platform.system() != 'Darwin':
        return []
    if shutil.which('pbpaste') is None:
        return []
    try:
        result = subprocess.run(
            ['pbpaste'], capture_output=True, text=True, timeout=5,
        )
        clipboard = result.stdout.strip()
        if not clipboard:
            return []
        return extract_youtube_urls(clipboard)
    except Exception:
        return []


def preview_urls(video_urls: list[str], con: Console, source: str = "clipboard") -> bool:
    """Show a preview table of video URLs and confirm."""
    from vidflow.capture.utils import format_timestamp

    title = "Clipboard URLs" if source == "clipboard" else "Input URLs"
    table = Table(title=title)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Title", style="bold")
    table.add_column("Channel")
    table.add_column("Duration", justify="right")

    with con.status("[bold blue]Fetching video metadata...", spinner="dots"):
        for i, url in enumerate(video_urls, 1):
            try:
                metadata = get_video_metadata(url)
                table.add_row(
                    str(i), metadata.title, metadata.channel,
                    format_timestamp(metadata.duration),
                )
            except VideoError:
                table.add_row(str(i), "(metadata unavailable)", url, "")

    con.print()
    con.print(table)
    con.print()

    if source == "clipboard":
        response = input("Proceed with capture? [Y/n]: ").strip().lower()
        return response in ["y", "yes", ""]
    return True


def ytcapture_main(argv=None):
    """Main entry point for ytcapture command."""
    parser = argparse.ArgumentParser(
        prog="ytcapture",
        description="Extract frames and transcript from YouTube videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  ytcapture "https://www.youtube.com/watch?v=VIDEO_ID"
  ytcapture dQw4w9WgXcQ
  ytcapture URL1 URL2 URL3
  ytcapture "https://www.youtube.com/playlist?list=PLAYLIST_ID"
""",
    )
    parser.add_argument("urls", nargs="*", help="YouTube video URL(s) or video ID(s)")
    parser.add_argument(
        "-o", "--output", type=str,
        help="Output directory (relative to cwd or absolute path)",
    )
    parser.add_argument(
        "--interval", type=int, default=_cfg.get("interval", 15),
        help=f"Frame extraction interval in seconds (default: {_cfg.get('interval', 15)})",
    )
    parser.add_argument(
        "--max-frames", type=int, default=_cfg.get("max_frames"),
        help="Maximum number of frames to extract",
    )
    parser.add_argument(
        "--frame-format", choices=["jpg", "png"],
        default=_cfg.get("frame_format", "jpg"),
        help=f"Frame image format (default: {_cfg.get('frame_format', 'jpg')})",
    )
    parser.add_argument(
        "--language", default=_cfg.get("language", "en"),
        help=f"Transcript language code (default: {_cfg.get('language', 'en')})",
    )
    parser.add_argument(
        "--prefer-manual", action="store_true",
        default=_cfg.get("prefer_manual", False),
        help="Only use manual transcripts",
    )
    parser.add_argument(
        "--dedup-threshold", type=float,
        default=_cfg.get("dedup_threshold", 0.85),
        help=f"Similarity threshold for frame deduplication (default: {_cfg.get('dedup_threshold', 0.85)})",
    )
    parser.add_argument("--no-dedup", action="store_true", help="Disable frame deduplication")
    parser.add_argument(
        "--keep-video", action="store_true",
        default=_cfg.get("keep_video", False),
        help="Keep downloaded video file after frame extraction",
    )
    parser.add_argument(
        "--no-ai-title", action="store_true",
        default=not _cfg.get("ai_title", True),
        help="Disable AI title generation",
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompts")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--version", action="version", version=f"ytcapture (vidflow {__version__})")

    args = parser.parse_args(argv)

    # Show message if config was auto-created on this run
    if config_was_auto_created():
        console.print(f"[dim]Created config:[/] {get_config_path()}")

    # 1. Collect URLs from arguments and/or clipboard
    url_list = list(args.urls) if args.urls else []
    from_clipboard = False
    if not url_list:
        clipboard_urls = get_clipboard_urls()
        if clipboard_urls:
            from_clipboard = True
            url_list = clipboard_urls
            if len(url_list) == 1:
                console.print(f"[dim]Using URL from clipboard:[/] {url_list[0]}")
            else:
                console.print(f"[dim]Found {len(url_list)} URLs from clipboard[/]")
        else:
            parser.error(
                "No URLs provided. Pass YouTube URLs as arguments or copy one to clipboard."
            )

    # 2. Determine output directory
    if args.output:
        output_dir = resolve_output_path(args.output)
    else:
        output_dir = Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[dim]Output directory:[/] {shorten_path(str(output_dir))}/")

    # 3. Classify and expand URLs (also accepts bare video IDs)
    video_urls: list[str] = []
    for url in url_list:
        if is_video_id(url):
            full_url = video_id_to_url(url)
            console.print(f"[dim]Video ID:[/] {url} -> {full_url}")
            video_urls.append(full_url)
        elif is_playlist_url(url):
            console.print(f"\n[dim]Expanding playlist:[/] {url}")
            with console.status("[bold blue]Fetching playlist...", spinner="dots"):
                try:
                    playlist_videos = expand_playlist(url)
                except VideoError as e:
                    console.print(f"[red]x[/] Failed to expand playlist: {e}")
                    continue
            console.print(f"[green]+[/] Found {len(playlist_videos)} videos in playlist")
            video_urls.extend(playlist_videos)
        elif is_video_url(url):
            video_urls.append(url)
        else:
            console.print(f"[yellow]![/] Skipping invalid URL: {url}")

    if not video_urls:
        parser.error("No valid video URLs found.")

    # 4. Deduplicate video URLs
    seen: set[str] = set()
    unique_urls: list[str] = []
    for url in video_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    video_urls = unique_urls

    # 5. Preview URLs
    if len(video_urls) > 1 and not args.yes:
        if not preview_urls(video_urls, console, source="clipboard" if from_clipboard else "args"):
            parser.error("Cancelled by user.")

    # 6. Confirm if >10 videos
    if len(video_urls) > 10 and not args.yes:
        console.print(f"\n[bold]Found {len(video_urls)} videos to process.[/]")
        response = input("Continue? [Y/n]: ").strip().lower()
        if response not in ["y", "yes", ""]:
            parser.error("Cancelled by user.")

    # 7. Process each video
    console.print(f"\n[bold]Processing {len(video_urls)} video(s)...[/]\n")

    success_count = 0
    error_count = 0

    for i, video_url in enumerate(video_urls, 1):
        console.print(f"[bold blue][{i}/{len(video_urls)}][/] {video_url}")
        try:
            md_file = process_video(
                video_url, output_dir, args.interval, args.max_frames,
                args.frame_format, args.language, args.prefer_manual,
                args.dedup_threshold, args.no_dedup, args.keep_video,
                args.no_ai_title,
            )
            console.print(f"[green]+[/] {md_file.name}")
            success_count += 1
        except (VideoError, FrameExtractionError) as e:
            console.print(f"[red]x[/] Failed: {e}")
            error_count += 1

    # 8. Summary
    if error_count > 0:
        console.print(f"\n[bold yellow]Complete![/] {success_count} succeeded, {error_count} failed")
    else:
        console.print(f"\n[bold green]Complete![/] {success_count} video(s) processed")

    return 0


def vidcapture_main(argv=None):
    """Main entry point for vidcapture command."""
    parser = argparse.ArgumentParser(
        prog="vidcapture",
        description="Extract frames from local video files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  vidcapture meeting.mp4
  vidcapture video1.mp4 video2.mkv -o notes/
  vidcapture recording.mov --interval 30 --max-frames 50
  vidcapture long-workshop.mp4 --no-fast --interval 60
""",
    )
    parser.add_argument("files", nargs="*", help="Local video file(s)")
    parser.add_argument(
        "-o", "--output", type=str,
        help="Output directory (relative to cwd or absolute path)",
    )
    parser.add_argument(
        "--interval", type=int, default=_cfg.get("interval", 15),
        help=f"Frame extraction interval in seconds (default: {_cfg.get('interval', 15)})",
    )
    parser.add_argument(
        "--max-frames", type=int, default=_cfg.get("max_frames"),
        help="Maximum number of frames to extract",
    )
    parser.add_argument(
        "--frame-format", choices=["jpg", "png"],
        default=_cfg.get("frame_format", "jpg"),
        help=f"Frame image format (default: {_cfg.get('frame_format', 'jpg')})",
    )
    parser.add_argument(
        "--dedup-threshold", type=float,
        default=_cfg.get("dedup_threshold", 0.85),
        help=f"Similarity threshold for frame deduplication (default: {_cfg.get('dedup_threshold', 0.85)})",
    )
    parser.add_argument("--no-dedup", action="store_true", help="Disable frame deduplication")
    parser.add_argument(
        "--fast", action="store_true",
        default=_cfg.get("fast", True),
        help="Use fast keyframe seeking (default)",
    )
    parser.add_argument("--no-fast", action="store_true", help="Disable fast keyframe seeking")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--json", dest="json_output", action="store_true",
        help="Output JSON instead of rich console output",
    )
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--version", action="version", version=f"vidcapture (vidflow {__version__})")

    args = parser.parse_args(argv)

    out_console = Console(quiet=True) if args.json_output else console

    if config_was_auto_created() and not args.json_output:
        out_console.print(f"[dim]Created config:[/] {get_config_path()}")

    if not args.files:
        if args.json_output:
            print(json.dumps({"status": "error", "error": "No video files provided"}))
            return 1
        parser.error("No video files provided. Pass paths to video files as arguments.")

    # Determine output directory
    if args.output:
        output_dir = resolve_output_path(args.output)
    else:
        output_dir = Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)

    out_console.print(f"[dim]Output directory:[/] {shorten_path(str(output_dir))}/")

    # Resolve fast flag
    fast = args.fast and not args.no_fast

    # Process each video file
    out_console.print(f"\n[bold]Processing {len(args.files)} video file(s)...[/]\n")

    success_count = 0
    error_count = 0
    results: list[dict] = []

    for i, file_path in enumerate(args.files, 1):
        video_path = Path(file_path)
        out_console.print(f"[bold blue][{i}/{len(args.files)}][/] {video_path.name}")
        try:
            result = process_local_video(
                video_path, output_dir, args.interval, args.max_frames,
                args.frame_format, args.dedup_threshold, args.no_dedup,
                fast, args.json_output, args.force,
            )
            if args.json_output:
                results.append(result)  # type: ignore[arg-type]
            else:
                out_console.print(f"[green]+[/] {result.name}")  # type: ignore[union-attr]
            success_count += 1
        except (LocalVideoError, FrameExtractionError) as e:
            if args.json_output:
                results.append({
                    "status": "error",
                    "video": str(video_path.resolve()),
                    "error": str(e),
                })
            else:
                out_console.print(f"[red]x[/] Failed: {e}")
            error_count += 1

    # JSON output
    if args.json_output:
        if len(args.files) == 1:
            print(json.dumps(results[0], indent=2))
        else:
            print(json.dumps({
                "status": "success" if error_count == 0 else "partial",
                "succeeded": success_count,
                "failed": error_count,
                "results": results,
            }, indent=2))
        return 0

    # Summary
    if error_count > 0:
        out_console.print(
            f"\n[bold yellow]Complete![/] {success_count} succeeded, {error_count} failed"
        )
    else:
        out_console.print(f"\n[bold green]Complete![/] {success_count} video(s) processed")

    return 0


def _handle_completion(command: str, args: list[str]) -> int:
    """Handle completion subcommand for backward compatibility."""
    from vidflow.capture.completion import completion_command
    return completion_command(command, args)


def ytcapture_entry() -> None:
    """Entry point for ytcapture command."""
    if len(sys.argv) > 1 and sys.argv[1] == "completion":
        sys.exit(_handle_completion("ytcapture", sys.argv[2:]))
    sys.exit(ytcapture_main())


def vidcapture_entry() -> None:
    """Entry point for vidcapture command."""
    if len(sys.argv) > 1 and sys.argv[1] == "completion":
        sys.exit(_handle_completion("vidcapture", sys.argv[2:]))
    sys.exit(vidcapture_main())
