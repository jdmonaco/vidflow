"""CLI entry point for vidflow.

Provides subcommands for video capture and transcription:
- youtube: Capture frames from YouTube videos
- local: Capture frames from local video files
- transcribe: Transcribe captured video frames
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from vidflow import __version__
from vidflow.cli_common import (
    ExitCode,
    OperationResult,
    add_common_args,
    output_result,
    setup_logging,
)


def _add_transcribe_args(parser: argparse.ArgumentParser) -> None:
    """Add vidscribe transcription options to a parser.

    Used by youtube and local subcommands when --transcribe is set,
    and by the transcribe subcommand directly.
    """
    parser.add_argument(
        "-m", "--model",
        default="claude-sonnet-4-20250514",
        help="Claude model for transcription (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Frames per API batch (default: 10)",
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=3,
        help="Previous frames for continuity context (default: 3)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="API temperature (default: 0.2)",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=1568,
        help="Max image dimension for resizing (default: 1568)",
    )
    parser.add_argument(
        "-c", "--context",
        action="append",
        dest="context_files",
        type=Path,
        help="Background context file (repeatable)",
    )
    parser.add_argument(
        "-t", "--title",
        help="Override title (auto-generated if omitted)",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without processing",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only estimate token usage",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="vidflow",
        description="Unified video capture and transcription CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Commands:
  youtube     Capture frames from YouTube videos
  local       Capture frames from local video files
  transcribe  Transcribe captured video frames with Claude Vision

Examples:
  vidflow youtube https://youtube.com/watch?v=...
  vidflow youtube URL1 URL2 --transcribe
  vidflow youtube URL --transcribe --merge -m claude-opus-4-20250514
  vidflow local recording.mp4 --transcribe
  vidflow local *.mp4 --merge --transcribe
  vidflow transcribe part1.md part2.md -o combined.md
""",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- youtube subcommand ---
    yt_parser = subparsers.add_parser(
        "youtube", help="Capture frames from YouTube videos"
    )
    yt_parser.add_argument(
        "urls", nargs="+", help="YouTube video URL(s)"
    )
    yt_parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output directory (default: current directory)",
    )
    yt_parser.add_argument(
        "--interval",
        type=int,
        default=15,
        help="Frame extraction interval in seconds (default: 15)",
    )
    yt_parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum number of frames to extract",
    )
    yt_parser.add_argument(
        "--frame-format",
        choices=["jpg", "png"],
        default="jpg",
        help="Frame image format (default: jpg)",
    )
    yt_parser.add_argument(
        "--language",
        default="en",
        help="Transcript language code (default: en)",
    )
    yt_parser.add_argument(
        "--prefer-manual",
        action="store_true",
        help="Only use manually created transcripts",
    )
    yt_parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.95,
        help="Similarity threshold for frame deduplication (default: 0.95)",
    )
    yt_parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable frame deduplication",
    )
    yt_parser.add_argument(
        "--keep-video",
        action="store_true",
        help="Keep downloaded video file",
    )
    yt_parser.add_argument(
        "--no-ai-title",
        action="store_true",
        help="Skip AI title generation",
    )
    yt_parser.add_argument(
        "--transcribe",
        action="store_true",
        help="Also transcribe captured frames with Claude Vision",
    )
    yt_parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge multiple URLs into a single output",
    )
    _add_transcribe_args(yt_parser)
    add_common_args(yt_parser)

    # --- local subcommand ---
    local_parser = subparsers.add_parser(
        "local", help="Capture frames from local video files"
    )
    local_parser.add_argument(
        "files", nargs="+", type=Path, help="Local video file(s)"
    )
    local_parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output directory (default: current directory)",
    )
    local_parser.add_argument(
        "--interval",
        type=int,
        default=15,
        help="Frame extraction interval in seconds (default: 15)",
    )
    local_parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum number of frames to extract",
    )
    local_parser.add_argument(
        "--frame-format",
        choices=["jpg", "png"],
        default="jpg",
        help="Frame image format (default: jpg)",
    )
    local_parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.95,
        help="Similarity threshold for frame deduplication (default: 0.95)",
    )
    local_parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable frame deduplication",
    )
    local_parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast keyframe-seeking extraction",
    )
    local_parser.add_argument(
        "--no-fast",
        action="store_true",
        help="Disable fast keyframe-seeking",
    )
    local_parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Overwrite existing output files",
    )
    local_parser.add_argument(
        "--transcribe",
        action="store_true",
        help="Also transcribe captured frames with Claude Vision",
    )
    local_parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge multiple files into a single output",
    )
    _add_transcribe_args(local_parser)
    add_common_args(local_parser)

    # --- transcribe subcommand ---
    tx_parser = subparsers.add_parser(
        "transcribe", help="Transcribe captured video frames with Claude Vision"
    )
    tx_parser.add_argument(
        "files", nargs="+", type=Path, help="Vidcapture markdown file(s)"
    )
    tx_parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output file path (auto-generated if omitted)",
    )
    _add_transcribe_args(tx_parser)
    add_common_args(tx_parser)

    return parser


def cmd_youtube(args: argparse.Namespace) -> int:
    """Handle the youtube subcommand."""
    logger = setup_logging(args.verbose, args.quiet)
    output_dir = args.output or Path.cwd()
    errors = []
    all_results = []
    captured_paths = []

    for url in args.urls:
        from vidflow.capture import capture_youtube

        result = capture_youtube(
            url=url,
            output_dir=output_dir,
            interval=args.interval,
            max_frames=args.max_frames,
            frame_format=args.frame_format,
            language=args.language,
            prefer_manual=args.prefer_manual,
            dedup_threshold=args.dedup_threshold,
            no_dedup=args.no_dedup,
            keep_video=args.keep_video,
            no_ai_title=args.no_ai_title,
        )

        if result.success:
            captured_paths.append(Path(result.data["output_path"]))
        else:
            errors.append(result.message)

        all_results.append(result)

    # If --transcribe, run transcription
    if args.transcribe and captured_paths:
        tx_results = _transcribe_youtube_captures(args, captured_paths, errors)
        all_results.extend(tx_results)

    # Build combined result
    success_count = sum(1 for r in all_results if r.success)
    total = len(all_results)

    if len(args.urls) == 1 and len(all_results) == 1:
        combined = all_results[0]
    else:
        combined = OperationResult(
            success=len(errors) == 0,
            message=f"Processed {success_count}/{total} operations",
            data={"results": [r.to_dict() for r in all_results]},
            errors=errors if errors else None,
        )

    output_result(combined, args.json_output, logger)
    return ExitCode.SUCCESS if combined.success else ExitCode.ERROR


def _transcribe_youtube_captures(
    args: argparse.Namespace,
    captured_paths: list[Path],
    errors: list[str],
) -> list[OperationResult]:
    """Run YouTube-aware transcription on captured markdown files."""
    from vidflow.youtube import transcribe_youtube

    results = []

    if args.merge:
        # TODO: Merged YouTube transcription for multi-part content
        # For now, process sequentially and note this as future work
        for path in captured_paths:
            result = transcribe_youtube(
                input_path=path,
                output=args.output,
                title=args.title,
                context_files=args.context_files,
                model=args.model,
                batch_size=args.batch_size,
                context_frames=args.context_frames,
                temperature=args.temperature,
                max_dimension=args.max_dimension,
                auto_confirm=args.yes,
                dry_run=args.dry_run,
                estimate_only=args.estimate_only,
                json_output=args.json_output,
            )
            if not result.success:
                errors.append(result.message)
            results.append(result)
    else:
        # Independent processing (default)
        for path in captured_paths:
            result = transcribe_youtube(
                input_path=path,
                output=args.output if len(captured_paths) == 1 else None,
                title=args.title if len(captured_paths) == 1 else None,
                context_files=args.context_files,
                model=args.model,
                batch_size=args.batch_size,
                context_frames=args.context_frames,
                temperature=args.temperature,
                max_dimension=args.max_dimension,
                auto_confirm=args.yes,
                dry_run=args.dry_run,
                estimate_only=args.estimate_only,
                json_output=args.json_output,
            )
            if not result.success:
                errors.append(result.message)
            results.append(result)

    return results


def cmd_local(args: argparse.Namespace) -> int:
    """Handle the local subcommand."""
    logger = setup_logging(args.verbose, args.quiet)
    output_dir = args.output or Path.cwd()
    errors = []
    all_results = []
    captured_paths = []

    # Resolve fast flag
    fast = args.fast and not args.no_fast

    for video_path in args.files:
        from vidflow.capture import capture_local

        result = capture_local(
            video_path=video_path,
            output_dir=output_dir,
            interval=args.interval,
            max_frames=args.max_frames,
            frame_format=args.frame_format,
            dedup_threshold=args.dedup_threshold,
            no_dedup=args.no_dedup,
            fast=fast,
            force=args.force,
            json_output=args.json_output,
        )

        if result.success and result.data:
            output_path = result.data.get("output_path") or result.data.get("output_file")
            if output_path:
                captured_paths.append(Path(output_path))
        if not result.success:
            errors.append(result.message)

        all_results.append(result)

    # If --transcribe, run standard vidscribe transcription
    if args.transcribe and captured_paths:
        tx_results = _transcribe_local_captures(args, captured_paths, errors)
        all_results.extend(tx_results)

    # Build combined result
    success_count = sum(1 for r in all_results if r.success)
    total = len(all_results)

    if len(args.files) == 1 and len(all_results) == 1:
        combined = all_results[0]
    else:
        combined = OperationResult(
            success=len(errors) == 0,
            message=f"Processed {success_count}/{total} operations",
            data={"results": [r.to_dict() for r in all_results]},
            errors=errors if errors else None,
        )

    output_result(combined, args.json_output, logger)
    return ExitCode.SUCCESS if combined.success else ExitCode.ERROR


def _transcribe_local_captures(
    args: argparse.Namespace,
    captured_paths: list[Path],
    errors: list[str],
) -> list[OperationResult]:
    """Run standard vidscribe transcription on local captures."""
    from vidflow.transcribe import transcribe_markdown

    results = []

    if args.merge:
        # Merge all into one transcription
        result = transcribe_markdown(
            input_paths=captured_paths,
            output=args.output,
            title=args.title,
            context_files=args.context_files,
            model=args.model,
            batch_size=args.batch_size,
            context_frames=args.context_frames,
            temperature=args.temperature,
            max_dimension=args.max_dimension,
            auto_confirm=args.yes,
            dry_run=args.dry_run,
            estimate_only=args.estimate_only,
            json_output=args.json_output,
        )
        if not result.success:
            errors.append(result.message)
        results.append(result)
    else:
        # Independent processing (default)
        for path in captured_paths:
            result = transcribe_markdown(
                input_paths=[path],
                output=args.output if len(captured_paths) == 1 else None,
                title=args.title if len(captured_paths) == 1 else None,
                context_files=args.context_files,
                model=args.model,
                batch_size=args.batch_size,
                context_frames=args.context_frames,
                temperature=args.temperature,
                max_dimension=args.max_dimension,
                auto_confirm=args.yes,
                dry_run=args.dry_run,
                estimate_only=args.estimate_only,
                json_output=args.json_output,
            )
            if not result.success:
                errors.append(result.message)
            results.append(result)

    return results


def cmd_transcribe(args: argparse.Namespace) -> int:
    """Handle the transcribe subcommand.

    Multiple inputs are always merged into a single output.
    """
    logger = setup_logging(args.verbose, args.quiet)

    from vidflow.transcribe import transcribe_markdown

    result = transcribe_markdown(
        input_paths=args.files,
        output=args.output,
        title=args.title,
        context_files=args.context_files,
        model=args.model,
        batch_size=args.batch_size,
        context_frames=args.context_frames,
        temperature=args.temperature,
        max_dimension=args.max_dimension,
        auto_confirm=args.yes,
        dry_run=args.dry_run,
        estimate_only=args.estimate_only,
        json_output=args.json_output,
    )

    output_result(result, args.json_output, logger)
    return ExitCode.SUCCESS if result.success else ExitCode.ERROR


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for vidflow command."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help(sys.stderr)
        return ExitCode.USAGE_ERROR

    handlers = {
        "youtube": cmd_youtube,
        "local": cmd_local,
        "transcribe": cmd_transcribe,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help(sys.stderr)
        return ExitCode.USAGE_ERROR

    return handler(args)
