"""CLI entry point for vidflow.

Provides subcommands for video capture and transcription:
- youtube: Capture frames from YouTube videos
- local: Capture frames from local video files
- transcribe: Full visual transcription of captured video frames
- polish: Text-only cleanup of captured caption text
"""

import argparse
import sys
from pathlib import Path

from vidflow import __version__
from vidflow.cli_common import (
    ExitCode,
    OperationResult,
    add_common_args,
    output_result,
    setup_logging,
)
from vidflow.models_config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONTEXT_FRAMES,
    DEFAULT_POLISH_BATCH_SIZE,
    add_model_args,
)


def _add_transcribe_args(parser: argparse.ArgumentParser, images: bool = True) -> None:
    """Add vidscribe transcription options to a parser.

    Used by youtube and local subcommands (shared by --transcribe and
    --polish), and by the transcribe and polish subcommands directly.
    images=False (polish) skips image options and raises the batch default,
    since text-only requests carry no frame payloads.
    """
    add_model_args(parser)
    batch_default = DEFAULT_BATCH_SIZE if images else DEFAULT_POLISH_BATCH_SIZE
    parser.add_argument(
        "--batch-size",
        type=int,
        default=batch_default,
        help=f"Sections per API batch (default: {batch_default})",
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=DEFAULT_CONTEXT_FRAMES,
        help=f"Previous sections for continuity context (default: {DEFAULT_CONTEXT_FRAMES})",
    )
    if images:
        parser.add_argument(
            "--max-dimension",
            type=int,
            default=1568,
            help="Max image dimension for resizing (default: 1568)",
        )
    parser.add_argument(
        "-c",
        "--context",
        action="append",
        dest="context_files",
        type=Path,
        help="Background context file (repeatable)",
    )
    parser.add_argument(
        "-t",
        "--title",
        help="Override title (auto-generated if omitted)",
    )
    parser.add_argument(
        "-y",
        "--yes",
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
  transcribe  Full visual transcription of captured frames (frames + captions)
  polish      Text-only cleanup of captured caption text (in place by default)

Models default to the local inference gateway; claude-* ids route to the
Anthropic API as the quality escape hatch.

Examples:
  vidflow youtube https://youtube.com/watch?v=...
  vidflow youtube URL1 URL2 --transcribe
  vidflow youtube URL --polish
  vidflow youtube URL --transcribe --merge -m claude-opus-5
  vidflow local recording.mp4 --transcribe
  vidflow local *.mp4 --merge --transcribe
  vidflow transcribe part1.md part2.md -o combined.md
  vidflow polish capture.md
""",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- youtube subcommand ---
    yt_parser = subparsers.add_parser("youtube", help="Capture frames from YouTube videos")
    yt_parser.add_argument("urls", nargs="+", help="YouTube video URL(s)")
    yt_parser.add_argument(
        "-o",
        "--output",
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
    yt_post = yt_parser.add_mutually_exclusive_group()
    yt_post.add_argument(
        "--transcribe",
        action="store_true",
        help="Also run full visual transcription on captured frames",
    )
    yt_post.add_argument(
        "--polish",
        action="store_true",
        help="Also polish captured caption text in place (text-only, no frames sent)",
    )
    yt_parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge multiple URLs into a single output",
    )
    _add_transcribe_args(yt_parser)
    add_common_args(yt_parser)

    # --- local subcommand ---
    local_parser = subparsers.add_parser("local", help="Capture frames from local video files")
    local_parser.add_argument("files", nargs="+", type=Path, help="Local video file(s)")
    local_parser.add_argument(
        "-o",
        "--output",
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
        "-f",
        "--force",
        action="store_true",
        help="Overwrite existing output files",
    )
    local_parser.add_argument(
        "--no-subtitles",
        action="store_true",
        help="Ignore embedded subtitle tracks (skip extraction)",
    )
    local_parser.add_argument(
        "--subtitle-track",
        type=int,
        metavar="N",
        help="Use subtitle track index N (0-based among subtitle streams)",
    )
    local_parser.add_argument(
        "--list-subtitles",
        action="store_true",
        help="List embedded subtitle tracks and exit (no capture)",
    )
    local_post = local_parser.add_mutually_exclusive_group()
    local_post.add_argument(
        "--transcribe",
        action="store_true",
        help="Also run full visual transcription on captured frames",
    )
    local_post.add_argument(
        "--polish",
        action="store_true",
        help="Also polish captured caption text in place (text-only, no frames sent)",
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
        "transcribe",
        help="Full visual transcription of captured frames with the configured model",
    )
    tx_parser.add_argument("files", nargs="+", type=Path, help="Vidcapture markdown file(s)")
    tx_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file path (auto-generated if omitted)",
    )
    _add_transcribe_args(tx_parser)
    add_common_args(tx_parser)

    # --- polish subcommand ---
    pol_parser = subparsers.add_parser(
        "polish",
        help=(
            "Polish captured caption text with the configured model "
            "(text-only; a single input is polished in place)"
        ),
    )
    pol_parser.add_argument("files", nargs="+", type=Path, help="Vidcapture markdown file(s)")
    pol_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Write a new file here instead of polishing in place "
            "(multiple inputs always merge into a new file)"
        ),
    )
    _add_transcribe_args(pol_parser, images=False)
    add_common_args(pol_parser)

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

    # If --transcribe or --polish, run post-processing (mutually exclusive)
    if args.transcribe and captured_paths:
        tx_results = _transcribe_youtube_captures(args, captured_paths, errors)
        all_results.extend(tx_results)
    elif args.polish and captured_paths:
        pol_results = _polish_captures(args, captured_paths, errors)
        all_results.extend(pol_results)

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
                provider=args.provider,
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
                provider=args.provider,
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


def _polish_captures(
    args: argparse.Namespace,
    captured_paths: list[Path],
    errors: list[str],
) -> list[OperationResult]:
    """Run text-only caption polish on captured markdown files."""
    from vidflow.transcribe import polish_markdown

    results = []
    input_groups = [captured_paths] if args.merge else [[p] for p in captured_paths]

    # Each captured note is polished in place (output/title None); a
    # merged run needs a new file, auto-named beside the first capture.
    for paths in input_groups:
        result = polish_markdown(
            input_paths=paths,
            output=None,
            title=args.title if args.merge else None,
            context_files=args.context_files,
            model=args.model,
            provider=args.provider,
            batch_size=args.batch_size,
            context_frames=args.context_frames,
            temperature=args.temperature,
            auto_confirm=args.yes,
            dry_run=args.dry_run,
            estimate_only=args.estimate_only,
            json_output=args.json_output,
        )
        if not result.success:
            errors.append(result.message)
        results.append(result)

    return results


def _list_subtitles(args: argparse.Namespace) -> int:
    """Print embedded subtitle tracks for each input file and exit."""
    import json as _json

    from vidflow.capture.subtitles import SubtitleError, probe_subtitle_streams

    all_data = []
    exit_code = ExitCode.SUCCESS

    for video_path in args.files:
        entry: dict = {"file": str(video_path)}
        try:
            streams = probe_subtitle_streams(video_path)
        except SubtitleError as e:
            entry["error"] = str(e)
            exit_code = ExitCode.ERROR
            if not args.json_output:
                print(f"{video_path}: ERROR: {e}", file=sys.stderr)
            all_data.append(entry)
            continue

        entry["tracks"] = [
            {
                "subtitle_index": s.subtitle_index,
                "stream_index": s.index,
                "codec": s.codec,
                "language": s.language,
                "title": s.title,
                "default": s.is_default,
                "forced": s.is_forced,
                "hearing_impaired": s.is_hearing_impaired,
                "text_based": s.is_text_based,
            }
            for s in streams
        ]
        all_data.append(entry)

        if not args.json_output:
            print(f"{video_path}:")
            if not streams:
                print("  (no embedded subtitle tracks)")
            else:
                for s in streams:
                    print(f"  {s.describe()}")

    if args.json_output:
        print(_json.dumps(all_data, indent=2))

    return exit_code


def cmd_local(args: argparse.Namespace) -> int:
    """Handle the local subcommand."""
    logger = setup_logging(args.verbose, args.quiet)

    # --list-subtitles is a pure inspection mode; no capture/transcription
    if getattr(args, "list_subtitles", False):
        return _list_subtitles(args)

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
            use_subtitles=not args.no_subtitles,
            subtitle_track=args.subtitle_track,
        )

        if result.success and result.data:
            output_path = result.data.get("output_path") or result.data.get("output_file")
            if output_path:
                captured_paths.append(Path(output_path))
        if not result.success:
            errors.append(result.message)

        all_results.append(result)

    # If --transcribe or --polish, run post-processing (mutually exclusive)
    if args.transcribe and captured_paths:
        tx_results = _transcribe_local_captures(args, captured_paths, errors)
        all_results.extend(tx_results)
    elif args.polish and captured_paths:
        pol_results = _polish_captures(args, captured_paths, errors)
        all_results.extend(pol_results)

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
            provider=args.provider,
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
                provider=args.provider,
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
        provider=args.provider,
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


def cmd_polish(args: argparse.Namespace) -> int:
    """Handle the polish subcommand.

    A single input with no -o is polished in place; multiple inputs are
    merged into a single new output file.
    """
    logger = setup_logging(args.verbose, args.quiet)

    from vidflow.transcribe import polish_markdown

    result = polish_markdown(
        input_paths=args.files,
        output=args.output,
        title=args.title,
        context_files=args.context_files,
        model=args.model,
        provider=args.provider,
        batch_size=args.batch_size,
        context_frames=args.context_frames,
        temperature=args.temperature,
        auto_confirm=args.yes,
        dry_run=args.dry_run,
        estimate_only=args.estimate_only,
        json_output=args.json_output,
    )

    output_result(result, args.json_output, logger)
    return ExitCode.SUCCESS if result.success else ExitCode.ERROR


def main(argv: list[str] | None = None) -> int:
    """Main entry point for vidflow command."""
    if argv is None:
        argv = sys.argv[1:]

    # Handle completion subcommand before argparse
    if argv and argv[0] == "completion":
        from vidflow.completion import completion_command

        return completion_command(argv[1:])

    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help(sys.stderr)
        return ExitCode.USAGE_ERROR

    handlers = {
        "youtube": cmd_youtube,
        "local": cmd_local,
        "transcribe": cmd_transcribe,
        "polish": cmd_polish,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help(sys.stderr)
        return ExitCode.USAGE_ERROR

    return handler(args)
