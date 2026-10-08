"""CLI entry point for vidflow.

Provides subcommands for video capture and transcription:
- youtube: Capture frames from YouTube videos
- local: Capture frames from local video files
- transcribe: Full visual transcription of captured video frames
- polish: Text-only cleanup of captured caption text
"""

import argparse
import os
import sys
from pathlib import Path

from vidflow import __version__
from vidflow.capture.config import DEDUP_THRESHOLD_HELP, get_config_for_defaults
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
    images=False (polish) skips image options and -t/--title (polish never
    retitles its input) and raises the batch default, since text-only
    requests carry no frame payloads.
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
    if images:
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
        help="Show what would be done without processing (no downloads or model calls)",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only estimate token usage",
    )
    parser.add_argument(
        "--keep-capture",
        action="store_true",
        help=(
            "Leave the capture note in place when a new transcript file is written "
            "(default: move it to the sibling transcripts/ folder)"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with all subcommands."""
    # Capture defaults come from ~/.config/vidflow/config.yml (merged over
    # DEFAULT_CONFIG), the same source the standalone ytcapture/vidcapture use.
    _cfg = get_config_for_defaults()

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
  completion  Print or install the bash completion script (see below)

Models default to the local inference gateway; claude-* ids route to the
Anthropic API as the quality escape hatch.

Examples:
  vidflow youtube https://youtube.com/watch?v=...
  vidflow youtube                 (no URLs: read from clipboard on macOS)
  vidflow youtube URL1 URL2 --transcribe
  vidflow youtube URL --polish
  vidflow youtube URL --transcribe -m claude-opus-5
  vidflow local recording.mp4 --transcribe
  vidflow local part1.mp4 part2.mp4 --merge --transcribe
  vidflow transcribe talk1.md talk2.md       (one transcript each)
  vidflow transcribe part1.md part2.md --merge -o combined.md
  vidflow polish capture.md

Shell Completion:
  vidflow completion bash            Output completion script
  vidflow completion bash --install  Install to user completions directory
  vidflow completion bash --path     Show installation path
""",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- youtube subcommand ---
    yt_parser = subparsers.add_parser("youtube", help="Capture frames from YouTube videos")
    yt_parser.add_argument(
        "urls",
        nargs="*",
        help="YouTube video URL(s) (default: read from clipboard on macOS)",
    )
    yt_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output directory (default: current directory)",
    )
    yt_parser.add_argument(
        "--interval",
        type=int,
        default=_cfg["interval"],
        help=f"Frame extraction interval in seconds (default: {_cfg['interval']})",
    )
    yt_parser.add_argument(
        "--max-frames",
        type=int,
        default=_cfg["max_frames"],
        help="Maximum number of frames to extract",
    )
    yt_parser.add_argument(
        "--frame-format",
        choices=["jpg", "png"],
        default=_cfg["frame_format"],
        help=f"Frame image format (default: {_cfg['frame_format']})",
    )
    yt_parser.add_argument(
        "--language",
        default=_cfg["language"],
        help=f"Transcript language code (default: {_cfg['language']})",
    )
    yt_parser.add_argument(
        "--prefer-manual",
        action="store_true",
        default=_cfg["prefer_manual"],
        help="Only use manually created transcripts",
    )
    yt_parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=_cfg["dedup_threshold"],
        help=DEDUP_THRESHOLD_HELP.format(_cfg["dedup_threshold"]),
    )
    yt_parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable frame deduplication",
    )
    yt_parser.add_argument(
        "--keep-video",
        action="store_true",
        default=_cfg["keep_video"],
        help="Keep downloaded video file",
    )
    yt_parser.add_argument(
        "--no-ai-title",
        action="store_true",
        default=not _cfg["ai_title"],
        help="Skip AI title generation",
    )
    yt_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Recapture videos that already have a note in the output directory",
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
        default=_cfg["interval"],
        help=f"Frame extraction interval in seconds (default: {_cfg['interval']})",
    )
    local_parser.add_argument(
        "--max-frames",
        type=int,
        default=_cfg["max_frames"],
        help="Maximum number of frames to extract",
    )
    local_parser.add_argument(
        "--frame-format",
        choices=["jpg", "png"],
        default=_cfg["frame_format"],
        help=f"Frame image format (default: {_cfg['frame_format']})",
    )
    local_parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=_cfg["dedup_threshold"],
        help=DEDUP_THRESHOLD_HELP.format(_cfg["dedup_threshold"]),
    )
    local_parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable frame deduplication",
    )
    local_parser.add_argument(
        "--fast",
        action="store_true",
        default=_cfg["fast"],
        help="Use fast keyframe-seeking extraction" + (" (default)" if _cfg["fast"] else ""),
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
        help="Ignore sidecar .vtt transcripts and embedded subtitle tracks",
    )
    local_parser.add_argument(
        "--subtitle-track",
        type=int,
        metavar="N",
        help="Use embedded subtitle track N (0-based among subtitle streams; skips sidecar .vtt)",
    )
    local_parser.add_argument(
        "--vtt",
        type=Path,
        metavar="FILE",
        help=(
            "Use this WebVTT transcript (single input; overrides sidecar discovery "
            "and embedded tracks)"
        ),
    )
    local_parser.add_argument(
        "--list-subtitles",
        action="store_true",
        help="List sidecar .vtt and embedded subtitle tracks, then exit (no capture)",
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
        help=(
            "Output file, or directory for auto-named files (default: beside each input); "
            "a file path needs a single input or --merge"
        ),
    )
    tx_parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge multiple inputs into one transcript (default: one transcript per input)",
    )
    _add_transcribe_args(tx_parser)
    add_common_args(tx_parser)

    # --- polish subcommand ---
    pol_parser = subparsers.add_parser(
        "polish",
        help=(
            "Polish captured caption text in place with the configured model "
            "(text-only; each input is improved, never merged or retitled)"
        ),
    )
    pol_parser.add_argument("files", nargs="+", type=Path, help="Vidcapture markdown file(s)")
    pol_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Write the polished copy to this file (single input) or directory "
            "(same filename) instead of in place"
        ),
    )
    _add_transcribe_args(pol_parser, images=False)
    add_common_args(pol_parser)

    return parser


def _resolve_youtube_urls(args: argparse.Namespace) -> list[str] | None:
    """Resolve capture URLs from arguments or, failing that, the clipboard.

    Returns None when no URLs are available (usage error) and an empty
    list when the user declines the clipboard confirmation prompt.
    """
    if args.urls:
        return args.urls

    from vidflow.capture.utils import get_clipboard_urls

    urls = get_clipboard_urls()
    if not urls:
        print(
            "No URLs provided and no YouTube URLs found in clipboard. "
            "Pass URLs as arguments or copy one to the clipboard.",
            file=sys.stderr,
        )
        return None

    print(f"Using {len(urls)} YouTube URL(s) from clipboard:", file=sys.stderr)
    for url in urls:
        print(f"  {url}", file=sys.stderr)

    # Confirm before downloading from a possibly stale clipboard, but only
    # when interactive; -y/--yes and --json skip the prompt
    if sys.stdin.isatty() and not args.yes and not args.json_output:
        response = input("Proceed with capture? [Y/n]: ").strip().lower()
        if response not in ("", "y", "yes"):
            print("Cancelled.", file=sys.stderr)
            return []

    return urls


def cmd_youtube(args: argparse.Namespace) -> int:
    """Handle the youtube subcommand."""
    logger = setup_logging(args.verbose, args.quiet)
    output_dir = args.output or Path.cwd()
    errors = []
    all_results = []
    captured_paths = []
    unattempted = 0  # videos never tried after a block aborts the run

    urls = _resolve_youtube_urls(args)
    if urls is None:
        return ExitCode.USAGE_ERROR
    if not urls:  # user declined the clipboard confirmation
        return ExitCode.SUCCESS

    # Normalize bare video IDs, expand playlist URLs, deduplicate. Playlist
    # expansion (one flat-playlist request per playlist) also runs for a dry
    # run so the plan can show which members are already captured.
    from vidflow.capture.video import normalize_video_urls

    urls = normalize_video_urls(urls, log=lambda msg: print(msg, file=sys.stderr))
    if not urls:
        print("No valid video URLs found.", file=sys.stderr)
        return ExitCode.USAGE_ERROR

    if args.dry_run:
        return _dry_run_youtube(args, urls, output_dir, logger)

    from vidflow.capture import capture_youtube
    from vidflow.capture.transcript import TranscriptBlocked
    from vidflow.capture.video import VideoBlocked

    # Per-video progress lines; a single video's result is its own summary
    per_video = len(urls) > 1 and not args.json_output

    for i, url in enumerate(urls):
        try:
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
                force=args.force,
            )
        except (TranscriptBlocked, VideoBlocked) as e:
            msg = (
                f"ABORTED: {e} (blocked at video {i + 1}/{len(urls)}). Stopping "
                "all requests to avoid deepening the block; retry later or via VPN."
            )
            errors.append(msg)
            if not args.json_output:
                print(msg, file=sys.stderr)
            unattempted = len(urls) - i
            break

        if result.success and result.data.get("skipped"):
            # Already captured (and already post-processed, if that was
            # requested on the earlier run): leave it out of this run's
            # post-processing.
            if per_video:
                print(f"Skipped [{i + 1}/{len(urls)}]: {result.message}", file=sys.stderr)
        elif result.success:
            captured_paths.append(Path(result.data["output_path"]))
        else:
            errors.append(result.message)
            if per_video:
                print(f"Failed [{i + 1}/{len(urls)}]: {result.message}", file=sys.stderr)

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
    total = len(all_results) + unattempted

    if len(urls) == 1 and len(all_results) == 1:
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


def _post_process_plan(args: argparse.Namespace) -> dict | None:
    """Describe the post-processing step a capture run would chain into."""
    if args.transcribe:
        return {"step": "transcribe", "model": args.model}
    if args.polish:
        return {"step": "polish", "model": args.model}
    return None


def _dry_run_youtube(
    args: argparse.Namespace,
    urls: list[str],
    output_dir: Path,
    logger,
) -> int:
    """Report what `vidflow youtube` would do, without capturing anything.

    `urls` are already-expanded video URLs. Each is checked against notes
    already in the output directory the same way the capture skip does; no
    per-video yt-dlp or model call is made.
    """
    from vidflow.capture.core import find_existing_capture
    from vidflow.capture.utils import extract_video_id

    rows = []
    for url in urls:
        video_id = extract_video_id(url)
        existing = find_existing_capture(output_dir, video_id) if video_id else None
        if existing is None:
            action = "capture"
        elif args.force:
            action = "recapture"
        else:
            action = "skip"
        rows.append(
            {
                "url": url,
                "video_id": video_id,
                "action": action,
                "existing": str(existing) if existing else None,
            }
        )

    counts = {k: sum(1 for r in rows if r["action"] == k) for k in ("capture", "recapture", "skip")}
    plan = _post_process_plan(args)
    parts = [f"would capture {counts['capture'] + counts['recapture']} video(s)"]
    if counts["skip"]:
        parts.append(f"skip {counts['skip']} already captured")
    if plan:
        parts.append(f"then {plan['step']} with {plan['model']}")
    message = "Dry run: " + ", ".join(parts)

    result = OperationResult(
        success=True,
        message=message,
        data={
            "dry_run": True,
            "output_dir": str(output_dir),
            "videos": rows,
            "post_process": plan,
        },
    )

    if not args.json_output:
        print(f"Dry run (no capture). Output directory: {output_dir}", file=sys.stderr)
        for i, row in enumerate(rows, 1):
            if row["action"] == "skip":
                detail = f"skip, already captured: {Path(row['existing']).name}"
            elif row["action"] == "recapture":
                detail = f"recapture (--force), replacing: {Path(row['existing']).name}"
            else:
                detail = "capture"
            print(f"  [{i}/{len(rows)}] {row['video_id'] or '-':<11}  {detail}", file=sys.stderr)
            print(f"        {row['url']}", file=sys.stderr)

    output_result(result, args.json_output, logger)
    return ExitCode.SUCCESS


def _transcribe_youtube_captures(
    args: argparse.Namespace,
    captured_paths: list[Path],
    errors: list[str],
) -> list[OperationResult]:
    """Run YouTube-aware transcription on captured markdown files.

    Each capture is transcribed independently; merging is a local-video
    concern (stitching one long event) and not offered on the youtube path.
    """
    from vidflow.youtube import transcribe_youtube

    results = []
    single = len(captured_paths) == 1

    for path in captured_paths:
        result = transcribe_youtube(
            input_path=path,
            output=args.output if single else None,
            title=args.title if single else None,
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
            keep_capture=args.keep_capture,
        )
        if not result.success:
            errors.append(result.message)
        results.append(result)

    return results


def _polish_captures(
    args: argparse.Namespace,
    paths: list[Path],
    errors: list[str],
    output: Path | None = None,
) -> list[OperationResult]:
    """Polish each capture note's caption text, one input at a time.

    Polish never merges: every input is improved on its own, in place, or
    written to ``output`` (a file for a single input, else a directory).
    Capture subcommands pass no output — their -o is the capture directory,
    and the fresh capture notes are polished where they were written.
    """
    from vidflow.transcribe import polish_markdown

    results = []
    for path in paths:
        result = polish_markdown(
            input_path=path,
            output=output,
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
            keep_capture=args.keep_capture,
        )
        if not result.success:
            errors.append(result.message)
        results.append(result)

    return results


_SIDECAR_MATCH_LABELS = {
    "vtt": "--vtt",
    "stem": "sidecar",
    "teams-meeting": "Teams meeting sidecar",
}


def _caption_vtt(args: argparse.Namespace, video_path: Path) -> tuple[Path | None, str | None]:
    """The WebVTT file a local capture would read, and how it was matched.

    Match is "vtt" (explicit --vtt), "stem" (named after the video), or
    "teams-meeting" (named after a Teams recording's meeting; the capture
    still rejects it if its cues outrun the video). Discovery only, so a
    --subtitle-track or --no-subtitles run reports none.
    """
    from vidflow.capture.subtitles import find_sidecar_vtt, is_meeting_name_match

    if args.vtt is not None:
        return args.vtt, "vtt"
    if args.no_subtitles or args.subtitle_track is not None:
        return None, None
    sidecar = find_sidecar_vtt(video_path)
    if sidecar is None:
        return None, None
    return sidecar, "teams-meeting" if is_meeting_name_match(video_path, sidecar) else "stem"


def _list_subtitles(args: argparse.Namespace) -> int:
    """Print the sidecar .vtt and embedded subtitle tracks for each input file and exit."""
    import json as _json

    from vidflow.capture.subtitles import SubtitleError, probe_subtitle_streams

    all_data = []
    exit_code = ExitCode.SUCCESS

    for video_path in args.files:
        sidecar, match = _caption_vtt(args, video_path)
        entry: dict = {
            "file": str(video_path),
            "sidecar": str(sidecar) if sidecar else None,
            "sidecar_match": match,
        }
        if sidecar and not args.json_output:
            how = _SIDECAR_MATCH_LABELS[match]
            print(f"{video_path}:\n  {how} {sidecar.name} (preferred)")
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
            if not sidecar:
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

    if args.vtt is not None:
        problem = None
        if len(args.files) > 1:
            problem = "--vtt names one transcript; pass a single video file"
        elif args.no_subtitles or args.subtitle_track is not None:
            problem = "--vtt cannot be combined with --no-subtitles or --subtitle-track"
        elif not args.vtt.is_file():
            problem = f"--vtt file not found: {args.vtt}"
        if problem:
            print(f"Error: {problem}", file=sys.stderr)
            return ExitCode.USAGE_ERROR

    if args.merge and args.polish:
        print(
            "Error: --merge applies to --transcribe only; polish improves each capture "
            "on its own and never merges",
            file=sys.stderr,
        )
        return ExitCode.USAGE_ERROR

    output_dir = args.output or Path.cwd()
    errors = []
    all_results = []
    captured_paths = []

    # Resolve fast flag
    fast = args.fast and not args.no_fast

    if args.dry_run:
        return _dry_run_local(args, output_dir, logger)

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
            vtt=args.vtt,
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
        tx_results = _transcribe_captures(args, captured_paths, errors)
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


def _dry_run_local(args: argparse.Namespace, output_dir: Path, logger) -> int:
    """Report what `vidflow local` would do without probing or capturing.

    Output filenames derive from ffprobe metadata, so they are not
    predicted here; missing inputs are flagged. Sidecar .vtt discovery is
    a directory listing only, so the plan names the caption source.
    """
    rows = []
    for p in args.files:
        row: dict = {"file": str(p), "action": "capture" if p.is_file() else "missing"}
        vtt, match = _caption_vtt(args, p) if p.is_file() else (None, None)
        row["sidecar"] = vtt.name if vtt else None
        row["sidecar_match"] = match
        rows.append(row)
    missing = sum(1 for r in rows if r["action"] == "missing")
    plan = _post_process_plan(args)
    parts = [f"would capture {len(rows) - missing} file(s)"]
    if missing:
        parts.append(f"{missing} missing")
    if plan:
        merged = " (merged)" if getattr(args, "merge", False) and len(rows) > 1 else ""
        parts.append(f"then {plan['step']}{merged} with {plan['model']}")
    result = OperationResult(
        success=missing == 0,
        message="Dry run: " + ", ".join(parts),
        data={
            "dry_run": True,
            "output_dir": str(output_dir),
            "files": rows,
            "post_process": plan,
        },
        errors=[r["file"] + ": not found" for r in rows if r["action"] == "missing"] or None,
    )
    if not args.json_output:
        print(f"Dry run. Output directory: {output_dir}", file=sys.stderr)
        for i, row in enumerate(rows, 1):
            print(f"  [{i}/{len(rows)}] {row['action']:<8} {row['file']}", file=sys.stderr)
            if row["sidecar"]:
                how = _SIDECAR_MATCH_LABELS[row["sidecar_match"]]
                print(f"         captions from {how} {row['sidecar']}", file=sys.stderr)
    output_result(result, args.json_output, logger)
    return ExitCode.SUCCESS if result.success else ExitCode.ERROR


def _is_dir_target(path: Path | None) -> bool:
    """True if -o names a directory (existing, or spelled with a trailing separator)."""
    return path is not None and (path.is_dir() or str(path).endswith(os.sep))


def _transcribe_captures(
    args: argparse.Namespace,
    paths: list[Path],
    errors: list[str],
) -> list[OperationResult]:
    """Run full visual transcription on capture notes.

    One transcript per input by default; ``--merge`` stitches all inputs
    into a single transcript (an H1 per source file). Shared by
    ``vidflow transcribe`` and ``vidflow local --transcribe``. A directory
    ``-o`` receives every auto-named transcript; a file ``-o`` or ``-t``
    applies only when there is a single output.
    """
    from vidflow.transcribe import transcribe_markdown

    merge = getattr(args, "merge", False)
    groups = [paths] if merge else [[p] for p in paths]
    single = len(groups) == 1

    results = []
    for group in groups:
        result = transcribe_markdown(
            input_paths=group,
            output=args.output if single or _is_dir_target(args.output) else None,
            title=args.title if single else None,
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
            keep_capture=args.keep_capture,
        )
        if not result.success:
            errors.append(result.message)
        results.append(result)

    return results


def _combine_results(
    results: list[OperationResult], errors: list[str], verb: str
) -> OperationResult:
    """One input's result as-is, or an aggregate over several."""
    if len(results) == 1:
        return results[0]
    succeeded = sum(1 for r in results if r.success)
    return OperationResult(
        success=not errors,
        message=f"{verb} {succeeded}/{len(results)} inputs",
        data={"results": [r.to_dict() for r in results]},
        errors=errors or None,
    )


def cmd_transcribe(args: argparse.Namespace) -> int:
    """Handle the transcribe subcommand.

    Each input gets its own transcript unless --merge is given.
    """
    logger = setup_logging(args.verbose, args.quiet)

    if len(args.files) > 1 and not args.merge:
        if args.output and not _is_dir_target(args.output):
            print(
                "Error: -o names a single file; with multiple inputs pass a directory, "
                "or --merge for one combined transcript",
                file=sys.stderr,
            )
            return ExitCode.USAGE_ERROR
        if args.title:
            print(
                "Error: -t/--title needs a single input, or --merge for one combined transcript",
                file=sys.stderr,
            )
            return ExitCode.USAGE_ERROR

    errors: list[str] = []
    results = _transcribe_captures(args, args.files, errors)
    combined = _combine_results(results, errors, "Transcribed")

    output_result(combined, args.json_output, logger)
    return ExitCode.SUCCESS if combined.success else ExitCode.ERROR


def cmd_polish(args: argparse.Namespace) -> int:
    """Handle the polish subcommand.

    Each input is polished on its own (in place, or to -o); polish never
    merges, retitles, or regenerates frontmatter.
    """
    logger = setup_logging(args.verbose, args.quiet)

    if len(args.files) > 1 and args.output and not _is_dir_target(args.output):
        print(
            "Error: -o names a single file; with multiple inputs pass a directory",
            file=sys.stderr,
        )
        return ExitCode.USAGE_ERROR

    errors: list[str] = []
    results = _polish_captures(args, args.files, errors, output=args.output)
    combined = _combine_results(results, errors, "Polished")

    output_result(combined, args.json_output, logger)
    return ExitCode.SUCCESS if combined.success else ExitCode.ERROR


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
