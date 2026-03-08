"""Standalone vidscribe CLI entry point.

Provides backward-compatible `vidscribe` command.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from vidflow.transcribe.models import VidcaptureDocument
from vidflow.transcribe.output import (
    determine_output_path,
    handle_existing_output,
    load_context_files,
    shorten_path,
)
from vidflow.transcribe.parser import (
    merge_vidcapture_documents,
    parse_vidcapture_markdown,
)
from vidflow.transcribe.processor import VidscribeProcessor
from vidflow.models_config import DEFAULT_BATCH_SIZE, DEFAULT_CONTEXT_FRAMES, add_model_args
from vidflow.transcribe.prompts import DEFAULT_MAX_DIMENSION


def main(argv=None):
    """Standalone vidscribe CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Transcribe vidcapture markdown files using Claude Vision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  vidscribe workshop.md
  vidscribe part1.md part2.md part3.md
  vidscribe session*.md -o full_transcript.md
  vidscribe workshop.md -t "My Workshop Title"
  vidscribe workshop.md -c agenda.md -c glossary.md
  vidscribe workshop.md --dry-run
        """,
    )

    # Required input(s)
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="Vidcapture markdown file(s) to process",
    )

    # Output control
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file path (default: auto-generated from title)",
    )
    parser.add_argument(
        "-t",
        "--title",
        help="Override auto-generated title",
    )

    # Context arguments
    parser.add_argument(
        "-c",
        "--context",
        type=Path,
        action="append",
        dest="context_files",
        help="Background context file(s) to include (can specify multiple)",
    )

    # Processing options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of sections per API request (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=DEFAULT_CONTEXT_FRAMES,
        help=f"Number of previous sections to include for context (default: {DEFAULT_CONTEXT_FRAMES})",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=DEFAULT_MAX_DIMENSION,
        help=f"Maximum image dimension in pixels (default: {DEFAULT_MAX_DIMENSION})",
    )

    # API arguments
    parser.add_argument(
        "-k",
        "--api-key",
        default=os.environ.get("ANTHROPIC_API_KEY"),
        help="Anthropic API key (default: ANTHROPIC_API_KEY env var)",
    )
    add_model_args(parser)

    # Workflow control
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip cost confirmation prompts",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show sections without processing",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only estimate token usage without processing",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output JSON instead of rich console output",
    )

    args = parser.parse_args(argv)

    # Validate API key
    if not args.api_key and not args.dry_run and not args.estimate_only:
        parser.error(
            "Anthropic API key required. Set ANTHROPIC_API_KEY environment "
            "variable or use -k/--api-key"
        )

    # Validate temperature
    if not (0.0 <= args.temperature <= 1.0):
        parser.error("Temperature must be between 0.0 and 1.0")

    # Validate batch size
    if args.batch_size < 1:
        parser.error("Batch size must be at least 1")

    # Helper function to print only in non-JSON mode
    def log(msg: str) -> None:
        if not args.json_output:
            print(msg)

    def log_err(msg: str) -> None:
        if args.json_output:
            print(json.dumps({"status": "error", "error": msg}))
        else:
            print(f"Error: {msg}", file=sys.stderr)

    # Detect Exa API key for citation search
    exa_api_key = os.environ.get("EXA_API_KEY")
    if exa_api_key:
        log("Citation search enabled (EXA_API_KEY detected)")
    elif args.verbose:
        log("Citation search disabled (no EXA_API_KEY)")

    # Validate input files
    for input_file in args.inputs:
        if not input_file.exists():
            log_err(f"Input file not found: {input_file}")
            return 1
        if input_file.suffix.lower() != ".md":
            log_err(f"Input file must be markdown: {input_file}")
            return 1

    # Parse all vidcapture markdown files
    documents = []
    for input_file in args.inputs:
        try:
            doc = parse_vidcapture_markdown(input_file)
            documents.append(doc)
        except (FileNotFoundError, ValueError) as e:
            log_err(str(e))
            return 1

    # Merge into single document
    document = merge_vidcapture_documents(documents)

    # Compute checkpoint path
    ckpt_path = VidscribeProcessor.checkpoint_path(args.inputs)

    # Dry run - show what would be processed
    if args.dry_run:
        ckpt_data = None
        if ckpt_path.exists():
            try:
                with open(ckpt_path, "r", encoding="utf-8") as f:
                    ckpt_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        if args.json_output:
            result = {
                "status": "dry_run",
                "inputs": [str(p.resolve()) for p in args.inputs],
                "sections": len(document.sections),
                "timestamps": [s.timestamp for s in document.sections],
            }
            missing = [
                str(s.image_path)
                for s in document.sections
                if not s.image_path.exists()
            ]
            if missing:
                result["missing_images"] = missing
            if ckpt_data:
                result["checkpoint"] = {
                    "completed_batches": ckpt_data.get("completed_batches", 0),
                    "completed_sections": len(ckpt_data.get("sections", [])),
                    "total_sections": ckpt_data.get("total_sections", 0),
                }
            print(json.dumps(result, indent=2))
        else:
            print(f"Would process {len(args.inputs)} file(s):")
            for p in args.inputs:
                print(f"  - {shorten_path(str(p.resolve()))}")
            print(f"Total sections: {len(document.sections)}")

            if ckpt_data:
                completed = ckpt_data.get("completed_batches", 0)
                total_s = ckpt_data.get("total_sections", 0)
                completed_s = len(ckpt_data.get("sections", []))
                print(
                    f"\nCheckpoint found: {completed} batches completed "
                    f"({completed_s}/{total_s} sections)"
                )
                print("Run without --dry-run to resume from checkpoint.")

            print("\nTimestamps:")
            for section in document.sections[:20]:
                exists = "+" if section.image_path.exists() else "x"
                print(
                    f"  [{exists}] {section.timestamp} - {shorten_path(str(section.image_path))}"
                )
            if len(document.sections) > 20:
                print(f"  ... and {len(document.sections) - 20} more")

            missing = sum(1 for s in document.sections if not s.image_path.exists())
            if missing:
                print(f"\n[Warning: {missing} images not found]")
        return 0

    # Load background context files if provided
    background_context = ""
    if args.context_files:
        try:
            background_context = load_context_files(args.context_files)
            log(f"Loaded {len(args.context_files)} context file(s)")
        except FileNotFoundError as e:
            log_err(str(e))
            return 1

    # Initialize processor
    try:
        processor = VidscribeProcessor(
            args.api_key,
            args.model,
            args.temperature,
            args.batch_size,
            args.context_frames,
            args.max_dimension,
            background_context,
            args.json_output,
            exa_api_key=exa_api_key,
        )
    except Exception as e:
        log_err(f"Initialization failed: {e}")
        return 1

    # Estimate tokens only
    if args.estimate_only:
        all_sections = document.sections
        num_batches = (len(all_sections) + args.batch_size - 1) // args.batch_size
        remaining_sections = all_sections
        completed_batches = 0

        ckpt_data = None
        if ckpt_path.exists():
            try:
                with open(ckpt_path, "r", encoding="utf-8") as f:
                    ckpt_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        if ckpt_data:
            completed_batches = ckpt_data.get("completed_batches", 0)
            completed_count = len(ckpt_data.get("sections", []))
            remaining_sections = all_sections[completed_count:]

        total_tokens = processor.estimate_tokens(remaining_sections)
        remaining_batches = num_batches - completed_batches

        if args.json_output:
            result = {
                "status": "estimate",
                "inputs": [str(p.resolve()) for p in args.inputs],
                "sections": len(all_sections),
                "estimated_tokens": total_tokens,
                "batches": num_batches,
                "sections_per_batch": args.batch_size,
            }
            if ckpt_data:
                result["checkpoint"] = {
                    "completed_batches": completed_batches,
                    "remaining_batches": remaining_batches,
                    "remaining_sections": len(remaining_sections),
                }
            print(json.dumps(result, indent=2))
        else:
            if ckpt_data:
                print(
                    f"Token estimation for {len(remaining_sections)} remaining sections "
                    f"({len(all_sections)} total, {completed_batches} batches checkpointed):"
                )
            else:
                print(f"Token estimation for {len(all_sections)} sections:")
            print(f"  Estimated input tokens: ~{total_tokens:,}")
            print(f"  Number of API batches: {remaining_batches}")
            print(f"  Sections per batch: {args.batch_size}")
            if total_tokens > 50000:
                print(
                    "[WARNING: Large batch may hit rate limits. "
                    "Consider reducing batch size.]"
                )
        return 0

    # Token estimation and confirmation
    if not args.yes and not args.json_output and len(document.sections) > 20:
        total_tokens = processor.estimate_tokens(document.sections)
        print(f"Estimated input tokens: ~{total_tokens:,}")
        num_batches = (len(document.sections) + args.batch_size - 1) // args.batch_size
        print(f"Number of API batches: {num_batches}")

        if total_tokens > 50000:
            print(
                f"\n[WARNING] This will use approximately {total_tokens:,} input tokens."
            )
            print("Large batches may hit API rate limits, causing delays.")
            response = input("Continue processing? (y/N): ").strip().lower()
            if response not in ["y", "yes"]:
                print("Processing cancelled.")
                return 0

    # Process the document
    import yaml

    input_desc = (
        args.inputs[0].name if len(args.inputs) == 1 else f"{len(args.inputs)} files"
    )
    log(f"Processing {len(document.sections)} sections from {input_desc}...")
    try:
        transcript, frontmatter = processor.process_all(
            document,
            checkpoint_path=ckpt_path,
            input_paths=args.inputs,
        )
    except Exception as e:
        log_err(f"Error processing document: {e}")
        return 1

    # Override title if provided
    if args.title:
        frontmatter["title"] = args.title
        if args.verbose:
            log(f"Using user-provided title: {args.title}")

    # Build complete document
    frontmatter_yaml = yaml.dump(
        frontmatter, default_flow_style=False, allow_unicode=True
    )
    full_document = (
        f"---\n{frontmatter_yaml}---\n\n# {frontmatter['title']}\n\n{transcript}"
    )

    # Determine output path
    output_path = determine_output_path(
        args.inputs[0], frontmatter["title"], args.output
    )

    # Check if output file exists (prompt only in interactive mode)
    if output_path.exists() and not args.json_output:
        result_path = handle_existing_output(output_path, args.inputs[0].parent)
        if result_path is None:
            print("Output cancelled.")
            return 0
        output_path = result_path

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_document)

    # Belt-and-suspenders checkpoint cleanup
    if ckpt_path.exists():
        ckpt_path.unlink()

    if args.json_output:
        result = {
            "status": "success",
            "sections_processed": len(document.sections),
            "output": str(output_path),
            "title": frontmatter["title"],
        }
        if args.context_files:
            result["context_files"] = [
                str(Path(f).resolve()) for f in args.context_files
            ]
        print(json.dumps(result, indent=2))
    else:
        print(f"\nSuccessfully processed {len(document.sections)} sections!")
        print(f"Output written to: {shorten_path(str(output_path))}")

    return 0


def vidscribe_entry():
    """Entry point for standalone vidscribe command."""
    sys.exit(main())


if __name__ == "__main__":
    vidscribe_entry()
