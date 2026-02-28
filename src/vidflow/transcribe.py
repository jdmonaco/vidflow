"""Wrapper around vidscribe VidscribeProcessor for standard (skeleton) input.

Handles transcription of vidcapture markdown files that have empty
timestamp sections (no pre-existing transcript text).
"""

import os
from pathlib import Path
from typing import Optional

from .cli_common import OperationResult


DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_BATCH_SIZE = 10
DEFAULT_CONTEXT_FRAMES = 3
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_DIMENSION = 1568


def transcribe_markdown(
    input_paths: list[Path],
    output: Optional[Path] = None,
    title: Optional[str] = None,
    context_files: Optional[list[Path]] = None,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    context_frames: int = DEFAULT_CONTEXT_FRAMES,
    temperature: float = DEFAULT_TEMPERATURE,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    auto_confirm: bool = False,
    dry_run: bool = False,
    estimate_only: bool = False,
    json_output: bool = False,
) -> OperationResult:
    """Transcribe one or more vidcapture markdown files.

    Multiple inputs are merged into a single output (matching vidscribe behavior).

    Args:
        input_paths: Paths to vidcapture markdown files.
        output: Explicit output path (auto-generated from title if None).
        title: Override title (auto-generated from content if None).
        context_files: Optional background context files.
        model: Claude model to use.
        batch_size: Frames per API batch.
        context_frames: Number of previous frames for continuity context.
        temperature: API temperature.
        max_dimension: Max image dimension for resizing.
        auto_confirm: Skip confirmation prompts.
        dry_run: Show what would be done without processing.
        estimate_only: Only estimate token usage.
        json_output: Output JSON instead of human-readable messages.

    Returns:
        OperationResult with transcription results.
    """
    from vidscribe import (
        VidscribeProcessor,
        determine_output_path,
        load_context_files,
        merge_vidcapture_documents,
        parse_vidcapture_markdown,
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return OperationResult(
            success=False,
            message="ANTHROPIC_API_KEY environment variable not set",
            errors=["ANTHROPIC_API_KEY is required for transcription"],
        )

    exa_api_key = os.environ.get("EXA_API_KEY")

    try:
        # Parse input documents
        documents = [parse_vidcapture_markdown(p) for p in input_paths]

        # Merge if multiple inputs
        document = merge_vidcapture_documents(documents)

        total_sections = len(document.sections)

        # Load context
        background_context = ""
        if context_files:
            background_context = load_context_files(context_files)

        # Initialize processor
        processor = VidscribeProcessor(
            api_key=api_key,
            model=model,
            temperature=temperature,
            batch_size=batch_size,
            context_frames=context_frames,
            max_dimension=max_dimension,
            background_context=background_context,
            json_output=json_output,
            exa_api_key=exa_api_key,
        )

        if estimate_only:
            estimate = processor.estimate_tokens(document.sections)
            return OperationResult(
                success=True,
                message=f"Estimated {estimate['total_tokens']:,} tokens for {total_sections} sections",
                data={"estimate": estimate, "sections": total_sections},
            )

        if dry_run:
            return OperationResult(
                success=True,
                message=f"Would transcribe {total_sections} sections from {len(input_paths)} file(s)",
                data={
                    "sections": total_sections,
                    "input_files": [str(p) for p in input_paths],
                    "model": model,
                    "batch_size": batch_size,
                },
            )

        # Process all sections
        results = processor.process_all_sections(document.sections)

        # Fill in section content
        for section, content in zip(document.sections, results):
            section.content = content

        # Generate frontmatter (title auto-generation)
        full_transcript = "\n\n".join(
            f"## {s.timestamp}\n{s.image_embed}\n{s.content}" for s in document.sections
        )

        if title:
            frontmatter_data = {"title": title}
        else:
            frontmatter_data = processor.generate_frontmatter(full_transcript)
            title = frontmatter_data.get("title", "Untitled")

        # Determine output path
        output_path = determine_output_path(
            input_path=document.source_path,
            title=title,
            explicit_output=output,
        )

        # Build final markdown
        import yaml

        fm_yaml = yaml.dump(frontmatter_data, default_flow_style=False, sort_keys=False).strip()
        final_md = f"---\n{fm_yaml}\n---\n\n"
        final_md += f"# {title}\n\n"
        final_md += full_transcript

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(final_md, encoding="utf-8")

        return OperationResult(
            success=True,
            message=f"Transcribed {total_sections} sections to {output_path}",
            data={
                "output_path": str(output_path),
                "sections": total_sections,
                "title": title,
                "model": model,
            },
        )

    except Exception as e:
        return OperationResult(
            success=False,
            message=f"Transcription failed: {e}",
            errors=[str(e)],
        )
