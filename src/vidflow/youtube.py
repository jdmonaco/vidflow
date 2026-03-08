"""YouTube-specific transcription.

Now that vidscribe natively handles pre-existing transcript text via
TimestampSection.existing_text, this module is a thin wrapper that
parses, counts transcript sections, and delegates to VidscribeProcessor.
"""

import os
from pathlib import Path
from typing import Optional

import yaml

from .cli_common import OperationResult
from .models_config import DEFAULT_BATCH_SIZE, DEFAULT_CONTEXT_FRAMES, DEFAULT_MODEL, DEFAULT_TEMPERATURE


def transcribe_youtube(
    input_path: Path,
    output: Optional[Path] = None,
    title: Optional[str] = None,
    context_files: Optional[list[Path]] = None,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    context_frames: int = DEFAULT_CONTEXT_FRAMES,
    temperature: float = DEFAULT_TEMPERATURE,
    max_dimension: int = 1568,
    auto_confirm: bool = False,
    dry_run: bool = False,
    estimate_only: bool = False,
    json_output: bool = False,
) -> OperationResult:
    """Transcribe a YouTube-captured markdown file.

    Parses the ytcapture markdown (which includes YouTube auto-caption text
    per section) and delegates to VidscribeProcessor, which now natively
    handles existing transcript text via TimestampSection.existing_text.

    Args:
        input_path: Path to ytcapture markdown file.
        output: Explicit output path.
        title: Override title.
        context_files: Optional background context files.
        model: Claude model to use.
        batch_size: Frames per API batch.
        context_frames: Previous frames for continuity context.
        temperature: API temperature.
        max_dimension: Max image dimension.
        auto_confirm: Skip confirmation prompts.
        dry_run: Show what would be done.
        estimate_only: Only estimate tokens.
        json_output: JSON output mode.

    Returns:
        OperationResult with transcription results.
    """
    from vidflow.transcribe import (
        VidscribeProcessor,
        determine_output_path,
        load_context_files,
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
        # Parse with vidscribe's native parser (captures existing_text)
        document = parse_vidcapture_markdown(input_path)
        total_sections = len(document.sections)
        sections_with_transcript = sum(
            1 for s in document.sections if s.existing_text
        )

        # Load context
        background_context = ""
        if context_files:
            background_context = load_context_files(context_files)

        # Create standard processor (vidscribe handles existing text natively)
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
                message=(
                    f"Estimated {estimate:,} tokens "
                    f"for {total_sections} sections"
                ),
                data={"estimate": estimate, "sections": total_sections},
            )

        if dry_run:
            return OperationResult(
                success=True,
                message=(
                    f"Would transcribe {total_sections} sections "
                    f"({sections_with_transcript} with existing YouTube transcript)"
                ),
                data={
                    "sections": total_sections,
                    "sections_with_transcript": sections_with_transcript,
                    "input_file": str(input_path),
                    "model": model,
                    "batch_size": batch_size,
                },
            )

        # Delegate to vidscribe's process_all
        transcript_text, frontmatter_data = processor.process_all(document)

        # Use provided title or generated frontmatter title
        if title:
            frontmatter_data = {"title": title}
        else:
            title = frontmatter_data.get("title", document.title or "Untitled")

        # Determine output path
        output_path = determine_output_path(
            input_path=input_path.resolve(),
            title=title,
            explicit_output=output,
        )

        # Build final markdown
        fm_yaml = yaml.dump(
            frontmatter_data, default_flow_style=False, sort_keys=False
        ).strip()
        final_md = f"---\n{fm_yaml}\n---\n\n"
        final_md += f"# {title}\n\n"
        final_md += transcript_text

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(final_md, encoding="utf-8")

        return OperationResult(
            success=True,
            message=f"Transcribed {total_sections} YouTube sections to {output_path}",
            data={
                "output_path": str(output_path),
                "sections": total_sections,
                "sections_with_transcript": sections_with_transcript,
                "title": title,
                "model": model,
            },
        )

    except Exception as e:
        return OperationResult(
            success=False,
            message=f"YouTube transcription failed: {e}",
            errors=[str(e)],
        )
