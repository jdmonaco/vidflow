"""Transcription module for vidflow (formerly vidscribe).

Provides Claude Vision-based transcription of video frame snapshots
from vidcapture markdown files.
"""

from vidflow.transcribe.models import TimestampSection, VidcaptureDocument
from vidflow.transcribe.parser import (
    merge_vidcapture_documents,
    parse_vidcapture_markdown,
    resolve_image_path,
)
from vidflow.transcribe.processor import VidscribeProcessor
from vidflow.transcribe.image import (
    find_magick_command,
    get_image_dimensions,
    resize_image,
)
from vidflow.transcribe.output import (
    determine_output_path,
    handle_existing_output,
    load_context_files,
    sanitize_filename,
    shorten_path,
)
from vidflow.models_config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONTEXT_FRAMES,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
)
from vidflow.transcribe.prompts import (
    CITATION_SEARCH_PROMPT,
    DEFAULT_MAX_DIMENSION,
    EXA_SEARCH_TOOL,
    FRONTMATTER_PROMPT,
    MAX_REQUEST_SIZE_BYTES,
    MAX_REQUEST_SIZE_MB,
    MAX_TOOL_CALLS_PER_BATCH,
    SUPPORTED_FORMATS,
    TEMPLATE_FILL_PROMPT,
)


def transcribe_markdown(
    input_paths,
    output=None,
    title=None,
    context_files=None,
    model=DEFAULT_MODEL,
    batch_size=DEFAULT_BATCH_SIZE,
    context_frames=DEFAULT_CONTEXT_FRAMES,
    temperature=DEFAULT_TEMPERATURE,
    max_dimension=DEFAULT_MAX_DIMENSION,
    auto_confirm=False,
    dry_run=False,
    estimate_only=False,
    json_output=False,
):
    """Transcribe vidcapture markdown files with OperationResult output.

    Multiple inputs are merged into a single output.
    Used by the vidflow CLI layer.
    """
    import os
    from pathlib import Path

    from vidflow.cli_common import OperationResult

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return OperationResult(
            success=False,
            message="ANTHROPIC_API_KEY environment variable not set",
            errors=["ANTHROPIC_API_KEY is required for transcription"],
        )

    exa_api_key = os.environ.get("EXA_API_KEY")

    try:
        documents = [parse_vidcapture_markdown(p) for p in input_paths]
        document = merge_vidcapture_documents(documents)
        total_sections = len(document.sections)

        background_context = ""
        if context_files:
            background_context = load_context_files(context_files)

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
                message=f"Estimated {estimate:,} tokens for {total_sections} sections",
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

        transcript_text, frontmatter_data = processor.process_all(document)

        if title:
            frontmatter_data = {"title": title}
        else:
            title = frontmatter_data.get("title", "Untitled")

        output_path = determine_output_path(
            input_path=document.source_path,
            title=title,
            explicit_output=output,
        )

        import yaml

        fm_yaml = yaml.dump(frontmatter_data, default_flow_style=False, sort_keys=False).strip()
        final_md = f"---\n{fm_yaml}\n---\n\n"
        final_md += f"# {title}\n\n"
        final_md += transcript_text

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


__all__ = [
    # Models
    "TimestampSection",
    "VidcaptureDocument",
    # Parser
    "parse_vidcapture_markdown",
    "merge_vidcapture_documents",
    "resolve_image_path",
    # Processor
    "VidscribeProcessor",
    # Image
    "find_magick_command",
    "get_image_dimensions",
    "resize_image",
    # Output
    "determine_output_path",
    "handle_existing_output",
    "load_context_files",
    "sanitize_filename",
    "shorten_path",
    # Prompts/Constants
    "CITATION_SEARCH_PROMPT",
    "DEFAULT_MAX_DIMENSION",
    "EXA_SEARCH_TOOL",
    "FRONTMATTER_PROMPT",
    "MAX_REQUEST_SIZE_BYTES",
    "MAX_REQUEST_SIZE_MB",
    "MAX_TOOL_CALLS_PER_BATCH",
    "SUPPORTED_FORMATS",
    "TEMPLATE_FILL_PROMPT",
    # Convenience wrapper
    "transcribe_markdown",
]
