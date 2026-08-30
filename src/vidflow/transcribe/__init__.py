"""Transcription module for vidflow (formerly vidscribe).

Provides AI transcription of video frame snapshots from vidcapture markdown
files (local gateway by default, claude-* models via the Anthropic API), plus
a text-only polish mode that cleans up raw caption text without sending frames.
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
    DEFAULT_POLISH_BATCH_SIZE,
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
    POLISH_PROMPT,
    SUPPORTED_FORMATS,
    TEMPLATE_FILL_PROMPT,
)


def resolve_api_key(model, provider=None):
    """Resolve the Anthropic API key for the chosen lane.

    Returns (api_key, error_result). The key is only required when the
    model routes to the Anthropic API; the local gateway needs none.
    """
    import os

    import aikit

    from vidflow.cli_common import OperationResult

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if aikit.provider_for(model, provider) == "anthropic" and not api_key:
        return None, OperationResult(
            success=False,
            message="ANTHROPIC_API_KEY environment variable not set",
            errors=["ANTHROPIC_API_KEY is required for claude-* models"],
        )
    return api_key, None


def merge_frontmatter(original_yaml, generated):
    """Merge generated frontmatter over the capture's original frontmatter.

    Original keys (source, published, author, ...) are preserved; the
    generated keys win for the fields the generator owns (title, created,
    description), and tag lists are unioned. An empty or unparseable
    original yields the generated dict unchanged.
    """
    import yaml

    try:
        original = yaml.safe_load(original_yaml) if original_yaml else None
    except yaml.YAMLError:
        original = None
    if not isinstance(original, dict):
        return generated

    merged = {**original, **generated}
    orig_tags = original.get("tags")
    gen_tags = generated.get("tags")
    if isinstance(orig_tags, list) and isinstance(gen_tags, list):
        merged["tags"] = orig_tags + [t for t in gen_tags if t not in orig_tags]
    return merged


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
    provider=None,
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

    from vidflow.cli_common import OperationResult

    api_key, key_error = resolve_api_key(model, provider)
    if key_error:
        return key_error

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
            provider=provider,
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
        frontmatter_data = merge_frontmatter(document.frontmatter, frontmatter_data)

        if title:
            frontmatter_data["title"] = title
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


def polish_markdown(
    input_paths,
    output=None,
    title=None,
    context_files=None,
    model=DEFAULT_MODEL,
    batch_size=DEFAULT_POLISH_BATCH_SIZE,
    context_frames=DEFAULT_CONTEXT_FRAMES,
    temperature=DEFAULT_TEMPERATURE,
    provider=None,
    auto_confirm=False,
    dry_run=False,
    estimate_only=False,
    json_output=False,
):
    """Polish raw caption text in vidcapture markdown files (text-only).

    Sends the collated caption text — YouTube auto-captions or embedded
    subtitles — to the configured model for cleanup (speech-to-text errors,
    filler words, punctuation, paragraphing) without sending any frame
    images. Sections without caption text pass through unchanged.

    A single input with no explicit output is polished IN PLACE: the
    section text is replaced while the file's frontmatter, title, and any
    preamble (video embed, description) are preserved verbatim, and no
    frontmatter is generated. Passing -o/--output, or multiple inputs
    (always merged), writes a new file with generated frontmatter instead.
    """
    from vidflow.cli_common import OperationResult

    api_key, key_error = resolve_api_key(model, provider)
    if key_error:
        return key_error

    in_place = output is None and len(input_paths) == 1

    try:
        documents = [parse_vidcapture_markdown(p) for p in input_paths]
        document = merge_vidcapture_documents(documents)
        total_sections = len(document.sections)
        sections_with_text = sum(1 for s in document.sections if s.existing_text)

        if sections_with_text == 0:
            return OperationResult(
                success=False,
                message="No caption text found to polish (frames-only capture?)",
                errors=[
                    "Polish requires caption text in the capture markdown; "
                    "use transcribe for frames-only captures"
                ],
            )

        background_context = ""
        if context_files:
            background_context = load_context_files(context_files)

        processor = VidscribeProcessor(
            api_key=api_key,
            model=model,
            temperature=temperature,
            batch_size=batch_size,
            context_frames=context_frames,
            background_context=background_context,
            json_output=json_output,
            provider=provider,
            text_only=True,
        )

        if estimate_only:
            estimate = processor.estimate_tokens(document.sections)
            return OperationResult(
                success=True,
                message=f"Estimated {estimate:,} tokens for {total_sections} sections",
                data={"estimate": estimate, "sections": total_sections},
            )

        if dry_run:
            target = str(document.source_path) if in_place else "new output file"
            return OperationResult(
                success=True,
                message=(
                    f"Would polish {sections_with_text} of {total_sections} "
                    f"sections from {len(input_paths)} file(s) "
                    f"({'in place' if in_place else 'to a new file'})"
                ),
                data={
                    "sections": total_sections,
                    "sections_with_text": sections_with_text,
                    "input_files": [str(p) for p in input_paths],
                    "in_place": in_place,
                    "target": target,
                    "model": model,
                    "batch_size": batch_size,
                },
            )

        transcript_text, frontmatter_data = processor.process_all(
            document, with_frontmatter=not in_place
        )

        if in_place:
            import re
            import sys

            if title:
                print(
                    "Warning: -t/--title is ignored for in-place polish "
                    "(the file's own title is preserved); use -o to write "
                    "a new titled file",
                    file=sys.stderr,
                )

            original = document.source_path.read_text(encoding="utf-8")
            first_section = re.search(r"^##\s+\d{2}:\d{2}:\d{2}\s*$", original, re.MULTILINE)
            # The parser found sections, so the heading must be present
            prefix = original[: first_section.start()].rstrip()
            final_md = f"{prefix}\n\n{transcript_text}"
            document.source_path.write_text(final_md, encoding="utf-8")
            output_path = document.source_path
        else:
            frontmatter_data = merge_frontmatter(document.frontmatter, frontmatter_data)
            if title:
                frontmatter_data["title"] = title
            else:
                title = frontmatter_data.get("title", document.title or "Untitled")

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
            message=(
                f"Polished {sections_with_text} of {total_sections} sections "
                f"{'in place' if in_place else 'to new file'}: {output_path}"
            ),
            data={
                "output_path": str(output_path),
                "sections": total_sections,
                "sections_with_text": sections_with_text,
                "in_place": in_place,
                "model": model,
            },
        )

    except Exception as e:
        return OperationResult(
            success=False,
            message=f"Polish failed: {e}",
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
    "POLISH_PROMPT",
    "SUPPORTED_FORMATS",
    "TEMPLATE_FILL_PROMPT",
    # Convenience wrappers
    "resolve_api_key",
    "merge_frontmatter",
    "transcribe_markdown",
    "polish_markdown",
]
