"""YouTube-specific transcript-aware parsing and transcription.

Bridges the gap between ytcapture output (which includes YouTube transcript
text per section) and vidscribe (which expects empty skeleton sections).

Provides:
- Transcript-preserving parser for ytcapture markdown
- Modified template construction with <existing-transcript> tags
- Adapted prompt for enhancing YouTube transcripts with visual context
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from rich.console import Console

from .cli_common import OperationResult


# Adapted prompt for YouTube sources with existing transcripts
YOUTUBE_TEMPLATE_FILL_PROMPT = """
<task>
You are enhancing a transcript for a YouTube video recording. Each timestamp section
contains an existing auto-generated YouTube transcript and a corresponding video frame image.
Your job is to produce a polished transcript that combines visual analysis with the existing
text, correcting errors, improving formatting, and adding visual context.
</task>

<instructions>
For each timestamp section in the template, produce TWO clearly separated output blocks:

### 1. Visual Content (slide/presentation description)

Describe ONLY relevant presented visual content: slide bullet points, titles, graph and plot
descriptions, diagrams, equations, tables, or other substantive displayed material. Write in
concise narrative form — key points and what figures convey, not a visual inventory.

- Do NOT describe faces, people, webcam feeds, or video call participant thumbnails.
- Do NOT describe screen chrome, window borders, recording indicators, or UI elements.
- If the frame shows nothing of visual interest (e.g., only a speaker's face or a blank
  screen), write "[No slide content visible]" and move on.

### 2. Speaker Text (enhanced transcript)

Use the existing YouTube transcript as a starting point and enhance it:

- Fix obvious speech-to-text errors using surrounding context and domain knowledge.
- Clean up filler words (um, uh, like, you know) and false starts.
- Merge sentence fragments into complete, flowing sentences.
- Add proper punctuation, capitalization, and paragraph breaks.
- Remove repeated words or stuttered phrases.
- Correct technical terminology that was misrecognized by YouTube's auto-captioning.
- Preserve the speaker's meaning and technical terminology faithfully.
- If the existing transcript is empty or clearly wrong, transcribe from the frame's
  live transcription sidebar (if visible) as a fallback.

When multiple speakers are identifiable (from visual cues or transcript context), provide
each speaker's text in a separate paragraph, starting with the speaker's name in bold:

**Speaker Name**: Their cleaned-up speech content continues here as a flowing paragraph...

**Other Speaker**: Their response follows in a new paragraph...

### Continuity

The output for each frame must flow coherently and continuously from the previous frame's
content provided in context. Do not repeat content already transcribed. Pick up where the
previous frame left off — continue mid-sentence if the prior context ended mid-thought.
When the same speaker continues across frames, do not re-introduce them; just continue
their text. Only start a new speaker paragraph when the speaker changes.
</instructions>

<output-format>
Output the completed template sections. Keep exact timestamp headings and image embeds.
After each image embed, on a new line, add the visual content description (if any), then
a blank line, then the enhanced speaker text. Use this structure:

## HH:MM:SS
![[image_embed]]

[Visual content description here, or "[No slide content visible]"]

**Speaker Name**: Enhanced transcript text here...
</output-format>
"""


@dataclass
class YouTubeTimestampSection:
    """A timestamp section from ytcapture markdown with preserved transcript text."""

    timestamp: str  # "00:05:30"
    image_embed: str  # "![[images/id/frame-0010.jpg]]"
    image_path: Path  # Resolved absolute path
    existing_transcript: str = ""  # YouTube transcript text for this section
    content: str = ""  # Filled by Claude


def parse_youtube_markdown(path: Path) -> tuple[str, str, list[YouTubeTimestampSection]]:
    """Parse a ytcapture markdown file, preserving existing transcript text.

    Unlike vidscribe's parse_vidcapture_markdown() which discards text after
    image embeds, this parser preserves it in each section's existing_transcript field.

    Args:
        path: Path to the ytcapture markdown file.

    Returns:
        Tuple of (frontmatter, title, sections).

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If no valid timestamp sections found.
    """
    from vidscribe import resolve_image_path

    if not path.exists():
        raise FileNotFoundError(f"Markdown file not found: {path}")

    content = path.read_text(encoding="utf-8")

    # Extract frontmatter
    frontmatter = ""
    title = ""
    body = content

    frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if frontmatter_match:
        frontmatter = frontmatter_match.group(1)
        body = content[frontmatter_match.end():]

        try:
            fm_data = yaml.safe_load(frontmatter)
            if isinstance(fm_data, dict):
                title = fm_data.get("title", "")
        except yaml.YAMLError:
            pass

    # If no title in frontmatter, look for H1 heading
    if not title:
        h1_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
        if h1_match:
            title = h1_match.group(1).strip()

    # Parse timestamp sections
    timestamp_pattern = re.compile(r"^##\s+(\d{2}:\d{2}:\d{2})\s*$", re.MULTILINE)
    image_embed_pattern = re.compile(r"!\[\[([^\]]+)\]\]")

    sections = []
    matches = list(timestamp_pattern.finditer(body))

    for i, match in enumerate(matches):
        timestamp = match.group(1)
        start_pos = match.end()

        # Section boundary
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(body)

        section_content = body[start_pos:end_pos].strip()

        # Find image embed
        image_match = image_embed_pattern.search(section_content)
        if image_match:
            image_embed = image_match.group(0)
            relative_path = image_match.group(1)
            image_path = resolve_image_path(path, relative_path)

            # Extract text after the image embed as existing transcript
            after_image = section_content[image_match.end():].strip()

            sections.append(
                YouTubeTimestampSection(
                    timestamp=timestamp,
                    image_embed=image_embed,
                    image_path=image_path,
                    existing_transcript=after_image,
                )
            )

    if not sections:
        raise ValueError(
            f"No valid timestamp sections found in {path}. "
            "Expected format: ## HH:MM:SS followed by ![[image_path]]"
        )

    return frontmatter, title, sections


def build_youtube_template(sections: list[YouTubeTimestampSection]) -> str:
    """Build a template-to-fill that includes existing transcript text.

    Unlike vidscribe's empty template, this includes <existing-transcript>
    XML tags per section so Claude can enhance rather than transcribe from scratch.
    """
    template = "\n<template-to-fill>\n"
    for sec in sections:
        template += f"## {sec.timestamp}\n{sec.image_embed}\n"
        if sec.existing_transcript:
            template += (
                f"<existing-transcript>\n"
                f"{sec.existing_transcript}\n"
                f"</existing-transcript>\n"
            )
        template += "\n"
    template += "</template-to-fill>"
    return template


def transcribe_youtube(
    input_path: Path,
    output: Optional[Path] = None,
    title: Optional[str] = None,
    context_files: Optional[list[Path]] = None,
    model: str = "claude-sonnet-4-20250514",
    batch_size: int = 10,
    context_frames: int = 3,
    temperature: float = 0.2,
    max_dimension: int = 1568,
    auto_confirm: bool = False,
    dry_run: bool = False,
    estimate_only: bool = False,
    json_output: bool = False,
) -> OperationResult:
    """Transcribe a YouTube-captured markdown file with transcript-aware processing.

    Uses the YouTube-adapted prompt and template construction to enhance
    existing YouTube transcripts with visual context from frame images.

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
    from vidscribe import (
        VidscribeProcessor,
        determine_output_path,
        load_context_files,
        sanitize_filename,
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
        # Parse with transcript preservation
        frontmatter, parsed_title, sections = parse_youtube_markdown(input_path)
        total_sections = len(sections)

        # Load context
        background_context = ""
        if context_files:
            background_context = load_context_files(context_files)

        # Initialize processor (we'll customize its batch processing)
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
            # Convert to vidscribe sections for estimation
            from vidscribe import TimestampSection

            vs_sections = [
                TimestampSection(
                    timestamp=s.timestamp,
                    image_embed=s.image_embed,
                    image_path=s.image_path,
                )
                for s in sections
            ]
            estimate = processor.estimate_tokens(vs_sections)
            return OperationResult(
                success=True,
                message=(
                    f"Estimated {estimate['total_tokens']:,} tokens "
                    f"for {total_sections} sections"
                ),
                data={"estimate": estimate, "sections": total_sections},
            )

        if dry_run:
            sections_with_transcript = sum(
                1 for s in sections if s.existing_transcript
            )
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

        # Process using YouTube-aware batch method
        results = process_youtube_batches(processor, sections)

        # Fill in section content
        for section, content in zip(sections, results):
            section.content = content

        # Build full transcript for frontmatter generation
        full_transcript = "\n\n".join(
            f"## {s.timestamp}\n{s.image_embed}\n{s.content}" for s in sections
        )

        # Generate or use provided title
        if title:
            frontmatter_data = {"title": title}
        else:
            frontmatter_data = processor.generate_frontmatter(full_transcript)
            title = frontmatter_data.get("title", parsed_title or "Untitled")

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
        final_md += full_transcript

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(final_md, encoding="utf-8")

        return OperationResult(
            success=True,
            message=f"Transcribed {total_sections} YouTube sections to {output_path}",
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
            message=f"YouTube transcription failed: {e}",
            errors=[str(e)],
        )


def process_youtube_batches(
    processor: "VidscribeProcessor",
    sections: list[YouTubeTimestampSection],
) -> list[str]:
    """Process YouTube sections in batches with transcript-aware templates.

    Reuses the processor's image handling and streaming infrastructure but
    customizes the prompt and template construction for YouTube sources.

    Args:
        processor: Initialized VidscribeProcessor instance.
        sections: YouTube timestamp sections with existing transcripts.

    Returns:
        List of transcribed content strings, one per section.
    """
    from vidscribe import TimestampSection

    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    results = [""] * len(sections)
    total_batches = (len(sections) + processor.batch_size - 1) // processor.batch_size

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=processor.console,
    ) as progress:
        overall_task = progress.add_task(
            "Processing YouTube sections", total=len(sections)
        )

        for batch_num in range(total_batches):
            start_idx = batch_num * processor.batch_size
            end_idx = min(start_idx + processor.batch_size, len(sections))
            batch_sections = sections[start_idx:end_idx]

            # Convert to vidscribe TimestampSections for image handling
            vs_sections = [
                TimestampSection(
                    timestamp=s.timestamp,
                    image_embed=s.image_embed,
                    image_path=s.image_path,
                )
                for s in batch_sections
            ]

            # Build image content blocks (reuse processor's infrastructure)
            content = processor._build_image_content(vs_sections)

            # Add previous context for continuity
            if start_idx > 0:
                context_start = max(0, start_idx - processor.context_frames)
                previous = sections[context_start:start_idx]
                if previous:
                    context_text = "<previous-transcription-context>\n"
                    context_text += "The following are the most recent transcribed sections. Use for continuity:\n\n"
                    for sec in previous:
                        context_text += f"## {sec.timestamp}\n{sec.image_embed}\n{sec.content}\n\n"
                    context_text += "</previous-transcription-context>\n\n"
                    content.append({"type": "text", "text": context_text})

            # Add YouTube-adapted prompt
            content.append({"type": "text", "text": YOUTUBE_TEMPLATE_FILL_PROMPT})

            # Build template with existing transcripts
            template_text = build_youtube_template(batch_sections)
            content.append({"type": "text", "text": template_text})

            # Make API request
            api_task = progress.add_task(
                f"Batch {batch_num + 1}/{total_batches} ({processor.model})",
                total=100,
            )

            messages = [{"role": "user", "content": content}]
            response_text, stop_reason, _ = processor._make_streaming_api_request(
                messages, api_task, progress
            )

            # Parse response into sections
            batch_results = processor._parse_batch_response(response_text, vs_sections)

            for i, result_content in enumerate(batch_results):
                results[start_idx + i] = result_content

            progress.update(overall_task, advance=len(batch_sections))
            progress.remove_task(api_task)

    return results
