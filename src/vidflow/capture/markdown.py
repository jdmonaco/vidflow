"""Markdown generation for Obsidian output."""

from datetime import datetime
from pathlib import Path

import yaml

from vidflow.capture.frames import FrameInfo
from vidflow.capture.local import LocalVideoMetadata
from vidflow.capture.metadata import VideoMetadataProtocol
from vidflow.capture.transcript import TranscriptSegment
from vidflow.capture.utils import format_date, format_timestamp, sanitize_title, truncate_title_words


def align_transcript_to_frames(
    transcript: list[TranscriptSegment] | None,
    frames: list[FrameInfo],
) -> list[tuple[FrameInfo, list[TranscriptSegment]]]:
    """Group transcript segments under the closest preceding frame."""
    if not frames:
        return []

    if not transcript:
        return [(frame, []) for frame in frames]

    grouped: list[tuple[FrameInfo, list[TranscriptSegment]]] = []

    for i, frame in enumerate(frames):
        frame_start = frame.timestamp
        if i + 1 < len(frames):
            frame_end = frames[i + 1].timestamp
        else:
            frame_end = float('inf')

        segments = [
            s for s in transcript
            if frame_start <= s.start < frame_end
        ]

        grouped.append((frame, segments))

    return grouped


def generate_frontmatter(
    metadata: VideoMetadataProtocol,
    url: str | None = None,
) -> str:
    """Generate YAML frontmatter for Obsidian."""
    frontmatter: dict[str, str | list[str]] = {
        'title': metadata.title,
        'created': datetime.now().strftime('%Y-%m-%d'),
        'published': format_date(metadata.source_date),
        'tags': [metadata.source_type],
    }

    if url:
        frontmatter['source'] = url

    if metadata.author:
        frontmatter['author'] = [metadata.author]

    if hasattr(metadata, '_original_title') and metadata._original_title and metadata._original_title != metadata.title:
        frontmatter['original_title'] = metadata._original_title

    if metadata.description:
        description = metadata.description
        if len(description) > 200:
            description = description[:197] + '...'
        frontmatter['description'] = description

    frontmatter = {k: v for k, v in frontmatter.items() if v}

    yaml_str = yaml.dump(
        frontmatter,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )

    return f'---\n{yaml_str}---\n'


def generate_markdown_body(
    grouped_data: list[tuple[FrameInfo, list[TranscriptSegment]]],
    identifier: str,
) -> str:
    """Generate markdown body with embedded frames and transcript."""
    sections = []

    for frame, segments in grouped_data:
        timestamp_str = format_timestamp(frame.timestamp)
        section = f'\n## {timestamp_str}\n\n'

        relative_path = f'images/{identifier}/{frame.path.name}'
        section += f'![[{relative_path}]]\n\n'

        if segments:
            text = ' '.join(s.text for s in segments)
            section += f'{text}\n'

        sections.append(section)

    return ''.join(sections)


def generate_frames_only(
    frames: list[FrameInfo],
    identifier: str,
) -> str:
    """Generate markdown body with frames only (no transcript)."""
    sections = []

    for frame in frames:
        timestamp_str = format_timestamp(frame.timestamp)
        relative_path = f'images/{identifier}/{frame.path.name}'

        section = f'\n## {timestamp_str}\n\n![[{relative_path}]]\n'
        sections.append(section)

    return ''.join(sections)


def generate_markdown_filename(metadata: VideoMetadataProtocol) -> str:
    """Generate the markdown filename from video metadata."""
    short_title = sanitize_title(truncate_title_words(metadata.title, 10))
    date_str = metadata.source_date

    if metadata.author:
        author = sanitize_title(metadata.author)
        return f'{short_title} ({author}) {date_str}.md'
    else:
        return f'{short_title} {date_str}.md'


def generate_local_markdown_filename(metadata: LocalVideoMetadata) -> str:
    """Generate markdown filename for local video files."""
    stem = metadata.file_path.stem
    if metadata._identifier_suffix > 0:
        return f'{stem}-{metadata._identifier_suffix}.md'
    return f'{stem}.md'


def generate_markdown_file(
    metadata: VideoMetadataProtocol,
    url: str | None,
    transcript: list[TranscriptSegment] | None,
    frames: list[FrameInfo],
    output_dir: Path,
    video_path: Path | None = None,
    filename: str | None = None,
) -> Path:
    """Generate complete markdown file."""
    identifier = metadata.identifier

    frontmatter = generate_frontmatter(metadata, url)

    if transcript and frames:
        grouped = align_transcript_to_frames(transcript, frames)
        body = generate_markdown_body(grouped, identifier)
    elif frames:
        body = generate_frames_only(frames, identifier)
    else:
        body = '\n*No frames or transcript available.*\n'

    video_embed = ''
    if video_path and video_path.exists():
        relative_video_path = f'videos/{video_path.name}'
        video_embed = f'\n<video src="{relative_video_path}" controls width="100%"></video>\n'

    description_section = ''
    if metadata.description:
        first_para = metadata.description.strip().split('\n\n')[0]
        desc_lines = first_para.strip().split('\n')
        desc_blockquote = '\n'.join(f'> {line}' for line in desc_lines if line.strip())
        if desc_blockquote:
            description_section = f'\n{desc_blockquote}\n'

    title_heading = f'\n# {metadata.title}\n'
    content = frontmatter + title_heading + video_embed + description_section + body

    if filename is None:
        filename = generate_markdown_filename(metadata)

    filepath = output_dir / filename
    filepath.write_text(content, encoding='utf-8')

    return filepath
