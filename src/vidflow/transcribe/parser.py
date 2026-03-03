"""Markdown parsing for vidcapture files."""

import re
from pathlib import Path
from typing import List

import yaml

from vidflow.transcribe.models import TimestampSection, VidcaptureDocument


def resolve_image_path(markdown_path: Path, relative_path: str) -> Path:
    """Resolve an image path from a markdown embed.

    Tries multiple resolution strategies:
    1. File-relative (path relative to markdown file)
    2. Images-suffix (extract just images/... portion)
    3. Vault-relative (path from detected vault root)

    Args:
        markdown_path: Path to the markdown file containing the embed
        relative_path: The path extracted from ![[path]] embed

    Returns:
        Resolved absolute path to the image
    """
    md_dir = markdown_path.parent

    # Strategy 1: File-relative (most common)
    file_relative = (md_dir / relative_path).resolve()
    if file_relative.exists():
        return file_relative

    # Strategy 2: Extract images/... suffix and try file-relative
    # Handles malformed paths like "Areas/.../images/foo.jpg" -> "images/foo.jpg"
    rel_path = Path(relative_path)
    rel_parts = rel_path.parts
    if "images" in rel_parts:
        images_idx = rel_parts.index("images")
        images_suffix = Path(*rel_parts[images_idx:])
        images_relative = (md_dir / images_suffix).resolve()
        if images_relative.exists():
            return images_relative

    # Strategy 3: Vault-relative (find common ancestor)
    md_parts = md_dir.parts
    if rel_parts:
        for i, part in enumerate(md_parts):
            if part == rel_parts[0]:
                ancestor = Path(*md_parts[:i])
                vault_relative = (ancestor / relative_path).resolve()
                if vault_relative.exists():
                    return vault_relative

    # Fallback: return file-relative (will show as missing)
    return file_relative


def parse_vidcapture_markdown(path: Path) -> VidcaptureDocument:
    """Parse a vidcapture markdown file into structured data.

    Args:
        path: Path to the vidcapture markdown file

    Returns:
        VidcaptureDocument with parsed sections

    Raises:
        ValueError: If the file format is invalid
    """
    if not path.exists():
        raise FileNotFoundError(f"Markdown file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract frontmatter (between --- delimiters)
    frontmatter = ""
    title = ""
    body = content

    frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if frontmatter_match:
        frontmatter = frontmatter_match.group(1)
        body = content[frontmatter_match.end():]

        # Try to extract title from frontmatter
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

    # Parse timestamp sections: ## HH:MM:SS
    timestamp_pattern = re.compile(r"^##\s+(\d{2}:\d{2}:\d{2})\s*$", re.MULTILINE)
    # Image embed pattern: ![[path/to/image.ext]]
    image_embed_pattern = re.compile(r"!\[\[([^\]]+)\]\]")

    sections = []
    matches = list(timestamp_pattern.finditer(body))

    for i, match in enumerate(matches):
        timestamp = match.group(1)
        start_pos = match.end()

        # Find the end of this section (start of next section or end of file)
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(body)

        section_content = body[start_pos:end_pos].strip()

        # Find image embed in this section
        image_match = image_embed_pattern.search(section_content)
        if image_match:
            image_embed = image_match.group(0)
            relative_path = image_match.group(1)

            # Resolve image path (handles both file-relative and vault-relative)
            image_path = resolve_image_path(path, relative_path)

            # Capture any text after the image embed as existing transcript
            after_image = section_content[image_match.end():].strip()

            sections.append(
                TimestampSection(
                    timestamp=timestamp,
                    image_embed=image_embed,
                    image_path=image_path,
                    existing_text=after_image,
                )
            )

    if not sections:
        raise ValueError(
            f"No valid timestamp sections found in {path}. "
            "Expected format: ## HH:MM:SS followed by ![[image_path]]"
        )

    return VidcaptureDocument(
        source_path=path.resolve(),
        frontmatter=frontmatter,
        title=title,
        sections=sections,
    )


def merge_vidcapture_documents(
    documents: List[VidcaptureDocument],
) -> VidcaptureDocument:
    """Merge multiple vidcapture documents into one.

    Concatenates sections in order. Uses first document's source_path for output directory.
    Title is left empty for auto-generation based on full content.
    """
    if len(documents) == 1:
        return documents[0]

    all_sections = []
    for doc in documents:
        all_sections.extend(doc.sections)

    return VidcaptureDocument(
        source_path=documents[0].source_path,  # For output directory
        frontmatter="",  # Will be regenerated
        title="",  # Will be auto-generated
        sections=all_sections,
    )
