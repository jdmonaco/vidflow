"""Output path determination and file utilities."""

import re
from pathlib import Path
from typing import Optional


def shorten_path(path: str) -> str:
    """Shorten paths for display: $HOME -> ~, OneDrive CloudStorage -> ~/OneDrive."""
    home = str(Path.home())
    onedrive_prefix = f"{home}/Library/CloudStorage/OneDrive-"
    if path.startswith(onedrive_prefix):
        rest = path[len(onedrive_prefix):]
        if "/" in rest:
            _, subpath = rest.split("/", 1)
            return f"~/OneDrive/{subpath}"
        return f"~/OneDrive/{rest}"
    if path.startswith(home + "/"):
        return "~" + path[len(home):]
    elif path == home:
        return "~"
    return path


def sanitize_filename(title: str) -> str:
    """Sanitize a title for use as a filename, keeping spaces and hyphens."""
    # Remove special characters and punctuation except hyphens and spaces
    sanitized = re.sub(r"[^\w\s-]", "", title)
    # Clean up multiple spaces
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized


def load_context_files(context_paths: list[Path]) -> str:
    """Load and concatenate context files for background information."""
    context_parts = []
    for path in context_paths:
        if not path.exists():
            raise FileNotFoundError(f"Context file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            context_parts.append(f"### {path.name}\n\n{f.read().strip()}")
    return "\n\n".join(context_parts)


def handle_existing_output(output_path: Path, input_dir: Path) -> Optional[Path]:
    """Handle case when output file already exists.

    Args:
        output_path: The intended output path that already exists
        input_dir: Directory for renamed files

    Returns:
        Path to use (same or renamed), or None to abort
    """
    while True:
        response = (
            input(f"\nFile exists: {output_path.name}\nOverwrite? [y/N/r(ename)]: ")
            .strip()
            .lower()
        )
        if response in ["y", "yes"]:
            return output_path
        elif response in ["n", "no", ""]:
            return None
        elif response in ["r", "rename"]:
            new_name = input("New filename: ").strip()
            if new_name:
                if not new_name.endswith(".md"):
                    new_name += ".md"
                return input_dir / new_name
        else:
            print("Please enter 'y' (overwrite), 'n' (abort), or 'r' (rename)")


def determine_output_path(
    input_path: Path, title: str, explicit_output: Optional[Path] = None
) -> Path:
    """Determine the output path for the final transcript.

    Args:
        input_path: Original vidcapture markdown file path
        title: Generated or provided title
        explicit_output: User-provided output path (overrides auto-generation)

    Returns:
        Path for the output file
    """
    if explicit_output:
        return explicit_output.resolve()

    # Auto-generate from title in same directory as input
    sanitized = sanitize_filename(title)
    return (input_path.parent / f"{sanitized}.md").resolve()
