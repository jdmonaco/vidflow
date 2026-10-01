"""ImageMagick operations for image resizing."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Tuple

from PIL import Image


def find_magick_command() -> str:
    """Find the ImageMagick command with fallback paths."""
    # Try 'magick' first (ImageMagick 7)
    magick_cmd = shutil.which("magick")
    if magick_cmd:
        return magick_cmd

    # Try 'convert' (ImageMagick 6 or legacy)
    convert_cmd = shutil.which("convert")
    if convert_cmd:
        return convert_cmd

    # Check common locations
    possible_paths = [
        "/opt/homebrew/bin/magick",
        "/usr/local/bin/magick",
        "/opt/homebrew/bin/convert",
        "/usr/local/bin/convert",
        "/usr/bin/convert",
    ]

    for path in possible_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path

    raise RuntimeError(
        "ImageMagick not found. Please install it:\n"
        "  macOS: brew install imagemagick\n"
        "  Ubuntu/Debian: sudo apt install imagemagick"
    )


def get_image_dimensions(image_path: Path) -> Tuple[int, int]:
    """Get image dimensions using Pillow."""
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def resize_image(src: Path, dst: Path, max_dim: int, magick_cmd: str) -> bool:
    """Resize image if needed, preserving aspect ratio, as 3-channel sRGB.

    ImageMagick writes a grayscale JPEG when a frame has no colour content
    (a black or white-only slide). Vision backends that stack a batch's
    image tensors then fail on the channel mismatch ("stack expects each
    tensor to be equal size"), so every prepared image is written as
    TrueColor sRGB whether or not it is resized.

    Returns True if the image was resized.
    """
    width, height = get_image_dimensions(src)
    resized = max(width, height) > max_dim

    cmd = [
        magick_cmd,
        str(src),
        "-resize",
        f"{max_dim}x{max_dim}>",  # shrink only when larger
        "-colorspace",
        "sRGB",
        "-type",
        "TrueColor",
        str(dst),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return resized
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to resize image: {e.stderr.decode()}")
