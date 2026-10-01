"""Prepared frames are always 3-channel sRGB, whatever their content."""

import shutil

import pytest
from PIL import Image

from vidflow.transcribe.image import find_magick_command, resize_image

pytestmark = pytest.mark.skipif(
    shutil.which("magick") is None and shutil.which("convert") is None,
    reason="ImageMagick not installed",
)


@pytest.fixture
def magick():
    return find_magick_command()


def _black_jpeg(path, size):
    Image.new("RGB", size, (0, 0, 0)).save(path, "JPEG")
    return path


def test_black_frame_stays_rgb_when_resized(tmp_path, magick):
    src = _black_jpeg(tmp_path / "frame.jpg", (1920, 1080))
    dst = tmp_path / "prepared.jpg"

    assert resize_image(src, dst, 1568, magick) is True
    with Image.open(dst) as img:
        assert img.mode == "RGB"
        assert max(img.size) == 1568


def test_black_frame_stays_rgb_when_not_resized(tmp_path, magick):
    src = _black_jpeg(tmp_path / "frame.jpg", (800, 450))
    dst = tmp_path / "prepared.jpg"

    assert resize_image(src, dst, 1568, magick) is False
    with Image.open(dst) as img:
        assert img.mode == "RGB"
        assert img.size == (800, 450)
