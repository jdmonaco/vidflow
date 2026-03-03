"""Tests for bare YouTube video ID support."""

import pytest

from vidflow.capture.utils import is_video_id, video_id_to_url


class TestIsVideoId:
    """Tests for is_video_id()."""

    def test_standard_id(self):
        assert is_video_id("dQw4w9WgXcQ") is True

    def test_id_with_hyphens(self):
        assert is_video_id("abc-def_g12") is True

    def test_id_with_underscores(self):
        assert is_video_id("___________") is True

    def test_all_digits(self):
        assert is_video_id("12345678901") is True

    def test_too_short(self):
        assert is_video_id("abc123") is False

    def test_too_long(self):
        assert is_video_id("dQw4w9WgXcQx") is False

    def test_empty_string(self):
        assert is_video_id("") is False

    def test_url_not_id(self):
        assert is_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is False

    def test_invalid_chars(self):
        assert is_video_id("dQw4w9W.XcQ") is False
        assert is_video_id("dQw4w9W XcQ") is False
        assert is_video_id("dQw4w9W!XcQ") is False

    def test_mixed_case(self):
        assert is_video_id("AbCdEfGhIjK") is True


class TestVideoIdToUrl:
    """Tests for video_id_to_url()."""

    def test_converts_id_to_watch_url(self):
        assert video_id_to_url("dQw4w9WgXcQ") == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_roundtrip_with_extract(self):
        from vidflow.capture.utils import extract_video_id

        video_id = "dQw4w9WgXcQ"
        url = video_id_to_url(video_id)
        assert extract_video_id(url) == video_id
