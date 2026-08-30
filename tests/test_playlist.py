"""Tests for playlist URL handling: normalization, expansion, --no-playlist."""

import json
from unittest.mock import MagicMock, patch

import pytest

from vidflow.capture.video import VideoError, normalize_video_urls
from vidflow.cli import main as vidflow_main

WATCH = "https://www.youtube.com/watch?v=abcdefghijk"
PLAYLIST = "https://www.youtube.com/playlist?list=PLxyz123"
WATCH_WITH_LIST = "https://www.youtube.com/watch?v=abcdefghijk&list=PLxyz123"
EXPANDED = [
    "https://www.youtube.com/watch?v=aaaaaaaaaaa",
    "https://www.youtube.com/watch?v=bbbbbbbbbbb",
]


class TestNormalizeVideoUrls:
    """Tests for normalize_video_urls classification and expansion."""

    def test_video_url_passthrough(self):
        assert normalize_video_urls([WATCH]) == [WATCH]

    def test_bare_id_normalized(self):
        assert normalize_video_urls(["abcdefghijk"]) == [WATCH]

    @patch("vidflow.capture.video.expand_playlist", return_value=EXPANDED)
    def test_playlist_expanded(self, mock_expand):
        result = normalize_video_urls([PLAYLIST])
        assert result == EXPANDED
        mock_expand.assert_called_once_with(PLAYLIST)

    @patch("vidflow.capture.video.expand_playlist", side_effect=VideoError("private"))
    def test_failed_expansion_skipped(self, mock_expand):
        logs = []
        result = normalize_video_urls([PLAYLIST, WATCH], log=logs.append)
        assert result == [WATCH]
        assert any("Failed to expand playlist" in m for m in logs)

    def test_watch_with_list_param_passthrough(self):
        # Has a video id, so it is a video URL; --no-playlist pins yt-dlp
        assert normalize_video_urls([WATCH_WITH_LIST]) == [WATCH_WITH_LIST]

    def test_invalid_url_skipped(self):
        logs = []
        result = normalize_video_urls(["https://example.com/x", WATCH], log=logs.append)
        assert result == [WATCH]
        assert any("Skipping invalid URL" in m for m in logs)

    @patch("vidflow.capture.video.expand_playlist", return_value=[WATCH] + EXPANDED)
    def test_deduplication(self, mock_expand):
        result = normalize_video_urls([WATCH, PLAYLIST, "abcdefghijk"])
        assert result == [WATCH] + EXPANDED


class TestNoPlaylistFlag:
    """Single-video yt-dlp calls must pin to the video with --no-playlist."""

    def _fake_run(self, recorded):
        def fake(cmd, **kwargs):
            recorded.append(cmd)
            result = MagicMock()
            result.returncode = 0
            result.stdout = json.dumps(
                {"id": "abcdefghijk", "title": "T", "channel": "C", "duration": 10}
            )
            result.stderr = ""
            return result

        return fake

    def test_get_video_metadata(self):
        from vidflow.capture.video import get_video_metadata

        recorded = []
        with patch("vidflow.capture.video.subprocess.run", self._fake_run(recorded)):
            get_video_metadata(WATCH_WITH_LIST)
        assert "--no-playlist" in recorded[0]

    def test_download_video(self, tmp_path):
        from vidflow.capture.video import download_video

        recorded = []
        with patch("vidflow.capture.video.subprocess.run", self._fake_run(recorded)):
            with pytest.raises(VideoError):  # no file appears; flag still recorded
                download_video(WATCH_WITH_LIST, tmp_path)
        assert "--no-playlist" in recorded[0]

    def test_get_stream_url(self):
        from vidflow.capture.video import get_stream_url

        recorded = []
        with patch("vidflow.capture.video.subprocess.run", self._fake_run(recorded)):
            get_stream_url(WATCH)
        assert "--no-playlist" in recorded[0]


class TestVidflowPlaylistExpansion:
    """vidflow youtube expands playlist URLs before capturing."""

    def _capture_result(self):
        from vidflow.cli_common import OperationResult

        return OperationResult(
            success=True, message="Captured", data={"output_path": "/tmp/fake.md"}
        )

    @patch("vidflow.capture.capture_youtube")
    @patch("vidflow.capture.video.expand_playlist", return_value=EXPANDED)
    def test_playlist_arg_expanded(self, mock_expand, mock_capture):
        mock_capture.return_value = self._capture_result()

        assert vidflow_main(["youtube", PLAYLIST, "--json"]) == 0
        captured_urls = [c.kwargs["url"] for c in mock_capture.call_args_list]
        assert captured_urls == EXPANDED

    @patch("vidflow.capture.capture_youtube")
    @patch("vidflow.capture.video.expand_playlist", return_value=EXPANDED)
    @patch("vidflow.capture.cli.get_clipboard_urls")
    def test_clipboard_playlist_expanded(self, mock_clip, mock_expand, mock_capture):
        mock_clip.return_value = [PLAYLIST]
        mock_capture.return_value = self._capture_result()

        assert vidflow_main(["youtube", "--json"]) == 0
        captured_urls = [c.kwargs["url"] for c in mock_capture.call_args_list]
        assert captured_urls == EXPANDED

    def test_invalid_only_input_is_usage_error(self, capsys):
        assert vidflow_main(["youtube", "https://example.com/nope"]) == 2
