"""Tests for clipboard URL extraction and confirmation."""

from unittest.mock import MagicMock, patch

import pytest

from vidflow.capture.utils import extract_youtube_urls


class TestExtractYoutubeUrls:
    """Tests for extract_youtube_urls()."""

    def test_plain_urls(self):
        text = "https://www.youtube.com/watch?v=abc123\n" "https://youtu.be/def456\n"
        result = extract_youtube_urls(text)
        assert result == [
            "https://www.youtube.com/watch?v=abc123",
            "https://youtu.be/def456",
        ]

    def test_markdown_links(self):
        text = (
            "[Video Title](https://www.youtube.com/watch?v=abc123)\n"
            "[Another](https://youtu.be/def456)\n"
        )
        result = extract_youtube_urls(text)
        assert result == [
            "https://www.youtube.com/watch?v=abc123",
            "https://youtu.be/def456",
        ]

    def test_csv_format(self):
        text = (
            "Title,URL\n"
            "My Video,https://www.youtube.com/watch?v=abc123\n"
            "Other,https://youtu.be/def456\n"
        )
        result = extract_youtube_urls(text)
        assert result == [
            "https://www.youtube.com/watch?v=abc123",
            "https://youtu.be/def456",
        ]

    def test_mixed_text(self):
        text = (
            "Check out this video: https://www.youtube.com/watch?v=abc123 and also\n"
            "this one [here](https://youtu.be/def456) for more info.\n"
        )
        result = extract_youtube_urls(text)
        assert result == [
            "https://www.youtube.com/watch?v=abc123",
            "https://youtu.be/def456",
        ]

    def test_deduplication(self):
        text = (
            "https://www.youtube.com/watch?v=abc123\n"
            "https://www.youtube.com/watch?v=abc123\n"
            "https://youtu.be/def456\n"
        )
        result = extract_youtube_urls(text)
        assert result == [
            "https://www.youtube.com/watch?v=abc123",
            "https://youtu.be/def456",
        ]

    def test_no_urls(self):
        assert extract_youtube_urls("no urls here") == []
        assert extract_youtube_urls("") == []

    def test_playlist_urls(self):
        text = "https://www.youtube.com/playlist?list=PLabc123"
        result = extract_youtube_urls(text)
        assert result == ["https://www.youtube.com/playlist?list=PLabc123"]

    def test_urls_with_tracking_params(self):
        text = "https://www.youtube.com/watch?v=abc123&t=120&si=xyz"
        result = extract_youtube_urls(text)
        assert len(result) == 1
        assert "abc123" in result[0]

    def test_non_youtube_urls_ignored(self):
        text = (
            "https://vimeo.com/12345\n"
            "https://www.youtube.com/watch?v=abc123\n"
            "https://example.com/video\n"
        )
        result = extract_youtube_urls(text)
        assert result == ["https://www.youtube.com/watch?v=abc123"]

    def test_embed_url(self):
        text = "https://www.youtube.com/embed/abc123"
        result = extract_youtube_urls(text)
        assert result == ["https://www.youtube.com/embed/abc123"]

    def test_mobile_url(self):
        text = "https://m.youtube.com/watch?v=abc123"
        result = extract_youtube_urls(text)
        assert result == ["https://m.youtube.com/watch?v=abc123"]

    def test_url_in_angle_brackets(self):
        text = "<https://www.youtube.com/watch?v=abc123>"
        result = extract_youtube_urls(text)
        assert result == ["https://www.youtube.com/watch?v=abc123"]

    def test_single_url_no_newline(self):
        text = "https://www.youtube.com/watch?v=abc123"
        result = extract_youtube_urls(text)
        assert result == ["https://www.youtube.com/watch?v=abc123"]


class TestGetClipboardUrls:
    """Tests for get_clipboard_urls()."""

    @patch("vidflow.capture.cli.platform.system", return_value="Darwin")
    @patch("vidflow.capture.cli.shutil.which", return_value="/usr/bin/pbpaste")
    @patch("vidflow.capture.cli.subprocess.run")
    def test_multi_url_clipboard(self, mock_run, mock_which, mock_system):
        from vidflow.capture.cli import get_clipboard_urls

        mock_run.return_value = MagicMock(
            stdout=("https://www.youtube.com/watch?v=abc123\n" "https://youtu.be/def456\n"),
        )
        result = get_clipboard_urls()
        assert result == [
            "https://www.youtube.com/watch?v=abc123",
            "https://youtu.be/def456",
        ]

    @patch("vidflow.capture.cli.platform.system", return_value="Darwin")
    @patch("vidflow.capture.cli.shutil.which", return_value="/usr/bin/pbpaste")
    @patch("vidflow.capture.cli.subprocess.run")
    def test_empty_clipboard(self, mock_run, mock_which, mock_system):
        from vidflow.capture.cli import get_clipboard_urls

        mock_run.return_value = MagicMock(stdout="")
        result = get_clipboard_urls()
        assert result == []

    @patch("vidflow.capture.cli.platform.system", return_value="Darwin")
    @patch("vidflow.capture.cli.shutil.which", return_value="/usr/bin/pbpaste")
    @patch("vidflow.capture.cli.subprocess.run")
    def test_no_youtube_urls(self, mock_run, mock_which, mock_system):
        from vidflow.capture.cli import get_clipboard_urls

        mock_run.return_value = MagicMock(stdout="just some text")
        result = get_clipboard_urls()
        assert result == []

    @patch("vidflow.capture.cli.platform.system", return_value="Linux")
    def test_non_macos(self, mock_system):
        from vidflow.capture.cli import get_clipboard_urls

        result = get_clipboard_urls()
        assert result == []

    @patch("vidflow.capture.cli.platform.system", return_value="Darwin")
    @patch("vidflow.capture.cli.shutil.which", return_value=None)
    def test_no_pbpaste(self, mock_which, mock_system):
        from vidflow.capture.cli import get_clipboard_urls

        result = get_clipboard_urls()
        assert result == []

    @patch("vidflow.capture.cli.platform.system", return_value="Darwin")
    @patch("vidflow.capture.cli.shutil.which", return_value="/usr/bin/pbpaste")
    @patch("vidflow.capture.cli.subprocess.run", side_effect=Exception("timeout"))
    def test_subprocess_error(self, mock_run, mock_which, mock_system):
        from vidflow.capture.cli import get_clipboard_urls

        result = get_clipboard_urls()
        assert result == []


class TestPreviewUrls:
    """preview_urls() lists URLs offline and confirms only for the clipboard source."""

    URL = "https://www.youtube.com/watch?v=abcdefghijk"

    @pytest.fixture(autouse=True)
    def no_network(self, monkeypatch):
        def boom(*args, **kwargs):
            raise AssertionError("preview must not spawn yt-dlp")

        monkeypatch.setattr("vidflow.capture.video.subprocess.run", boom)

    def _console(self):
        import io

        from rich.console import Console

        return Console(file=io.StringIO(), record=True, width=200)

    @patch("builtins.input", return_value="y")
    def test_confirm_accepted(self, mock_input):
        from vidflow.capture.cli import preview_urls

        assert preview_urls([self.URL], self._console(), source="clipboard") is True

    @patch("builtins.input", return_value="n")
    def test_confirm_rejected(self, mock_input):
        from vidflow.capture.cli import preview_urls

        assert preview_urls([self.URL], self._console(), source="clipboard") is False

    @patch("builtins.input", return_value="y")
    def test_table_lists_ids_and_urls_without_network(self, mock_input):
        from vidflow.capture.cli import preview_urls

        con = self._console()
        urls = [self.URL, "https://youtu.be/lmnopqrstuv"]
        assert preview_urls(urls, con, source="clipboard") is True
        text = con.export_text()
        assert "abcdefghijk" in text and "lmnopqrstuv" in text
        assert self.URL in text

    def test_non_clipboard_source_auto_confirms(self):
        from vidflow.capture.cli import preview_urls

        assert preview_urls([self.URL], self._console(), source="args") is True


class TestVidflowClipboardFallback:
    """Tests for the unified CLI's clipboard fallback (vidflow youtube)."""

    URL = "https://www.youtube.com/watch?v=abc123"

    def _capture_result(self):
        from vidflow.cli_common import OperationResult

        return OperationResult(
            success=True,
            message="Captured",
            data={"output_path": "/tmp/fake.md"},
        )

    @patch("vidflow.capture.cli.get_clipboard_urls", return_value=[])
    def test_no_args_no_clipboard_is_usage_error(self, mock_clip):
        from vidflow.cli import main as vidflow_main

        assert vidflow_main(["youtube"]) == 2
        mock_clip.assert_called_once()

    @patch("vidflow.capture.capture_youtube")
    @patch("vidflow.capture.cli.get_clipboard_urls")
    def test_clipboard_urls_are_captured(self, mock_clip, mock_capture):
        from vidflow.cli import main as vidflow_main

        mock_clip.return_value = [self.URL]
        mock_capture.return_value = self._capture_result()

        # Non-TTY stdin under pytest, so no confirmation prompt fires
        assert vidflow_main(["youtube"]) == 0
        assert mock_capture.call_args.kwargs["url"] == self.URL

    @patch("vidflow.capture.capture_youtube")
    @patch("vidflow.capture.cli.get_clipboard_urls")
    def test_explicit_args_skip_clipboard(self, mock_clip, mock_capture):
        from vidflow.cli import main as vidflow_main

        mock_capture.return_value = self._capture_result()

        assert vidflow_main(["youtube", self.URL]) == 0
        mock_clip.assert_not_called()

    @patch("vidflow.capture.capture_youtube")
    @patch("vidflow.capture.cli.get_clipboard_urls")
    def test_declined_confirmation_cancels(self, mock_clip, mock_capture, monkeypatch):
        import io

        from vidflow.cli import main as vidflow_main

        mock_clip.return_value = [self.URL]
        fake_stdin = io.StringIO()
        fake_stdin.isatty = lambda: True
        monkeypatch.setattr("sys.stdin", fake_stdin)
        monkeypatch.setattr("builtins.input", lambda *a: "n")

        assert vidflow_main(["youtube"]) == 0
        mock_capture.assert_not_called()
