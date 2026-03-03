"""Integration tests for CLI entry points."""

import subprocess
import sys

import pytest

from vidflow.cli import main as vidflow_main


class TestVidflowCli:
    """Tests for vidflow main CLI."""

    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            vidflow_main(["--version"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "0.2.0" in captured.out

    def test_help(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            vidflow_main(["--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "youtube" in captured.out
        assert "local" in captured.out
        assert "transcribe" in captured.out

    def test_no_subcommand(self):
        result = vidflow_main([])
        assert result == 2


class TestYtcaptureEntry:
    """Tests for ytcapture standalone entry point."""

    def test_version(self):
        from vidflow.capture.cli import ytcapture_main

        with pytest.raises(SystemExit) as exc_info:
            ytcapture_main(["--version"])
        assert exc_info.value.code == 0

    def test_help(self):
        with pytest.raises(SystemExit) as exc_info:
            from vidflow.capture.cli import ytcapture_main
            ytcapture_main(["--help"])
        assert exc_info.value.code == 0

    def test_no_args_no_clipboard(self):
        """Without URLs or clipboard, should exit with error."""
        from unittest.mock import patch
        from vidflow.capture.cli import ytcapture_main

        with patch("vidflow.capture.cli.get_clipboard_urls", return_value=[]):
            with pytest.raises(SystemExit) as exc_info:
                ytcapture_main([])
            assert exc_info.value.code == 2


class TestVidcaptureEntry:
    """Tests for vidcapture standalone entry point."""

    def test_version(self):
        from vidflow.capture.cli import vidcapture_main

        with pytest.raises(SystemExit) as exc_info:
            vidcapture_main(["--version"])
        assert exc_info.value.code == 0

    def test_help(self):
        with pytest.raises(SystemExit) as exc_info:
            from vidflow.capture.cli import vidcapture_main
            vidcapture_main(["--help"])
        assert exc_info.value.code == 0

    def test_no_args(self):
        """Without files, should exit with error."""
        from vidflow.capture.cli import vidcapture_main

        with pytest.raises(SystemExit) as exc_info:
            vidcapture_main([])
        assert exc_info.value.code == 2


class TestVidscribeEntry:
    """Tests for vidscribe standalone entry point."""

    def test_help(self):
        from vidflow.transcribe.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_no_args(self):
        """Without inputs, should exit with error."""
        from vidflow.transcribe.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 2


class TestImports:
    """Tests for package-level imports."""

    def test_capture_imports(self):
        from vidflow.capture import capture_youtube, capture_local, process_video, process_local_video
        assert callable(capture_youtube)
        assert callable(capture_local)
        assert callable(process_video)
        assert callable(process_local_video)

    def test_transcribe_imports(self):
        from vidflow.transcribe import (
            VidscribeProcessor,
            parse_vidcapture_markdown,
            merge_vidcapture_documents,
            transcribe_markdown,
        )
        assert callable(transcribe_markdown)

    def test_youtube_import(self):
        from vidflow.youtube import transcribe_youtube
        assert callable(transcribe_youtube)
