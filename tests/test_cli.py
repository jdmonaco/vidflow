"""Integration tests for CLI entry points."""

import pytest

from vidflow.cli import main as vidflow_main


class TestVidflowCli:
    """Tests for vidflow main CLI."""

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
        from vidflow.capture import (
            capture_youtube,
            capture_local,
            process_video,
            process_local_video,
        )

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


class TestCaptureDefaultsFromConfig:
    """vidflow youtube/local take capture defaults from the config file."""

    @pytest.fixture
    def config_file(self, tmp_path, monkeypatch):
        from vidflow.capture import config as cfg

        path = tmp_path / "config.yml"
        monkeypatch.setattr(cfg, "get_config_path", lambda: path)
        cfg.clear_config_cache()
        yield path
        cfg.clear_config_cache()

    def test_builtin_defaults(self, config_file):
        from vidflow.capture.config import DEFAULT_DEDUP_THRESHOLD
        from vidflow.cli import build_parser

        config_file.write_text("")
        args = build_parser().parse_args(["youtube", "x"])
        assert args.dedup_threshold == DEFAULT_DEDUP_THRESHOLD
        assert args.interval == 15
        args = build_parser().parse_args(["local", "x.mp4"])
        assert args.dedup_threshold == DEFAULT_DEDUP_THRESHOLD
        assert args.fast is True

    def test_config_overrides_defaults(self, config_file):
        from vidflow.cli import build_parser

        config_file.write_text("dedup_threshold: 0.7\ninterval: 30\nfast: false\n")
        args = build_parser().parse_args(["youtube", "x"])
        assert args.dedup_threshold == 0.7
        assert args.interval == 30
        args = build_parser().parse_args(["local", "x.mp4"])
        assert args.fast is False

    def test_cli_overrides_config(self, config_file):
        from vidflow.cli import build_parser

        config_file.write_text("dedup_threshold: 0.7\n")
        args = build_parser().parse_args(["youtube", "x", "--dedup-threshold", "0.9"])
        assert args.dedup_threshold == 0.9

    def test_help_shows_config_default(self, config_file, capsys):
        from vidflow.cli import main

        config_file.write_text("dedup_threshold: 0.7\n")
        with pytest.raises(SystemExit):
            main(["youtube", "--help"])
        assert "(default: 0.7)" in capsys.readouterr().out
