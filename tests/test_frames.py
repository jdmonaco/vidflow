"""Tests for ffmpeg version detection and VFR flag selection."""

from unittest.mock import patch

from vidflow.capture import frames


class TestFfmpegVersion:
    """Tests for ffmpeg_version() parsing."""

    def _version_for(self, stdout: str) -> tuple[int, int]:
        frames.ffmpeg_version.cache_clear()
        with patch("vidflow.capture.frames.subprocess.run") as mock_run:
            mock_run.return_value.stdout = stdout
            result = frames.ffmpeg_version()
        frames.ffmpeg_version.cache_clear()
        return result

    def test_homebrew_style(self):
        assert self._version_for("ffmpeg version 9.0.1 Copyright ...") == (9, 0)

    def test_tagged_style(self):
        assert self._version_for("ffmpeg version n5.1.2 Copyright ...") == (5, 1)

    def test_ubuntu_style(self):
        assert self._version_for(
            "ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright ..."
        ) == (4, 4)

    def test_unparseable(self):
        assert self._version_for("garbage") == (0, 0)


class TestVfrOutputArgs:
    """Tests for the -fps_mode / -vsync selection."""

    def _args_for(self, version: tuple[int, int]) -> list[str]:
        with patch("vidflow.capture.frames.ffmpeg_version", return_value=version):
            return frames.vfr_output_args()

    def test_modern_ffmpeg_uses_fps_mode(self):
        assert self._args_for((9, 0)) == ["-fps_mode", "vfr"]

    def test_boundary_5_1_uses_fps_mode(self):
        assert self._args_for((5, 1)) == ["-fps_mode", "vfr"]

    def test_old_ffmpeg_uses_vsync(self):
        assert self._args_for((4, 4)) == ["-vsync", "vfr"]
        assert self._args_for((5, 0)) == ["-vsync", "vfr"]

    def test_unknown_version_defaults_to_fps_mode(self):
        assert self._args_for((0, 0)) == ["-fps_mode", "vfr"]
