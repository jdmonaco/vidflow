"""Tests for the single-call yt-dlp fetch (metadata + captions + video)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vidflow.capture.video import VideoBlocked, VideoError, fetch_video

URL = "https://www.youtube.com/watch?v=abcdefghijk"
VID = "abcdefghijk"


def _fake_run(
    files: dict[str, str],
    stderr: str = "",
    returncode: int = 0,
    info: dict | None = None,
    write_on_failure: bool = False,
):
    """Simulate yt-dlp writing files next to the -o template."""

    def fake(cmd, **kwargs):
        out_dir = Path(cmd[cmd.index("--output") + 1]).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        if returncode == 0 or write_on_failure:
            payload = {
                "id": VID,
                "title": "T",
                "channel": "C",
                "duration": 10,
                "upload_date": "20240101",
            }
            payload.update(info or {})
            (out_dir / f"{VID}.info.json").write_text(json.dumps(payload), encoding="utf-8")
            for name, text in files.items():
                (out_dir / name).write_text(text, encoding="utf-8")
        return MagicMock(returncode=returncode, stdout="", stderr=stderr)

    return fake


class TestFetchVideo:
    def test_single_call_requests_everything(self, tmp_path):
        recorded = []

        def fake(cmd, **kwargs):
            recorded.append(cmd)
            return _fake_run({f"{VID}.mp4": "video"})(cmd, **kwargs)

        with patch("vidflow.capture.video.subprocess.run", fake):
            fetch_video(URL, tmp_path, language="en")

        assert len(recorded) == 1
        cmd = recorded[0]
        for flag in (
            "--write-info-json",
            "--write-subs",
            "--write-auto-subs",
            "--ignore-errors",
            "--no-playlist",
        ):
            assert flag in cmd
        assert cmd[cmd.index("--sub-langs") + 1] == "en.*,en"
        assert cmd[cmd.index("--sub-format") + 1] == "json3"
        assert "--no-warnings" not in cmd  # caption 429 warnings must stay visible

    def test_returns_metadata_video_and_captions(self, tmp_path):
        fake = _fake_run(
            {f"{VID}.mp4": "video", f"{VID}.en.json3": "{}", f"{VID}.en-orig.json3": "{}"},
            info={"subtitles": {"en": [{"ext": "json3"}]}, "automatic_captions": {"en-orig": []}},
        )
        with patch("vidflow.capture.video.subprocess.run", fake):
            fetch = fetch_video(URL, tmp_path)

        assert fetch.metadata.video_id == VID
        assert fetch.metadata.title == "T"
        assert fetch.video_path == tmp_path / f"{VID}.mp4"
        assert [f.name for f in fetch.caption_files] == [f"{VID}.en-orig.json3", f"{VID}.en.json3"]
        assert fetch.manual_langs == frozenset({"en"})
        assert fetch.captions_blocked is False
        assert not (tmp_path / f"{VID}.info.json").exists()  # consumed

    def test_caption_429_flagged_when_video_still_downloads(self, tmp_path):
        """Under --ignore-errors yt-dlp exits 1 after a caption 429 but still downloads."""
        fake = _fake_run(
            {f"{VID}.mp4": "video"},
            stderr="ERROR: Unable to download video subtitles for 'en': HTTP Error 429: Too Many Requests",
            returncode=1,
            write_on_failure=True,
        )
        with patch("vidflow.capture.video.subprocess.run", fake):
            fetch = fetch_video(URL, tmp_path)
        assert fetch.captions_blocked is True
        assert fetch.caption_files == []
        assert fetch.video_path.name == f"{VID}.mp4"

    def test_bot_gate_raises_video_blocked(self, tmp_path):
        fake = _fake_run({}, stderr="ERROR: Sign in to confirm you’re not a bot.", returncode=1)
        with patch("vidflow.capture.video.subprocess.run", fake):
            with pytest.raises(VideoBlocked):
                fetch_video(URL, tmp_path)

    def test_missing_video_file_is_error(self, tmp_path):
        with patch("vidflow.capture.video.subprocess.run", _fake_run({})):
            with pytest.raises(VideoError, match="no video file"):
                fetch_video(URL, tmp_path)
