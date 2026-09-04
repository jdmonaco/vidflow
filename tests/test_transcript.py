"""Tests for the two-lane transcript fetcher (API primary, yt-dlp fallback)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vidflow.capture import transcript as transcript_mod
from vidflow.capture.transcript import (
    TranscriptBlocked,
    TranscriptSegment,
    get_transcript,
    get_transcript_ytdlp,
    parse_json3_captions,
    reset_block_state,
)


@pytest.fixture(autouse=True)
def clear_block_state():
    reset_block_state()
    yield
    reset_block_state()


JSON3 = {
    "events": [
        {"tStartMs": 0, "dDurationMs": 2000},  # filler: no segs
        {"tStartMs": 100, "dDurationMs": 1900, "segs": [{"utf8": "\n"}]},  # newline only
        {
            "tStartMs": 500,
            "dDurationMs": 3200,
            "segs": [{"utf8": "hello "}, {"utf8": "world"}],
        },
        {"tStartMs": 3700, "dDurationMs": 2100, "segs": [{"utf8": "second\ncue"}]},
    ]
}


def _write_json3(path: Path, events=None, text=None):
    data = JSON3 if events is None else {"events": events}
    if text is not None:
        data = {"events": [{"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": text}]}]}
    path.write_text(json.dumps(data), encoding="utf-8")


class TestParseJson3:
    def test_parses_cues_and_drops_filler(self, tmp_path):
        f = tmp_path / "x.en.json3"
        _write_json3(f)
        segments = parse_json3_captions(f)
        assert segments == [
            TranscriptSegment(text="hello world", start=0.5, duration=3.2),
            TranscriptSegment(text="second cue", start=3.7, duration=2.1),
        ]


class TestGetTranscriptFallback:
    @patch("vidflow.capture.transcript.get_transcript_ytdlp")
    @patch("vidflow.capture.transcript.get_transcript_api")
    def test_api_success_short_circuits(self, mock_api, mock_ytdlp):
        sentinel = [TranscriptSegment(text="t", start=0.0, duration=1.0)]
        mock_api.return_value = sentinel
        assert get_transcript("abcdefghijk") == sentinel
        mock_ytdlp.assert_not_called()

    @patch("vidflow.capture.transcript.get_transcript_ytdlp")
    @patch("vidflow.capture.transcript.get_transcript_api", return_value=None)
    def test_api_failure_falls_back_to_ytdlp(self, mock_api, mock_ytdlp):
        sentinel = [TranscriptSegment(text="t", start=0.0, duration=1.0)]
        mock_ytdlp.return_value = sentinel
        assert get_transcript("abcdefghijk", language="en", prefer_manual=False) == sentinel
        mock_ytdlp.assert_called_once_with("abcdefghijk", "en", False)


class TestGetTranscriptYtdlp:
    """yt-dlp lane with a mocked subprocess that writes caption files."""

    def _fake_run(self, files_by_flag, recorded_flags):
        """files_by_flag: {'--write-subs': {name: text}, ...} written per pass."""

        def fake(cmd, **kwargs):
            flag = next(f for f in ("--write-subs", "--write-auto-subs") if f in cmd)
            recorded_flags.append(flag)
            out_dir = Path(cmd[cmd.index("-o") + 1]).parent
            for name, text in files_by_flag.get(flag, {}).items():
                _write_json3(out_dir / name, text=text)

            class R:
                returncode = 0
                stdout = ""
                stderr = ""

            return R()

        return fake

    def test_auto_captions_found_on_second_pass(self, monkeypatch):
        flags = []
        fake = self._fake_run(
            {"--write-auto-subs": {"abcdefghijk.en-orig.json3": "auto text"}}, flags
        )
        with patch("vidflow.capture.transcript.subprocess.run", fake):
            segments = get_transcript_ytdlp("abcdefghijk")
        assert flags == ["--write-subs", "--write-auto-subs"]
        assert segments is not None
        assert segments[0].text == "auto text"

    def test_manual_pass_preferred(self, monkeypatch):
        flags = []
        fake = self._fake_run(
            {
                "--write-subs": {"abcdefghijk.en.json3": "manual text"},
                "--write-auto-subs": {"abcdefghijk.en.json3": "auto text"},
            },
            flags,
        )
        with patch("vidflow.capture.transcript.subprocess.run", fake):
            segments = get_transcript_ytdlp("abcdefghijk", prefer_manual=True)
        assert flags == ["--write-subs"]
        assert segments[0].text == "manual text"

    def test_exact_language_beats_variant(self, monkeypatch):
        flags = []
        fake = self._fake_run(
            {
                "--write-subs": {
                    "abcdefghijk.en-orig.json3": "variant text",
                    "abcdefghijk.en.json3": "exact text",
                }
            },
            flags,
        )
        with patch("vidflow.capture.transcript.subprocess.run", fake):
            segments = get_transcript_ytdlp("abcdefghijk")
        assert segments[0].text == "exact text"

    def test_nothing_found_returns_none(self, monkeypatch):
        flags = []
        fake = self._fake_run({}, flags)
        with patch("vidflow.capture.transcript.subprocess.run", fake):
            assert get_transcript_ytdlp("abcdefghijk") is None
        assert flags == ["--write-subs", "--write-auto-subs"]


class TestBlockedDetection:
    """Blocked fetches raise TranscriptBlocked instead of returning None."""

    def _fake_run_429(self, cmd, **kwargs):
        class R:
            returncode = 1
            stdout = ""
            stderr = "ERROR: Unable to download video subtitles: HTTP Error 429: Too Many Requests"

        return R()

    def test_ytdlp_429_raises_blocked(self):
        with patch("vidflow.capture.transcript.subprocess.run", self._fake_run_429):
            with pytest.raises(TranscriptBlocked):
                get_transcript_ytdlp("abcdefghijk")

    @patch("vidflow.capture.transcript.get_transcript_ytdlp", return_value=None)
    @patch(
        "vidflow.capture.transcript.get_transcript_api",
        side_effect=TranscriptBlocked("blocked"),
    )
    def test_api_blocked_and_ytdlp_empty_raises(self, mock_api, mock_ytdlp):
        with pytest.raises(TranscriptBlocked):
            get_transcript("abcdefghijk")

    @patch("vidflow.capture.transcript.get_transcript_ytdlp")
    @patch(
        "vidflow.capture.transcript.get_transcript_api",
        side_effect=TranscriptBlocked("blocked"),
    )
    def test_api_blocked_but_ytdlp_succeeds(self, mock_api, mock_ytdlp):
        sentinel = [TranscriptSegment(text="t", start=0.0, duration=1.0)]
        mock_ytdlp.return_value = sentinel
        assert get_transcript("abcdefghijk") == sentinel
        assert transcript_mod._block_detected is False

    @patch("vidflow.capture.transcript.get_transcript_ytdlp", return_value=None)
    @patch(
        "vidflow.capture.transcript.get_transcript_api",
        side_effect=TranscriptBlocked("blocked"),
    )
    def test_block_is_sticky_and_fails_fast(self, mock_api, mock_ytdlp):
        with pytest.raises(TranscriptBlocked):
            get_transcript("abcdefghijk")
        # Second call short-circuits: no further lane attempts
        with pytest.raises(TranscriptBlocked):
            get_transcript("lmnopqrstuv")
        assert mock_api.call_count == 1
        assert mock_ytdlp.call_count == 1

    def test_api_maps_request_blocked(self):
        from youtube_transcript_api import RequestBlocked

        from vidflow.capture.transcript import get_transcript_api

        with patch("vidflow.capture.transcript.YouTubeTranscriptApi") as mock_api_cls:
            mock_api_cls.return_value.list.side_effect = RequestBlocked("abcdefghijk")
            with pytest.raises(TranscriptBlocked):
                get_transcript_api("abcdefghijk")


class TestBulkAbort:
    """A blocked transcript fetch aborts the whole capture run."""

    URLS = [
        "https://www.youtube.com/watch?v=aaaaaaaaaaa",
        "https://www.youtube.com/watch?v=bbbbbbbbbbb",
        "https://www.youtube.com/watch?v=ccccccccccc",
    ]

    def test_capture_youtube_reraises_blocked(self, monkeypatch):
        import vidflow.capture as capture_pkg

        monkeypatch.setattr(
            capture_pkg,
            "process_video",
            lambda **kw: (_ for _ in ()).throw(TranscriptBlocked("blocked")),
        )
        with pytest.raises(TranscriptBlocked):
            capture_pkg.capture_youtube(url=self.URLS[0], output_dir=Path("."))

    def test_cmd_youtube_aborts_run_on_block(self, monkeypatch, capsys):
        from vidflow.cli import main as vidflow_main
        from vidflow.cli_common import OperationResult

        # No pacing sleeps during the test
        monkeypatch.setattr("vidflow.capture.transcript.time.sleep", lambda s: None)

        calls = []

        def fake_capture(**kwargs):
            calls.append(kwargs["url"])
            if len(calls) == 2:
                raise TranscriptBlocked("blocked")
            return OperationResult(success=True, message="ok", data={"output_path": "/tmp/x.md"})

        monkeypatch.setattr("vidflow.capture.capture_youtube", fake_capture)

        result = vidflow_main(["youtube", *self.URLS])

        assert result == 1
        # Second video raised; the third was never attempted
        assert calls == self.URLS[:2]
        assert "ABORTED" in capsys.readouterr().err


class TestPacing:
    """Caption requests are paced against the previous caption request."""

    def test_no_sleep_before_first_request(self, monkeypatch):
        from vidflow.capture.transcript import pace_caption_request

        calls = []
        monkeypatch.setattr("vidflow.capture.transcript.time.sleep", calls.append)
        assert pace_caption_request() == 0.0
        assert calls == []

    def test_jittered_sleep_between_back_to_back_requests(self, monkeypatch):
        from vidflow.capture.transcript import pace_caption_request

        calls = []
        logs = []
        monkeypatch.setattr("vidflow.capture.transcript.time.sleep", calls.append)
        pace_caption_request()
        pace_caption_request(log=logs.append, span=(2.0, 5.0))
        assert len(calls) == 1
        assert 0.0 < calls[0] <= 5.0
        assert logs and "Pacing" in logs[0]

    def test_elapsed_time_counts_toward_interval(self, monkeypatch):
        from vidflow.capture.transcript import pace_caption_request

        calls = []
        monkeypatch.setattr("vidflow.capture.transcript.time.sleep", calls.append)
        pace_caption_request()
        # A slow download happened since the last caption request
        monkeypatch.setattr(
            "vidflow.capture.transcript._last_caption_request",
            transcript_mod.time.monotonic() - 60.0,
        )
        assert pace_caption_request(span=(2.0, 5.0)) == 0.0
        assert calls == []

    @patch("vidflow.capture.transcript.get_transcript_ytdlp", return_value=None)
    @patch("vidflow.capture.transcript.get_transcript_api", return_value=None)
    def test_get_transcript_paces_consecutive_calls(self, mock_api, mock_ytdlp, monkeypatch):
        calls = []
        monkeypatch.setattr("vidflow.capture.transcript.time.sleep", calls.append)
        get_transcript("aaaaaaaaaaa")
        get_transcript("bbbbbbbbbbb")
        get_transcript("ccccccccccc")
        assert len(calls) == 2

    @patch("vidflow.capture.transcript.get_transcript_ytdlp", return_value=None)
    @patch("vidflow.capture.transcript.get_transcript_api", return_value=None)
    def test_no_pacing_when_no_caption_request_was_made(self, mock_api, mock_ytdlp, monkeypatch):
        """A metadata failure before the caption stage must not trigger pacing."""
        calls = []
        monkeypatch.setattr("vidflow.capture.transcript.time.sleep", calls.append)
        # Simulate earlier videos failing before get_transcript was reached
        get_transcript("aaaaaaaaaaa")
        assert calls == []


class TestBotGate:
    """YouTube's sign-in bot challenge is an IP-level block, like a 429."""

    BOT_STDERR = (
        "ERROR: [youtube] abcdefghijk: Sign in to confirm you\u2019re not a bot. "
        "Use --cookies-from-browser or --cookies for the authentication."
    )

    def test_metadata_bot_gate_raises_video_blocked(self):
        from vidflow.capture.video import VideoBlocked, get_video_metadata

        proc = MagicMock(returncode=1, stdout="", stderr=self.BOT_STDERR)
        with patch("vidflow.capture.video.subprocess.run", return_value=proc):
            with pytest.raises(VideoBlocked):
                get_video_metadata("https://www.youtube.com/watch?v=abcdefghijk")

    def test_other_sign_in_stays_video_error(self):
        from vidflow.capture.video import VideoBlocked, VideoError, get_video_metadata

        proc = MagicMock(returncode=1, stdout="", stderr="ERROR: Sign in to confirm your age")
        with patch("vidflow.capture.video.subprocess.run", return_value=proc):
            with pytest.raises(VideoError) as exc:
                get_video_metadata("https://www.youtube.com/watch?v=abcdefghijk")
        assert not isinstance(exc.value, VideoBlocked)

    def test_ytdlp_caption_bot_gate_raises_blocked(self, monkeypatch):
        proc = MagicMock(returncode=1, stdout="", stderr=self.BOT_STDERR)
        with patch("vidflow.capture.transcript.subprocess.run", return_value=proc):
            with pytest.raises(TranscriptBlocked):
                get_transcript_ytdlp("abcdefghijk")

    def test_cmd_youtube_aborts_run_on_bot_gate(self, monkeypatch, capsys):
        from vidflow.capture.video import VideoBlocked
        from vidflow.cli import main as vidflow_main

        urls = TestBulkAbort.URLS
        calls = []

        def fake_capture(**kwargs):
            calls.append(kwargs["url"])
            raise VideoBlocked("YouTube is bot-gating requests from this IP")

        monkeypatch.setattr("vidflow.capture.capture_youtube", fake_capture)

        result = vidflow_main(["youtube", *urls])

        assert result == 1
        assert calls == urls[:1]
        err = capsys.readouterr().err
        assert "ABORTED" in err and "bot-gating" in err

    def test_cmd_youtube_prints_per_video_failure(self, monkeypatch, capsys):
        from vidflow.cli import main as vidflow_main
        from vidflow.cli_common import OperationResult

        def fake_capture(**kwargs):
            return OperationResult(
                success=False, message="YouTube capture failed: boom", errors=["boom"]
            )

        monkeypatch.setattr("vidflow.capture.capture_youtube", fake_capture)
        result = vidflow_main(["youtube", *TestBulkAbort.URLS])

        assert result == 1
        err = capsys.readouterr().err
        assert err.count("Failed [") == 3
        assert "boom" in err
