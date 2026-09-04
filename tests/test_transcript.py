"""Tests for transcript resolution (downloaded captions first, API fallback)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vidflow.capture import transcript as transcript_mod
from vidflow.capture.transcript import (
    TranscriptBlocked,
    TranscriptSegment,
    get_transcript,
    transcript_from_caption_files,
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


SEG = [TranscriptSegment(text="t", start=0.0, duration=1.0)]


class TestGetTranscriptLanes:
    """Downloaded caption files win; the API lane runs only when they yield nothing."""

    @patch("vidflow.capture.transcript.get_transcript_api")
    def test_caption_file_short_circuits_api(self, mock_api, tmp_path):
        f = tmp_path / "abcdefghijk.en.json3"
        _write_json3(f, text="from file")
        segments = get_transcript("abcdefghijk", caption_files=[f])
        assert segments[0].text == "from file"
        mock_api.assert_not_called()

    @patch("vidflow.capture.transcript.get_transcript_api", return_value=SEG)
    def test_no_caption_files_falls_back_to_api(self, mock_api):
        assert get_transcript("abcdefghijk", language="en", prefer_manual=False) == SEG
        mock_api.assert_called_once_with("abcdefghijk", "en", False)

    @patch("vidflow.capture.transcript.get_transcript_api", return_value=SEG)
    def test_empty_caption_file_falls_back_to_api(self, mock_api, tmp_path):
        f = tmp_path / "abcdefghijk.en.json3"
        _write_json3(f, events=[])
        assert get_transcript("abcdefghijk", caption_files=[f]) == SEG
        mock_api.assert_called_once()

    @patch("vidflow.capture.transcript.get_transcript_api", return_value=None)
    def test_nothing_anywhere_returns_none(self, mock_api):
        assert get_transcript("abcdefghijk") is None


class TestTranscriptFromCaptionFiles:
    """Selection among the json3 files yt-dlp wrote alongside the download."""

    def test_exact_language_beats_variant(self, tmp_path):
        exact = tmp_path / "abcdefghijk.en.json3"
        variant = tmp_path / "abcdefghijk.en-orig.json3"
        _write_json3(exact, text="exact text")
        _write_json3(variant, text="variant text")
        segments = transcript_from_caption_files([variant, exact], "abcdefghijk")
        assert segments[0].text == "exact text"

    def test_variant_used_when_exact_missing(self, tmp_path):
        variant = tmp_path / "abcdefghijk.en-orig.json3"
        _write_json3(variant, text="auto text")
        segments = transcript_from_caption_files([variant], "abcdefghijk")
        assert segments[0].text == "auto text"

    def test_prefer_manual_orders_manual_language_first(self, tmp_path):
        auto_exact = tmp_path / "abcdefghijk.en.json3"
        manual_variant = tmp_path / "abcdefghijk.en-GB.json3"
        _write_json3(auto_exact, text="auto text")
        _write_json3(manual_variant, text="manual text")
        files = [auto_exact, manual_variant]
        assert (
            transcript_from_caption_files(
                files, "abcdefghijk", prefer_manual=True, manual_langs=frozenset({"en-GB"})
            )[0].text
            == "manual text"
        )
        assert (
            transcript_from_caption_files(files, "abcdefghijk", prefer_manual=False)[0].text
            == "auto text"
        )

    def test_ignores_other_videos_and_unparseable(self, tmp_path):
        other = tmp_path / "zzzzzzzzzzz.en.json3"
        broken = tmp_path / "abcdefghijk.en.json3"
        _write_json3(other, text="other")
        broken.write_text("not json", encoding="utf-8")
        assert transcript_from_caption_files([other, broken], "abcdefghijk") is None


class TestBlockedDetection:
    """Blocked fetches raise TranscriptBlocked instead of returning None."""

    @patch("vidflow.capture.transcript.get_transcript_api")
    def test_caption_429_raises_without_touching_api(self, mock_api):
        with pytest.raises(TranscriptBlocked):
            get_transcript("abcdefghijk", captions_blocked=True)
        mock_api.assert_not_called()
        assert transcript_mod._block_detected is True

    @patch("vidflow.capture.transcript.get_transcript_api")
    def test_caption_429_ignored_when_a_file_still_parsed(self, mock_api, tmp_path):
        f = tmp_path / "abcdefghijk.en.json3"
        _write_json3(f, text="got one")
        segments = get_transcript("abcdefghijk", caption_files=[f], captions_blocked=True)
        assert segments[0].text == "got one"
        assert transcript_mod._block_detected is False

    @patch(
        "vidflow.capture.transcript.get_transcript_api",
        side_effect=TranscriptBlocked("blocked"),
    )
    def test_api_blocked_raises(self, mock_api):
        with pytest.raises(TranscriptBlocked):
            get_transcript("abcdefghijk")

    @patch(
        "vidflow.capture.transcript.get_transcript_api",
        side_effect=TranscriptBlocked("blocked"),
    )
    def test_block_is_sticky_and_fails_fast(self, mock_api, tmp_path):
        with pytest.raises(TranscriptBlocked):
            get_transcript("abcdefghijk")
        # Second call short-circuits even with a usable caption file
        f = tmp_path / "lmnopqrstuv.en.json3"
        _write_json3(f, text="unused")
        with pytest.raises(TranscriptBlocked):
            get_transcript("lmnopqrstuv", caption_files=[f])
        assert mock_api.call_count == 1

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

    @patch("vidflow.capture.transcript.get_transcript_api", return_value=None)
    def test_get_transcript_paces_consecutive_calls(self, mock_api, monkeypatch):
        calls = []
        monkeypatch.setattr("vidflow.capture.transcript.time.sleep", calls.append)
        get_transcript("aaaaaaaaaaa")
        get_transcript("bbbbbbbbbbb")
        get_transcript("ccccccccccc")
        assert len(calls) == 2

    @patch("vidflow.capture.transcript.get_transcript_api", return_value=None)
    def test_no_pacing_when_no_caption_request_was_made(self, mock_api, monkeypatch):
        """A metadata failure before the caption stage must not trigger pacing."""
        calls = []
        monkeypatch.setattr("vidflow.capture.transcript.time.sleep", calls.append)
        # Simulate earlier videos failing before get_transcript was reached
        get_transcript("aaaaaaaaaaa")
        assert calls == []

    @patch("vidflow.capture.transcript.get_transcript_api", return_value=None)
    def test_caption_files_do_not_consume_pacing(self, mock_api, monkeypatch, tmp_path):
        """Captions that came with the download are not caption requests."""
        calls = []
        monkeypatch.setattr("vidflow.capture.transcript.time.sleep", calls.append)
        f = tmp_path / "aaaaaaaaaaa.en.json3"
        _write_json3(f, text="x")
        get_transcript("aaaaaaaaaaa", caption_files=[f])
        get_transcript("bbbbbbbbbbb")  # first real API request: no sleep
        assert calls == []


class TestBotGate:
    """YouTube's sign-in bot challenge is an IP-level block, like a 429."""

    BOT_STDERR = (
        "ERROR: [youtube] abcdefghijk: Sign in to confirm you\u2019re not a bot. "
        "Use --cookies-from-browser or --cookies for the authentication."
    )

    def test_other_sign_in_stays_video_error(self, tmp_path):
        from vidflow.capture.video import VideoBlocked, VideoError, fetch_video

        proc = MagicMock(returncode=1, stdout="", stderr="ERROR: Sign in to confirm your age")
        with patch("vidflow.capture.video.subprocess.run", return_value=proc):
            with pytest.raises(VideoError) as exc:
                fetch_video("https://www.youtube.com/watch?v=abcdefghijk", tmp_path)
        assert not isinstance(exc.value, VideoBlocked)

    def test_fetch_video_bot_gate_raises_video_blocked(self, tmp_path):
        from vidflow.capture.video import VideoBlocked, fetch_video

        proc = MagicMock(returncode=1, stdout="", stderr=self.BOT_STDERR)
        with patch("vidflow.capture.video.subprocess.run", return_value=proc):
            with pytest.raises(VideoBlocked):
                fetch_video("https://www.youtube.com/watch?v=abcdefghijk", tmp_path)

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
