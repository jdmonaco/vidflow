"""Tests for sidecar WebVTT discovery, voice-tag diarization, and speaker rendering."""

from pathlib import Path

import pytest

from vidflow.capture import core
from vidflow.capture.frames import FrameInfo
from vidflow.capture.local import LocalVideoError, LocalVideoMetadata
from vidflow.capture.markdown import format_segments, generate_markdown_body
from vidflow.capture.subtitles import (
    cues_end,
    find_sidecar_vtt,
    is_meeting_name_match,
    list_speakers,
    load_sidecar_vtt,
    parse_webvtt,
    split_voice_spans,
    teams_meeting_name,
)
from vidflow.capture.transcript import TranscriptSegment

# Shape of a Microsoft Teams meeting transcript download: UUID cue ids,
# one voice tag per cue, an utterance split across several cues.
TEAMS_VTT = (
    "WEBVTT\r\n"
    "\r\n"
    "3d2b1f9e-0000-4000-8000-000000000001/12-0\r\n"
    "00:00:01.000 --> 00:00:04.000\r\n"
    "<v Monaco, Joseph (NIH/NINDS) [C]>Welcome, everyone.</v>\r\n"
    "\r\n"
    "3d2b1f9e-0000-4000-8000-000000000001/12-1\r\n"
    "00:00:04.000 --> 00:00:06.000\r\n"
    "<v Monaco, Joseph (NIH/NINDS) [C]>Let's begin.</v>\r\n"
    "\r\n"
    "3d2b1f9e-0000-4000-8000-000000000002/7-0\r\n"
    "00:00:20.000 --> 00:00:23.000\r\n"
    "<v Jane Doe>Thanks &amp; hello.</v>\r\n"
)


class TestVoiceTags:
    def test_teams_transcript_speakers(self):
        segments = parse_webvtt(TEAMS_VTT)
        assert [s.speaker for s in segments] == [
            "Monaco, Joseph (NIH/NINDS) [C]",
            "Monaco, Joseph (NIH/NINDS) [C]",
            "Jane Doe",
        ]
        assert segments[2].text == "Thanks & hello."

    def test_voice_class_annotation(self):
        segments = parse_webvtt("WEBVTT\n\n00:01.000 --> 00:02.000\n<v.loud Esme>Hi</v>\n")
        assert segments[0].speaker == "Esme"
        assert segments[0].text == "Hi"

    def test_unclosed_voice_tag(self):
        segments = parse_webvtt("WEBVTT\n\n00:01.000 --> 00:02.000\n<v Roger Bingham>We are\n")
        assert segments[0].speaker == "Roger Bingham"
        assert segments[0].text == "We are"

    def test_multiple_voices_in_one_cue(self):
        spans = split_voice_spans("<v A>one</v>\n<v B>two</v>")
        assert spans == [("A", "one</v>\n"), ("B", "two</v>")]
        segments = parse_webvtt("WEBVTT\n\n00:01.000 --> 00:02.000\n<v A>one</v>\n<v B>two</v>\n")
        assert [(s.speaker, s.text) for s in segments] == [("A", "one"), ("B", "two")]

    def test_untagged_cue_has_no_speaker(self):
        segments = parse_webvtt("WEBVTT\n\n00:01.000 --> 00:02.000\nplain\n")
        assert segments[0].speaker is None

    def test_list_speakers_first_appearance_order(self):
        assert list_speakers(parse_webvtt(TEAMS_VTT)) == [
            "Monaco, Joseph (NIH/NINDS) [C]",
            "Jane Doe",
        ]


class TestFindSidecar:
    def _video(self, tmp_path: Path, name: str = "talk.mp4") -> Path:
        video = tmp_path / name
        video.write_bytes(b"x")
        return video

    def test_none_when_absent(self, tmp_path):
        assert find_sidecar_vtt(self._video(tmp_path)) is None

    def test_exact_stem(self, tmp_path):
        video = self._video(tmp_path)
        (tmp_path / "talk.vtt").write_text("WEBVTT\n")
        assert find_sidecar_vtt(video).name == "talk.vtt"

    @pytest.mark.parametrize("name", ["talk-en-US.vtt", "talk.en.vtt", "talk.en-GB.vtt"])
    def test_language_suffixes(self, tmp_path, name):
        video = self._video(tmp_path)
        (tmp_path / name).write_text("WEBVTT\n")
        assert find_sidecar_vtt(video).name == name

    def test_teams_screen_recording_name(self, tmp_path):
        video = self._video(tmp_path, "Screen Recording 2026-10-06 124555_Duygu Kuzum.mp4")
        (tmp_path / "Screen Recording 2026-10-06 124555_Duygu Kuzum-en-US.vtt").write_text("x")
        assert find_sidecar_vtt(video).name.endswith("-en-US.vtt")

    def test_preference_order(self, tmp_path):
        video = self._video(tmp_path)
        for name in ["talk-fr.vtt", "talk-en-US.vtt", "talk.vtt"]:
            (tmp_path / name).write_text("WEBVTT\n")
        assert find_sidecar_vtt(video).name == "talk.vtt"
        (tmp_path / "talk.vtt").unlink()
        assert find_sidecar_vtt(video).name == "talk-en-US.vtt"
        (tmp_path / "talk-en-US.vtt").unlink()
        assert find_sidecar_vtt(video).name == "talk-fr.vtt"

    @pytest.mark.parametrize(
        "name", ["talk-old.vtt", "talk2.vtt", "talkative.vtt", "talk-v2.vtt", ".talk.vtt"]
    )
    def test_rejects_non_language_suffixes(self, tmp_path, name):
        video = self._video(tmp_path)
        (tmp_path / name).write_text("WEBVTT\n")
        assert find_sidecar_vtt(video) is None

    def test_load_strips_bom(self, tmp_path):
        path = tmp_path / "talk.vtt"
        path.write_bytes(b"\xef\xbb\xbf" + TEAMS_VTT.encode())
        assert len(load_sidecar_vtt(path)) == 3


def _seg(text: str, start: float, speaker: str | None = None) -> TranscriptSegment:
    return TranscriptSegment(text=text, start=start, duration=1.0, speaker=speaker)


class TestSpeakerRendering:
    def test_undiarized_runs_together(self):
        assert format_segments([_seg("a", 0), _seg("b", 1)]) == "a b"

    def test_groups_consecutive_turns(self):
        segs = [_seg("a", 0, "X"), _seg("b", 1, "X"), _seg("c", 2, "Y"), _seg("d", 3, "X")]
        assert format_segments(segs) == "**X**: a b\n\n**Y**: c\n\n**X**: d"

    def test_untagged_turn_unlabeled(self):
        assert format_segments([_seg("a", 0), _seg("b", 1, "Y")]) == "a\n\n**Y**: b"

    def test_body_labels_each_section(self, tmp_path):
        frames = [
            FrameInfo(path=tmp_path / "f1.jpg", timestamp=0.0),
            FrameInfo(path=tmp_path / "f2.jpg", timestamp=10.0),
        ]
        grouped = [(frames[0], [_seg("a", 0, "X")]), (frames[1], [_seg("b", 11, "X")])]
        body = generate_markdown_body(grouped, "talk")
        assert body.count("**X**:") == 2


class TestLocalTranscriptSource:
    def _meta(self, video: Path) -> LocalVideoMetadata:
        return LocalVideoMetadata(
            file_path=video, _base_title="talk", duration=60.0, creation_date="20261006"
        )

    def _quiet(self):
        from rich.console import Console

        return Console(quiet=True)

    def test_sidecar_preferred_over_embedded(self, tmp_path, monkeypatch):
        video = tmp_path / "talk.mp4"
        video.write_bytes(b"x")
        (tmp_path / "talk-en-US.vtt").write_text(TEAMS_VTT)
        monkeypatch.setattr(
            "vidflow.capture.subtitles.probe_subtitle_streams",
            lambda p: pytest.fail("embedded probe should not run"),
        )
        meta = self._meta(video)
        segs = core._load_local_transcript(video, meta, None, self._quiet())
        assert len(segs) == 3
        assert meta.subtitle_source == "sidecar talk-en-US.vtt"

    def test_explicit_track_skips_sidecar(self, tmp_path, monkeypatch):
        video = tmp_path / "talk.mp4"
        video.write_bytes(b"x")
        (tmp_path / "talk.vtt").write_text(TEAMS_VTT)
        probed = []
        monkeypatch.setattr(
            "vidflow.capture.subtitles.probe_subtitle_streams",
            lambda p: probed.append(p) or [],
        )
        assert core._load_local_transcript(video, self._meta(video), 0, self._quiet()) is None
        assert probed == [video]

    def test_empty_sidecar_falls_back(self, tmp_path, monkeypatch):
        video = tmp_path / "talk.mp4"
        video.write_bytes(b"x")
        (tmp_path / "talk.vtt").write_text("WEBVTT\n")
        probed = []
        monkeypatch.setattr(
            "vidflow.capture.subtitles.probe_subtitle_streams",
            lambda p: probed.append(p) or [],
        )
        core._load_local_transcript(video, self._meta(video), None, self._quiet())
        assert probed == [video]


TEAMS_STEM = "BRAIN NeuroAI WG Monthly Meeting-20261008_145749UTC-Meeting Recording"


class TestTeamsMeetingName:
    @pytest.mark.parametrize(
        "stem, expected",
        [
            (TEAMS_STEM, "BRAIN NeuroAI WG Monthly Meeting"),
            ("Sync-20261008_145749-Meeting Recording", "Sync"),
            ("Q4 Review - Budget-20261008_145749UTC-Meeting Recording", "Q4 Review - Budget"),
            ("Screen Recording 2026-10-06 124555_Duygu Kuzum", None),
            ("Meeting Recording", None),
        ],
    )
    def test_parse(self, stem, expected):
        assert teams_meeting_name(stem) == expected

    def _video(self, tmp_path: Path) -> Path:
        video = tmp_path / f"{TEAMS_STEM}.mp4"
        video.write_bytes(b"x")
        return video

    @pytest.mark.parametrize(
        "name",
        ["BRAIN NeuroAI WG Monthly Meeting.vtt", "BRAIN NeuroAI WG Monthly Meeting-en-US.vtt"],
    )
    def test_finds_meeting_name_sidecar(self, tmp_path, name):
        video = self._video(tmp_path)
        (tmp_path / name).write_text("WEBVTT\n")
        sidecar = find_sidecar_vtt(video)
        assert sidecar.name == name
        assert is_meeting_name_match(video, sidecar)

    def test_stem_match_preferred_over_meeting_name(self, tmp_path):
        video = self._video(tmp_path)
        (tmp_path / "BRAIN NeuroAI WG Monthly Meeting.vtt").write_text("WEBVTT\n")
        (tmp_path / f"{TEAMS_STEM}.vtt").write_text("WEBVTT\n")
        sidecar = find_sidecar_vtt(video)
        assert sidecar.name == f"{TEAMS_STEM}.vtt"
        assert not is_meeting_name_match(video, sidecar)

    def test_other_meeting_not_matched(self, tmp_path):
        video = self._video(tmp_path)
        (tmp_path / "BRAIN NeuroAI WG Monthly.vtt").write_text("WEBVTT\n")
        assert find_sidecar_vtt(video) is None


def _vtt_ending_at(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    return f"WEBVTT\n\n00:00:01.000 --> 00:{m:02d}:{s:02d}.000\n<v Chair>Welcome.</v>\n"


class TestMeetingSidecarGuard:
    """A recurring meeting's transcript name can belong to another month's recording."""

    def _setup(self, tmp_path, vtt_end: int, monkeypatch):
        video = tmp_path / f"{TEAMS_STEM}.mp4"
        video.write_bytes(b"x")
        (tmp_path / "BRAIN NeuroAI WG Monthly Meeting.vtt").write_text(_vtt_ending_at(vtt_end))
        probed = []
        monkeypatch.setattr(
            "vidflow.capture.subtitles.probe_subtitle_streams",
            lambda p: probed.append(p) or [],
        )
        meta = LocalVideoMetadata(
            file_path=video, _base_title="m", duration=600.0, creation_date="20261008"
        )
        return video, meta, probed

    def _quiet(self):
        from rich.console import Console

        return Console(quiet=True)

    def test_meeting_sidecar_within_video_accepted(self, tmp_path, monkeypatch):
        video, meta, probed = self._setup(tmp_path, 590, monkeypatch)
        segs = core._load_local_transcript(video, meta, None, self._quiet())
        assert segs and segs[0].speaker == "Chair"
        assert meta.subtitle_source == "Teams meeting sidecar BRAIN NeuroAI WG Monthly Meeting.vtt"
        assert probed == []

    def test_meeting_sidecar_outrunning_video_rejected(self, tmp_path, monkeypatch):
        video, meta, probed = self._setup(tmp_path, 3000, monkeypatch)
        assert core._load_local_transcript(video, meta, None, self._quiet()) is None
        assert probed == [video]  # fell back to embedded tracks


class TestExplicitVtt:
    def _meta(self, video: Path, duration: float = 600.0) -> LocalVideoMetadata:
        return LocalVideoMetadata(
            file_path=video, _base_title="t", duration=duration, creation_date="20261008"
        )

    def _quiet(self):
        from rich.console import Console

        return Console(quiet=True)

    def test_overrides_sidecar_and_embedded(self, tmp_path, monkeypatch):
        video = tmp_path / "talk.mp4"
        video.write_bytes(b"x")
        (tmp_path / "talk.vtt").write_text(TEAMS_VTT)
        explicit = tmp_path / "elsewhere.vtt"
        explicit.write_text(_vtt_ending_at(30))
        monkeypatch.setattr(
            "vidflow.capture.subtitles.probe_subtitle_streams",
            lambda p: pytest.fail("embedded probe should not run"),
        )
        meta = self._meta(video)
        segs = core._load_local_transcript(video, meta, None, self._quiet(), vtt=explicit)
        assert [s.speaker for s in segs] == ["Chair"]
        assert meta.subtitle_source == "--vtt elsewhere.vtt"

    def test_overrun_warns_but_is_used(self, tmp_path):
        video = tmp_path / "talk.mp4"
        explicit = tmp_path / "long.vtt"
        explicit.write_text(_vtt_ending_at(3000))
        segs = core._load_local_transcript(
            video, self._meta(video), None, self._quiet(), vtt=explicit
        )
        assert cues_end(segs) == pytest.approx(3000.0)

    def test_empty_vtt_fails(self, tmp_path):
        video = tmp_path / "talk.mp4"
        explicit = tmp_path / "empty.vtt"
        explicit.write_text("WEBVTT\n")
        with pytest.raises(LocalVideoError, match="no cues"):
            core._load_local_transcript(video, self._meta(video), None, self._quiet(), vtt=explicit)


class TestVttCli:
    @pytest.fixture
    def files(self, tmp_path):
        a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
        a.write_bytes(b"x")
        b.write_bytes(b"x")
        vtt = tmp_path / "t.vtt"
        vtt.write_text(TEAMS_VTT)
        return a, b, vtt

    @pytest.mark.parametrize(
        "extra, message",
        [
            (["MULTI"], "single video"),
            (["--no-subtitles"], "cannot be combined"),
            (["--subtitle-track", "0"], "cannot be combined"),
            (["MISSING"], "not found"),
        ],
    )
    def test_usage_errors(self, files, extra, message, capsys):
        from vidflow.cli import main as vidflow_main

        a, b, vtt = files
        vtt_arg = str(vtt.with_name("nope.vtt")) if extra == ["MISSING"] else str(vtt)
        argv = ["local", "--dry-run", "--vtt", vtt_arg, str(a)]
        if extra == ["MULTI"]:
            argv.append(str(b))
        elif extra != ["MISSING"]:
            argv += extra
        assert vidflow_main(argv) == 2
        assert message in capsys.readouterr().err
