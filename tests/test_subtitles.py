"""Tests for embedded subtitle detection, parsing, and sanitization."""

import pytest

from vidflow.capture.subtitles import (
    SubtitleError,
    SubtitleStream,
    parse_webvtt,
    sanitize_cue_text,
    select_subtitle_stream,
)

# --- sanitize_cue_text ---


class TestSanitizeCueText:
    def test_strips_font_tags(self):
        assert sanitize_cue_text('<font color="red">hello</font>') == "hello"

    def test_strips_nested_html_tags(self):
        assert sanitize_cue_text("<b><i>bold italic</i></b>") == "bold italic"

    def test_strips_webvtt_class_tags(self):
        assert sanitize_cue_text("<c.classname>classified</c>") == "classified"

    def test_preserves_speaker_labels_angle_v(self):
        # <v Speaker> is a WebVTT voice tag — the tag itself is stripped,
        # but ">> JOHN:" style speaker labels survive since they aren't tags.
        assert sanitize_cue_text(">> JOHN: hello there") == ">> JOHN: hello there"

    def test_preserves_bracket_speaker_cues(self):
        assert sanitize_cue_text("[Music] intro") == "[Music] intro"

    def test_strips_inline_timestamp_tags(self):
        text = "start<00:00:05.000>middle<00:00:08.500>end"
        assert sanitize_cue_text(text) == "startmiddleend"

    def test_strips_ass_override_blocks(self):
        assert sanitize_cue_text("{\\an8}top text") == "top text"
        assert sanitize_cue_text("{\\pos(100,200)}positioned") == "positioned"

    def test_decodes_html_entities(self):
        assert sanitize_cue_text("Tom &amp; Jerry") == "Tom & Jerry"
        assert sanitize_cue_text("&#39;quoted&#39;") == "'quoted'"

    def test_collapses_newlines_within_cue(self):
        assert sanitize_cue_text("line one\nline two") == "line one line two"

    def test_trims_whitespace(self):
        assert sanitize_cue_text("   spaced   ") == "spaced"

    def test_combined_markup(self):
        text = '<font color="#ffffff"><i>Hello</i></font>\n<b>world</b> &amp; more'
        assert sanitize_cue_text(text) == "Hello world & more"


# --- parse_webvtt ---


class TestParseWebVTT:
    def test_parses_basic_cues(self):
        vtt = (
            "WEBVTT\n"
            "\n"
            "00:00:01.000 --> 00:00:04.500\n"
            "First line\n"
            "\n"
            "00:00:05.000 --> 00:00:08.000\n"
            "Second line\n"
        )
        segments = parse_webvtt(vtt)
        assert len(segments) == 2
        assert segments[0].text == "First line"
        assert segments[0].start == pytest.approx(1.0)
        assert segments[0].duration == pytest.approx(3.5)
        assert segments[1].text == "Second line"
        assert segments[1].start == pytest.approx(5.0)

    def test_accepts_mm_ss_timestamps(self):
        vtt = "WEBVTT\n\n01:30.000 --> 01:32.500\nShort form\n"
        segments = parse_webvtt(vtt)
        assert len(segments) == 1
        assert segments[0].start == pytest.approx(90.0)
        assert segments[0].duration == pytest.approx(2.5)

    def test_strips_markup_in_cues(self):
        vtt = "WEBVTT\n\n" "00:00:01.000 --> 00:00:02.000\n" '<font color="red">red text</font>\n'
        segments = parse_webvtt(vtt)
        assert segments[0].text == "red text"

    def test_multi_line_cue_joined(self):
        vtt = "WEBVTT\n\n" "00:00:01.000 --> 00:00:03.000\n" "First part\n" "continues here\n"
        segments = parse_webvtt(vtt)
        assert len(segments) == 1
        assert segments[0].text == "First part continues here"

    def test_empty_cue_dropped(self):
        vtt = (
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:02.000\n"
            "<c></c>\n"
            "\n"
            "00:00:03.000 --> 00:00:04.000\n"
            "kept\n"
        )
        segments = parse_webvtt(vtt)
        assert len(segments) == 1
        assert segments[0].text == "kept"

    def test_crlf_line_endings(self):
        vtt = "WEBVTT\r\n\r\n" "00:00:01.000 --> 00:00:02.000\r\n" "crlf\r\n"
        segments = parse_webvtt(vtt)
        assert len(segments) == 1
        assert segments[0].text == "crlf"

    def test_cue_identifier_line_ignored(self):
        vtt = "WEBVTT\n\n" "cue-1\n" "00:00:01.000 --> 00:00:02.000\n" "text\n"
        segments = parse_webvtt(vtt)
        assert len(segments) == 1
        assert segments[0].text == "text"


# --- select_subtitle_stream ---


def _stream(
    sub_idx: int,
    codec: str = "mov_text",
    language: str | None = "eng",
    default: bool = False,
    forced: bool = False,
    title: str | None = None,
    hi: bool = False,
) -> SubtitleStream:
    return SubtitleStream(
        index=sub_idx,
        subtitle_index=sub_idx,
        codec=codec,
        language=language,
        title=title,
        is_default=default,
        is_forced=forced,
        is_hearing_impaired=hi,
    )


class TestSelectSubtitleStream:
    def test_empty_returns_none(self):
        assert select_subtitle_stream([]) is None

    def test_explicit_track_selects(self):
        streams = [_stream(0), _stream(1), _stream(2)]
        assert select_subtitle_stream(streams, track=1).subtitle_index == 1

    def test_explicit_track_missing_raises(self):
        with pytest.raises(SubtitleError, match="not found"):
            select_subtitle_stream([_stream(0)], track=5)

    def test_prefers_default_english(self):
        streams = [
            _stream(0, language="fre"),
            _stream(1, language="eng", default=True),
            _stream(2, language="eng"),
        ]
        assert select_subtitle_stream(streams).subtitle_index == 1

    def test_falls_back_to_any_english(self):
        streams = [
            _stream(0, language="fre", default=True),
            _stream(1, language="eng"),
        ]
        assert select_subtitle_stream(streams).subtitle_index == 1

    def test_avoids_forced_track_when_non_forced_exists(self):
        streams = [
            _stream(0, language="eng", forced=True, default=True),
            _stream(1, language="eng"),
        ]
        assert select_subtitle_stream(streams).subtitle_index == 1

    def test_falls_back_to_default_any_language(self):
        streams = [
            _stream(0, language="fre"),
            _stream(1, language="spa", default=True),
        ]
        assert select_subtitle_stream(streams).subtitle_index == 1

    def test_falls_back_to_first_text_track(self):
        streams = [
            _stream(0, language="fre"),
            _stream(1, language="spa"),
        ]
        assert select_subtitle_stream(streams).subtitle_index == 0

    def test_skips_bitmap_codecs(self):
        streams = [
            _stream(0, codec="hdmv_pgs_subtitle", language="eng", default=True),
            _stream(1, codec="mov_text", language="fre"),
        ]
        # Bitmap English isn't text-based, so French mov_text wins
        assert select_subtitle_stream(streams).subtitle_index == 1

    def test_returns_none_when_only_bitmap_available(self):
        streams = [_stream(0, codec="hdmv_pgs_subtitle", language="eng")]
        assert select_subtitle_stream(streams) is None

    def test_accepts_language_prefix_match(self):
        # ffprobe sometimes reports "eng", sometimes "en"
        streams = [_stream(0, language="en")]
        assert select_subtitle_stream(streams).subtitle_index == 0

    def test_matches_en_us_as_english(self):
        streams = [_stream(0, language="en-US")]
        assert select_subtitle_stream(streams).subtitle_index == 0
