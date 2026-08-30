"""Tests for the text-only polish mode (vidflow polish / --polish)."""

import json
from pathlib import Path

import pytest

from vidflow.cli import main as vidflow_main
from vidflow.transcribe import polish_markdown
from vidflow.transcribe.models import TimestampSection
from vidflow.transcribe.parser import parse_vidcapture_markdown
from vidflow.transcribe.processor import VidscribeProcessor
from vidflow.transcribe.prompts import POLISH_PROMPT, TEMPLATE_FILL_PROMPT

CAPTURE_MD = """\
---
title: Test Capture
---

# Test Capture

## 00:00:00

![[images/test/frame-000.jpg]]

hello uh this is raw caption text with um filler words

## 00:00:15

![[images/test/frame-001.jpg]]

## 00:00:30

![[images/test/frame-002.jpg]]

more raw caption text continuing the talk
"""

FRAMES_ONLY_MD = """\
---
title: Frames Only
---

# Frames Only

## 00:00:00

![[images/test/frame-000.jpg]]

## 00:00:15

![[images/test/frame-001.jpg]]
"""


@pytest.fixture
def capture_file(tmp_path) -> Path:
    md = tmp_path / "capture.md"
    md.write_text(CAPTURE_MD, encoding="utf-8")
    return md


@pytest.fixture
def frames_only_file(tmp_path) -> Path:
    md = tmp_path / "frames-only.md"
    md.write_text(FRAMES_ONLY_MD, encoding="utf-8")
    return md


@pytest.fixture
def no_warm(monkeypatch):
    """Prevent the fire-and-forget gateway warm-up during tests."""
    monkeypatch.setattr("vidflow.transcribe.processor.aikit.warm", lambda *a, **k: None)


@pytest.fixture
def text_only_processor(no_warm) -> VidscribeProcessor:
    return VidscribeProcessor(
        api_key=None,
        model="primary",
        json_output=True,
        text_only=True,
    )


class TestTextOnlyProcessor:
    """Unit tests for VidscribeProcessor text_only mode."""

    def test_polish_prompt_selected(self, text_only_processor):
        assert text_only_processor._get_batch_prompt() == POLISH_PROMPT

    def test_vision_prompt_selected_by_default(self, no_warm):
        processor = VidscribeProcessor(api_key=None, model="primary", json_output=True)
        assert processor._get_batch_prompt() == TEMPLATE_FILL_PROMPT

    def test_exa_disabled_in_text_only(self, no_warm):
        processor = VidscribeProcessor(
            api_key=None,
            model="primary",
            json_output=True,
            exa_api_key="fake-key",
            text_only=True,
        )
        assert processor.exa_enabled is False

    def test_estimate_excludes_image_tokens(self, no_warm, text_only_processor):
        sections = [
            TimestampSection(
                timestamp="00:00:00",
                image_embed="![[images/test/frame-000.jpg]]",
                image_path=Path("/nonexistent/frame-000.jpg"),
                existing_text="some caption text",
            )
        ]
        vision_processor = VidscribeProcessor(api_key=None, model="primary", json_output=True)
        text_estimate = text_only_processor.estimate_tokens(sections)
        vision_estimate = vision_processor.estimate_tokens(sections)
        assert vision_estimate - text_estimate >= 1200

    def test_template_includes_existing_transcript(self, text_only_processor, capture_file):
        document = parse_vidcapture_markdown(capture_file)
        template = text_only_processor._build_batch_template(document.sections)
        assert "<existing-transcript>" in template
        assert "hello uh this is raw caption text" in template
        # The captionless middle section keeps heading + embed with no tags
        assert "## 00:00:15" in template
        blocks = template.split("## ")
        middle = next(b for b in blocks if b.startswith("00:00:15"))
        assert "<existing-transcript>" not in middle

    def test_parse_batch_response_passthrough(self, text_only_processor):
        sections = [
            TimestampSection(
                timestamp="00:00:00",
                image_embed="![[images/test/frame-000.jpg]]",
                image_path=Path("/nonexistent/frame-000.jpg"),
                existing_text="raw",
            ),
            TimestampSection(
                timestamp="00:00:15",
                image_embed="![[images/test/frame-001.jpg]]",
                image_path=Path("/nonexistent/frame-001.jpg"),
            ),
        ]
        response = (
            "## 00:00:00\n"
            "![[images/test/frame-000.jpg]]\n\n"
            "Polished caption text.\n\n"
            "## 00:00:15\n"
            "![[images/test/frame-001.jpg]]\n"
        )
        contents = text_only_processor._parse_batch_response(response, sections)
        assert contents[0] == "Polished caption text."
        assert contents[1] == ""


class TestPolishMarkdown:
    """Tests for the polish_markdown wrapper (no API calls)."""

    def test_dry_run_counts_sections(self, no_warm, capture_file):
        result = polish_markdown([capture_file], dry_run=True, json_output=True)
        assert result.success
        assert result.data["sections"] == 3
        assert result.data["sections_with_text"] == 2

    def test_estimate_only(self, no_warm, capture_file):
        result = polish_markdown([capture_file], estimate_only=True, json_output=True)
        assert result.success
        assert result.data["estimate"] > 0

    def test_frames_only_capture_rejected(self, no_warm, frames_only_file):
        result = polish_markdown([frames_only_file], dry_run=True, json_output=True)
        assert not result.success
        assert "No caption text" in result.message

    def test_anthropic_lane_requires_key(self, monkeypatch, capture_file):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = polish_markdown(
            [capture_file], model="claude-opus-5", dry_run=True, json_output=True
        )
        assert not result.success
        assert "ANTHROPIC_API_KEY" in result.message

    def test_local_lane_needs_no_key(self, no_warm, monkeypatch, capture_file):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = polish_markdown([capture_file], dry_run=True, json_output=True)
        assert result.success


class TestInPlacePolish:
    """Tests for the in-place update mode (single input, no -o)."""

    POLISHED = (
        "## 00:00:00\n"
        "![[images/test/frame-000.jpg]]\n\n"
        "Hello, this is polished caption text.\n\n"
        "## 00:00:15\n"
        "![[images/test/frame-001.jpg]]\n\n"
        "## 00:00:30\n"
        "![[images/test/frame-002.jpg]]\n\n"
        "More polished caption text continuing the talk.\n"
    )

    @pytest.fixture
    def fake_process_all(self, monkeypatch):
        calls = {}

        def fake(self, document, checkpoint_path=None, input_paths=None, with_frontmatter=True):
            calls["with_frontmatter"] = with_frontmatter
            frontmatter = {"title": "Generated Title", "tags": ["gen"]} if with_frontmatter else {}
            return TestInPlacePolish.POLISHED, frontmatter

        monkeypatch.setattr(VidscribeProcessor, "process_all", fake)
        return calls

    def test_updates_input_file(self, no_warm, fake_process_all, capture_file):
        result = polish_markdown([capture_file], json_output=True)

        assert result.success
        assert result.data["in_place"] is True
        assert result.data["output_path"] == str(capture_file.resolve())
        content = capture_file.read_text(encoding="utf-8")
        # Original frontmatter and title preserved verbatim
        assert "title: Test Capture" in content
        assert "# Test Capture" in content
        # Section text replaced with polished output
        assert "Hello, this is polished caption text." in content
        assert "hello uh this is raw caption text" not in content
        # No frontmatter generation for in-place mode
        assert fake_process_all["with_frontmatter"] is False
        # No second file created
        siblings = [p.name for p in capture_file.parent.glob("*.md")]
        assert siblings == [capture_file.name]

    def test_explicit_output_writes_new_file(
        self, no_warm, fake_process_all, capture_file, tmp_path
    ):
        out = tmp_path / "polished.md"
        result = polish_markdown([capture_file], output=out, json_output=True)

        assert result.success
        assert result.data["in_place"] is False
        assert fake_process_all["with_frontmatter"] is True
        assert out.exists()
        # Original file untouched
        assert "hello uh this is raw caption text" in capture_file.read_text(encoding="utf-8")
        # Generated frontmatter merged over original: both survive
        out_content = out.read_text(encoding="utf-8")
        assert "Generated Title" in out_content
        assert "gen" in out_content

    def test_multiple_inputs_write_new_file(
        self, no_warm, fake_process_all, capture_file, tmp_path
    ):
        second = tmp_path / "capture2.md"
        second.write_text(CAPTURE_MD, encoding="utf-8")

        result = polish_markdown([capture_file, second], json_output=True)

        assert result.success
        assert result.data["in_place"] is False
        # Originals untouched
        assert "hello uh this is raw caption text" in capture_file.read_text(encoding="utf-8")

    def test_dry_run_reports_in_place(self, no_warm, capture_file):
        result = polish_markdown([capture_file], dry_run=True, json_output=True)
        assert result.data["in_place"] is True
        assert result.data["target"] == str(capture_file.resolve())


class TestMergeFrontmatter:
    """Tests for merge_frontmatter (original preserved under generated)."""

    def test_preserves_original_keys(self):
        from vidflow.transcribe import merge_frontmatter

        original = (
            "title: Original\npublished: 2026-01-30\nsource: https://youtu.be/x\n"
            "author:\n- Someone\ntags:\n- youtube\n"
        )
        generated = {
            "title": "Generated",
            "created": "2026-08-29",
            "tags": ["health", "youtube"],
            "description": "A talk.",
        }
        merged = merge_frontmatter(original, generated)

        import datetime

        assert merged["title"] == "Generated"
        # PyYAML parses unquoted dates as date objects (re-dumps unchanged)
        assert merged["published"] == datetime.date(2026, 1, 30)
        assert merged["source"] == "https://youtu.be/x"
        assert merged["author"] == ["Someone"]
        assert merged["tags"] == ["youtube", "health"]
        assert merged["description"] == "A talk."

    def test_empty_original_returns_generated(self):
        from vidflow.transcribe import merge_frontmatter

        generated = {"title": "Generated"}
        assert merge_frontmatter("", generated) == generated
        assert merge_frontmatter(None, generated) == generated

    def test_unparseable_original_returns_generated(self):
        from vidflow.transcribe import merge_frontmatter

        generated = {"title": "Generated"}
        assert merge_frontmatter("{unclosed: [", generated) == generated


class TestPolishCli:
    """Integration tests for the polish subcommand."""

    def test_help_lists_polish(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            vidflow_main(["--help"])
        assert exc_info.value.code == 0
        assert "polish" in capsys.readouterr().out

    def test_dry_run_json(self, no_warm, capture_file, capsys):
        result = vidflow_main(["polish", str(capture_file), "--dry-run", "--json"])
        assert result == 0
        data = json.loads(capsys.readouterr().out)
        assert data["success"] is True
        assert data["data"]["sections_with_text"] == 2

    def test_no_max_dimension_flag(self, capture_file):
        with pytest.raises(SystemExit) as exc_info:
            vidflow_main(["polish", str(capture_file), "--max-dimension", "800"])
        assert exc_info.value.code == 2

    def test_transcribe_polish_mutually_exclusive(self):
        for subcmd, target in (("youtube", "URL"), ("local", "f.mp4")):
            with pytest.raises(SystemExit) as exc_info:
                vidflow_main([subcmd, target, "--transcribe", "--polish"])
            assert exc_info.value.code == 2
