"""Tests for archiving capture notes once a transcript replaces them."""

import pytest

from vidflow.transcribe import transcribe_markdown
from vidflow.transcribe.output import archive_capture
from vidflow.transcribe.processor import VidscribeProcessor

from tests.test_polish import CAPTURE_MD

TRANSCRIPT = (
    "## 00:00:00\n![[images/test/frame-000.jpg]]\n\n[Title slide]\n\n"
    "**Speaker**: Hello, this is transcribed text.\n"
)


@pytest.fixture
def no_warm(monkeypatch):
    monkeypatch.setattr("vidflow.transcribe.processor.aikit.warm", lambda *a, **k: None)


@pytest.fixture
def fake_process_all(monkeypatch):
    def fake(self, document, checkpoint_path=None, input_paths=None, with_frontmatter=True):
        return TRANSCRIPT, {"title": "Generated Title", "tags": ["gen"], "created": "2026-09-19"}

    monkeypatch.setattr(VidscribeProcessor, "process_all", fake)


class TestArchiveCapture:
    def test_moves_into_sibling_transcripts_dir(self, tmp_path):
        note = tmp_path / "capture.md"
        note.write_text("x", encoding="utf-8")
        target = archive_capture(note)
        assert target == tmp_path / "transcripts" / "capture.md"
        assert target.read_text(encoding="utf-8") == "x"
        assert not note.exists()

    def test_never_overwrites_existing_archive(self, tmp_path):
        archive = tmp_path / "transcripts"
        archive.mkdir()
        (archive / "capture.md").write_text("old", encoding="utf-8")
        note = tmp_path / "capture.md"
        note.write_text("new", encoding="utf-8")
        target = archive_capture(note)
        assert target.name == "capture-1.md"
        assert (archive / "capture.md").read_text(encoding="utf-8") == "old"


class TestTranscribeArchives:
    def test_transcribe_replaces_capture_with_transcript(self, no_warm, fake_process_all, tmp_path):
        note = tmp_path / "capture.md"
        note.write_text(CAPTURE_MD, encoding="utf-8")

        result = transcribe_markdown([note], json_output=True)

        assert result.success, result.message
        out = tmp_path / "Generated Title.md"
        assert out.exists()
        assert "this is transcribed text" in out.read_text(encoding="utf-8")
        archived = tmp_path / "transcripts" / "capture.md"
        assert result.data["archived"] == [str(archived)]
        assert "raw caption text" in archived.read_text(encoding="utf-8")
        # The folder now holds exactly one note for the video
        assert [p.name for p in tmp_path.glob("*.md")] == ["Generated Title.md"]
        assert "transcripts/" in result.message

    def test_output_on_input_path_archives_first(self, no_warm, fake_process_all, tmp_path):
        """A capture already named after the title is replaced, not lost."""
        note = tmp_path / "Generated Title.md"
        note.write_text(CAPTURE_MD, encoding="utf-8")

        result = transcribe_markdown([note], json_output=True)

        assert result.success, result.message
        assert "this is transcribed text" in note.read_text(encoding="utf-8")
        archived = tmp_path / "transcripts" / "Generated Title.md"
        assert "raw caption text" in archived.read_text(encoding="utf-8")

    def test_keep_capture(self, no_warm, fake_process_all, tmp_path):
        note = tmp_path / "capture.md"
        note.write_text(CAPTURE_MD, encoding="utf-8")

        result = transcribe_markdown([note], keep_capture=True, json_output=True)

        assert result.success, result.message
        assert note.exists()
        assert result.data["archived"] == []
        assert not (tmp_path / "transcripts").exists()
