"""Tests for multi-input merge behavior: part tagging, batching, output structure."""

from pathlib import Path

import pytest

from vidflow.cli import main as vidflow_main
from vidflow.transcribe.parser import (
    merge_vidcapture_documents,
    parse_vidcapture_markdown,
)
from vidflow.transcribe.processor import VidscribeProcessor


def _capture_md(title: str) -> str:
    return (
        f"---\ntitle: {title}\n---\n\n"
        f"# {title}\n\n"
        "## 00:00:00\n\n![[images/x/frame-000.jpg]]\n\nraw captions part one\n\n"
        "## 00:00:15\n\n![[images/x/frame-001.jpg]]\n\nraw captions part two\n"
    )


@pytest.fixture
def two_captures(tmp_path) -> list[Path]:
    paths = []
    for name in ("Morning Session", "Afternoon Session"):
        p = tmp_path / f"{name}.md"
        p.write_text(_capture_md(name), encoding="utf-8")
        paths.append(p)
    return paths


@pytest.fixture
def no_warm(monkeypatch):
    monkeypatch.setattr("vidflow.transcribe.processor.aikit.warm", lambda *a, **k: None)


class TestMergeParts:
    """merge_vidcapture_documents tags sections with their source part."""

    def test_part_fields_set(self, two_captures):
        docs = [parse_vidcapture_markdown(p) for p in two_captures]
        merged = merge_vidcapture_documents(docs)

        assert [s.part_index for s in merged.sections] == [0, 0, 1, 1]
        assert merged.sections[0].part_title == "Morning Session"
        assert merged.sections[2].part_title == "Afternoon Session"
        # Timestamps deliberately repeat across parts
        assert [s.timestamp for s in merged.sections] == [
            "00:00:00",
            "00:00:15",
            "00:00:00",
            "00:00:15",
        ]

    def test_single_document_untagged(self, two_captures):
        doc = parse_vidcapture_markdown(two_captures[0])
        merged = merge_vidcapture_documents([doc])
        assert merged is doc
        assert all(s.part_index == 0 and s.part_title == "" for s in doc.sections)


class TestMergedProcessing:
    """process_all keeps parts in separate batches and emits per-part H1s."""

    @pytest.fixture
    def processor(self, no_warm) -> VidscribeProcessor:
        return VidscribeProcessor(
            api_key=None,
            model="primary",
            batch_size=10,  # far larger than any part: split must come from parts
            json_output=True,
            text_only=True,
        )

    @pytest.fixture
    def batch_calls(self, monkeypatch):
        calls = []

        def fake_batch(self, sections, previous_sections, temp_dir, progress, bn, tb):
            calls.append(
                {
                    "timestamps": [s.timestamp for s in sections],
                    "parts": sorted({s.part_index for s in sections}),
                    "context_parts": sorted({s.part_index for s in previous_sections}),
                }
            )
            return [f"polished {s.part_index}/{s.timestamp}" for s in sections]

        monkeypatch.setattr(VidscribeProcessor, "process_markdown_batch", fake_batch)
        return calls

    def test_batches_never_straddle_parts(self, processor, batch_calls, two_captures):
        docs = [parse_vidcapture_markdown(p) for p in two_captures]
        merged = merge_vidcapture_documents(docs)

        transcript, _ = processor.process_all(merged, with_frontmatter=False)

        # Two parts of 2 sections with batch_size 10 -> exactly 2 batches,
        # one per part, and the second batch starts with no carried context
        assert len(batch_calls) == 2
        assert batch_calls[0]["parts"] == [0]
        assert batch_calls[1]["parts"] == [1]
        assert batch_calls[1]["context_parts"] == []

    def test_merged_transcript_structure(self, processor, batch_calls, two_captures):
        docs = [parse_vidcapture_markdown(p) for p in two_captures]
        merged = merge_vidcapture_documents(docs)

        transcript, _ = processor.process_all(merged, with_frontmatter=False)
        lines = transcript.splitlines()

        # One H1 per original file, in order, with H2 timestamps restarting
        h1s = [ln for ln in lines if ln.startswith("# ")]
        assert h1s == ["# Morning Session", "# Afternoon Session"]
        h2s = [ln for ln in lines if ln.startswith("## ")]
        assert h2s == ["## 00:00:00", "## 00:00:15", "## 00:00:00", "## 00:00:15"]
        # Content matched to the right part despite duplicate timestamps
        assert "polished 0/00:00:00" in transcript
        assert "polished 1/00:00:00" in transcript

    def test_single_input_has_no_part_headings(self, processor, batch_calls, two_captures):
        doc = parse_vidcapture_markdown(two_captures[0])

        transcript, _ = processor.process_all(doc, with_frontmatter=False)

        assert not any(ln.startswith("# ") for ln in transcript.splitlines())


class TestYoutubeMergeRemoved:
    """--merge is no longer accepted on the youtube subcommand."""

    def test_youtube_merge_rejected(self):
        with pytest.raises(SystemExit) as exc_info:
            vidflow_main(["youtube", "URL", "--merge"])
        assert exc_info.value.code == 2

    def test_local_merge_still_accepted(self, tmp_path, capsys):
        # Parses fine; fails later only because the file doesn't exist
        missing = tmp_path / "missing.mp4"
        result = vidflow_main(["local", str(missing), "--merge", "--json"])
        assert result == 1


class TestOutputPathDirectories:
    """determine_output_path accepts a directory or a file target."""

    def test_directory_target_gets_auto_named_file(self, tmp_path):
        from vidflow.transcribe.output import determine_output_path

        out_dir = tmp_path / "notes"
        out_dir.mkdir()
        result = determine_output_path(
            input_path=tmp_path / "capture.md",
            title="My Talk",
            explicit_output=out_dir,
        )
        assert result == (out_dir / "My Talk.md").resolve()

    def test_file_target_used_verbatim(self, tmp_path):
        from vidflow.transcribe.output import determine_output_path

        target = tmp_path / "out.md"
        result = determine_output_path(
            input_path=tmp_path / "capture.md",
            title="My Talk",
            explicit_output=target,
        )
        assert result == target.resolve()

    def test_no_target_lands_beside_input(self, tmp_path):
        from vidflow.transcribe.output import determine_output_path

        result = determine_output_path(
            input_path=tmp_path / "capture.md",
            title="My Talk",
        )
        assert result == (tmp_path / "My Talk.md").resolve()
