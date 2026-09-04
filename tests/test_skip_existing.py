"""Tests for skip-existing on YouTube captures."""

from pathlib import Path

import pytest

from vidflow.capture.core import CaptureExists, find_existing_capture, process_video

VID = "abcdefghijk"
URL = f"https://www.youtube.com/watch?v={VID}"


def _note(path: Path, source: str | None, body: str = "# Title\n") -> Path:
    fm = "---\ntitle: T\ncreated: 2026-01-01\n"
    if source is not None:
        fm += f"source: {source}\n"
    fm += "tags:\n  - youtube\n---\n"
    path.write_text(fm + body, encoding="utf-8")
    return path


class TestFindExistingCapture:
    def test_matches_by_video_id_in_source(self, tmp_path):
        note = _note(tmp_path / "Some AI Title.md", f"https://youtu.be/{VID}")
        _note(tmp_path / "Other.md", "https://www.youtube.com/watch?v=zzzzzzzzzzz")
        assert find_existing_capture(tmp_path, VID) == note

    def test_quoted_source_and_list_param(self, tmp_path):
        note = _note(tmp_path / "n.md", f'"https://www.youtube.com/watch?v={VID}&list=PLx"')
        assert find_existing_capture(tmp_path, VID) == note

    def test_no_match(self, tmp_path):
        _note(tmp_path / "n.md", None)
        (tmp_path / "plain.md").write_text("no frontmatter\n", encoding="utf-8")
        assert find_existing_capture(tmp_path, VID) is None

    def test_missing_dir_or_empty_id(self, tmp_path):
        assert find_existing_capture(tmp_path / "nope", VID) is None
        assert find_existing_capture(tmp_path, "") is None

    def test_body_source_line_is_not_frontmatter(self, tmp_path):
        _note(tmp_path / "n.md", None, body=f"# T\nsource: {URL}\n")
        assert find_existing_capture(tmp_path, VID) is None


class TestProcessVideoSkip:
    def _kwargs(self, out):
        return dict(
            url=URL,
            output_dir=out,
            interval=15,
            max_frames=None,
            frame_format="jpg",
            language="en",
            prefer_manual=False,
            dedup_threshold=0.95,
            no_dedup=False,
            keep_video=False,
            no_ai_title=True,
        )

    def test_existing_note_raises_before_any_request(self, tmp_path, monkeypatch):
        note = _note(tmp_path / "Existing.md", URL)
        calls = []
        monkeypatch.setattr(
            "vidflow.capture.core.fetch_video",
            lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(RuntimeError),
        )
        with pytest.raises(CaptureExists) as exc:
            process_video(**self._kwargs(tmp_path))
        assert exc.value.path == note
        assert calls == []
        assert not (tmp_path / "videos").exists()

    def test_force_proceeds_to_fetch(self, tmp_path, monkeypatch):
        _note(tmp_path / "Existing.md", URL)

        def fake_fetch(*a, **k):
            raise RuntimeError("fetch reached")

        monkeypatch.setattr("vidflow.capture.core.fetch_video", fake_fetch)
        with pytest.raises(RuntimeError, match="fetch reached"):
            process_video(**self._kwargs(tmp_path), force=True)


class TestCliSkip:
    URLS = [URL, "https://www.youtube.com/watch?v=bbbbbbbbbbb"]

    def test_capture_youtube_reports_skipped(self, tmp_path, monkeypatch):
        import vidflow.capture as capture_pkg

        note = tmp_path / "Existing.md"
        monkeypatch.setattr(
            capture_pkg, "process_video", lambda **kw: (_ for _ in ()).throw(CaptureExists(note))
        )
        result = capture_pkg.capture_youtube(url=URL, output_dir=tmp_path)
        assert result.success is True
        assert result.data["skipped"] is True
        assert result.data["output_path"] == str(note)

    def test_cmd_youtube_excludes_skipped_from_post_processing(self, monkeypatch, capsys, tmp_path):
        from vidflow.cli import main as vidflow_main
        from vidflow.cli_common import OperationResult

        def fake_capture(**kwargs):
            if kwargs["url"] == URL:
                return OperationResult(
                    success=True,
                    message="Skipped (already captured): old.md",
                    data={"output_path": str(tmp_path / "old.md"), "skipped": True},
                )
            return OperationResult(
                success=True,
                message="ok",
                data={"output_path": str(tmp_path / "new.md"), "skipped": False},
            )

        polished = []
        monkeypatch.setattr("vidflow.capture.capture_youtube", fake_capture)
        monkeypatch.setattr(
            "vidflow.cli._polish_captures",
            lambda args, paths, errors: polished.extend(paths) or [],
        )

        result = vidflow_main(["youtube", "--polish", *self.URLS])

        assert result == 0
        assert polished == [tmp_path / "new.md"]
        err = capsys.readouterr().err
        assert "Skipped [1/2]" in err

    def test_force_flag_threads_through(self, monkeypatch, tmp_path):
        from vidflow.cli import main as vidflow_main
        from vidflow.cli_common import OperationResult

        seen = {}

        def fake_capture(**kwargs):
            seen.update(kwargs)
            return OperationResult(success=True, message="ok", data={"output_path": "x.md"})

        monkeypatch.setattr("vidflow.capture.capture_youtube", fake_capture)
        assert vidflow_main(["youtube", "-f", URL]) == 0
        assert seen["force"] is True
