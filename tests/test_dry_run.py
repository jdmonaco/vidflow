"""Dry-run mode for the vidflow capture subcommands makes no network calls."""

import json

import pytest

from vidflow.cli import main as vidflow_main

URL_A = "https://www.youtube.com/watch?v=aaaaaaaaaaa"
URL_B = "https://www.youtube.com/watch?v=bbbbbbbbbbb"
PLAYLIST = "https://www.youtube.com/playlist?list=PLxxxxxxxxxxxxxxxx"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail the test if anything spawns yt-dlp/ffmpeg or starts a capture."""

    def boom(*args, **kwargs):  # pragma: no cover - only on regression
        raise AssertionError(f"subprocess.run called during dry run: {args[0]}")

    monkeypatch.setattr("vidflow.capture.video.subprocess.run", boom)
    monkeypatch.setattr(
        "vidflow.capture.capture_youtube",
        lambda **kw: (_ for _ in ()).throw(AssertionError("capture_youtube called")),
    )
    monkeypatch.setattr(
        "vidflow.capture.capture_local",
        lambda **kw: (_ for _ in ()).throw(AssertionError("capture_local called")),
    )


def _existing_note(tmp_path, video_id):
    note = tmp_path / "Old Talk.md"
    note.write_text(
        f"---\ntitle: Old Talk\nsource: https://www.youtube.com/watch?v={video_id}\n---\n\n# Old\n"
    )
    return note


class TestYoutubeDryRun:
    def test_lists_videos_without_network(self, tmp_path, capsys, caplog):
        rc = vidflow_main(["youtube", "--dry-run", "-o", str(tmp_path), URL_A, URL_B])
        assert rc == 0
        err = capsys.readouterr().err
        assert "no network requests" in err
        assert "aaaaaaaaaaa" in err and "bbbbbbbbbbb" in err
        assert "would capture 2 video(s)" in caplog.text

    def test_existing_note_reported_as_skip(self, tmp_path, capsys, caplog):
        note = _existing_note(tmp_path, "aaaaaaaaaaa")
        rc = vidflow_main(["youtube", "--dry-run", "-o", str(tmp_path), URL_A, URL_B])
        assert rc == 0
        assert f"already captured: {note.name}" in capsys.readouterr().err
        assert "would capture 1 video(s), skip 1 already captured" in caplog.text

    def test_force_reports_recapture(self, tmp_path, capsys):
        _existing_note(tmp_path, "aaaaaaaaaaa")
        vidflow_main(["youtube", "--dry-run", "-f", "-o", str(tmp_path), URL_A])
        assert "recapture (--force)" in capsys.readouterr().err

    def test_playlist_left_unexpanded(self, tmp_path, capsys, caplog):
        rc = vidflow_main(["youtube", "--dry-run", "-o", str(tmp_path), PLAYLIST, URL_A])
        assert rc == 0
        assert "playlist (expanded at capture time)" in capsys.readouterr().err
        assert "expand 1 playlist(s)" in caplog.text

    def test_bare_id_normalized(self, tmp_path, capsys):
        vidflow_main(["youtube", "--dry-run", "-o", str(tmp_path), "aaaaaaaaaaa"])
        assert URL_A in capsys.readouterr().err

    def test_post_processing_plan(self, tmp_path, caplog):
        vidflow_main(
            ["youtube", "--dry-run", "--polish", "-m", "foo-model", "-o", str(tmp_path), URL_A]
        )
        assert "then polish with foo-model" in caplog.text

    def test_json_output(self, tmp_path, capsys):
        _existing_note(tmp_path, "bbbbbbbbbbb")
        rc = vidflow_main(
            [
                "youtube",
                "--dry-run",
                "--transcribe",
                "--json",
                "-o",
                str(tmp_path),
                URL_A,
                URL_B,
                PLAYLIST,
            ]
        )
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["success"] is True
        assert data["data"]["dry_run"] is True
        actions = [v["action"] for v in data["data"]["videos"]]
        assert actions == ["capture", "skip", "expand"]
        assert data["data"]["videos"][1]["existing"].endswith("Old Talk.md")
        assert data["data"]["post_process"]["step"] == "transcribe"


class TestLocalDryRun:
    def test_lists_files_and_flags_missing(self, tmp_path, capsys, caplog):
        present = tmp_path / "a.mp4"
        present.write_bytes(b"x")
        missing = tmp_path / "nope.mp4"
        rc = vidflow_main(["local", "--dry-run", "-o", str(tmp_path), str(present), str(missing)])
        assert rc == 1
        err = capsys.readouterr().err
        assert "capture" in err and "missing" in err
        assert "would capture 1 file(s), 1 missing" in caplog.text

    def test_all_present_succeeds_json(self, tmp_path, capsys):
        present = tmp_path / "a.mp4"
        present.write_bytes(b"x")
        rc = vidflow_main(
            ["local", "--dry-run", "--json", "--polish", "-o", str(tmp_path), str(present)]
        )
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["success"] is True
        assert data["data"]["files"] == [{"file": str(present), "action": "capture"}]
        assert data["data"]["post_process"]["step"] == "polish"
