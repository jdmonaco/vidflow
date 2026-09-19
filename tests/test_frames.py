"""Tests for ffmpeg version detection and VFR flag selection."""

from unittest.mock import patch

from vidflow.capture import frames


class TestFfmpegVersion:
    """Tests for ffmpeg_version() parsing."""

    def _version_for(self, stdout: str) -> tuple[int, int]:
        frames.ffmpeg_version.cache_clear()
        with patch("vidflow.capture.frames.subprocess.run") as mock_run:
            mock_run.return_value.stdout = stdout
            result = frames.ffmpeg_version()
        frames.ffmpeg_version.cache_clear()
        return result

    def test_homebrew_style(self):
        assert self._version_for("ffmpeg version 9.0.1 Copyright ...") == (9, 0)

    def test_tagged_style(self):
        assert self._version_for("ffmpeg version n5.1.2 Copyright ...") == (5, 1)

    def test_ubuntu_style(self):
        assert self._version_for("ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright ...") == (4, 4)

    def test_unparseable(self):
        assert self._version_for("garbage") == (0, 0)


class TestVfrOutputArgs:
    """Tests for the -fps_mode / -vsync selection."""

    def _args_for(self, version: tuple[int, int]) -> list[str]:
        with patch("vidflow.capture.frames.ffmpeg_version", return_value=version):
            return frames.vfr_output_args()

    def test_modern_ffmpeg_uses_fps_mode(self):
        assert self._args_for((9, 0)) == ["-fps_mode", "vfr"]

    def test_boundary_5_1_uses_fps_mode(self):
        assert self._args_for((5, 1)) == ["-fps_mode", "vfr"]

    def test_old_ffmpeg_uses_vsync(self):
        assert self._args_for((4, 4)) == ["-vsync", "vfr"]
        assert self._args_for((5, 0)) == ["-vsync", "vfr"]

    def test_unknown_version_defaults_to_fps_mode(self):
        assert self._args_for((0, 0)) == ["-fps_mode", "vfr"]


class TestDedupThreshold:
    """Semantics of hash_similarity and the dedup threshold."""

    def _hashes(self, distance: int):
        import numpy as np
        import imagehash

        a = np.zeros(64, dtype=bool)
        a[:32] = True
        b = a.copy()
        # Flip distance/2 set bits off and distance/2 clear bits on
        half = distance // 2
        b[:half] = False
        b[32 : 32 + half] = True
        return imagehash.ImageHash(a.reshape(8, 8)), imagehash.ImageHash(b.reshape(8, 8))

    def test_identical_hashes_are_fully_similar(self):
        h1, h2 = self._hashes(0)
        assert frames.hash_similarity(h1, h2) == 1.0

    def test_each_differing_bit_costs_one_64th(self):
        h1, h2 = self._hashes(16)
        assert frames.hash_similarity(h1, h2) == 0.75

    def test_lower_threshold_drops_more_frames(self):
        """A frame is a duplicate when similarity >= threshold."""
        h1, h2 = self._hashes(10)  # similarity 0.84375
        sim = frames.hash_similarity(h1, h2)
        assert sim >= 0.80  # dropped at the default
        assert sim < 0.95  # kept at the old default

    def test_default_threshold_is_single_sourced(self):
        import inspect

        from vidflow.capture import capture_local, capture_youtube
        from vidflow.capture.config import DEFAULT_CONFIG, DEFAULT_DEDUP_THRESHOLD

        assert DEFAULT_CONFIG["dedup_threshold"] == DEFAULT_DEDUP_THRESHOLD
        for fn in (
            frames.extract_frames_fast,
            frames.extract_frames_from_file,
            capture_youtube,
            capture_local,
        ):
            assert inspect.signature(fn).parameters["dedup_threshold"].default == (
                DEFAULT_DEDUP_THRESHOLD
            ), fn.__name__


class TestFrameCollector:
    """Near-duplicate runs collapse to the first timestamp with the last image."""

    @staticmethod
    def _hash(distance_from_base: int):
        import numpy as np
        import imagehash

        bits = np.zeros(64, dtype=bool)
        bits[:32] = True
        half = distance_from_base // 2
        bits[:half] = False
        bits[32 : 32 + half] = True
        return imagehash.ImageHash(bits.reshape(8, 8))

    def _run(self, tmp_path, threshold, sequence):
        """sequence: list of (name, distance_from_first_frame_of_video)."""
        from unittest.mock import patch

        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        out.mkdir()
        hashes = {}
        for name, dist in sequence:
            (src / f"{name}.jpg").write_bytes(name.encode())
            hashes[name] = self._hash(dist)
        with patch(
            "vidflow.capture.frames.compute_phash",
            side_effect=lambda p: hashes[p.stem],
        ):
            collector = frames.FrameCollector(out, "jpg", threshold)
            for i, (name, _) in enumerate(sequence):
                collector.add(src / f"{name}.jpg", float(i * 15))
        return collector.frames, out

    def test_run_keeps_first_timestamp_and_last_image(self, tmp_path):
        kept, out = self._run(
            tmp_path, 0.80, [("a", 0), ("a2", 6), ("a3", 10), ("b", 30), ("b2", 36)]
        )
        assert [(f.path.name, f.timestamp) for f in kept] == [
            ("frame-0000.jpg", 0.0),
            ("frame-0001.jpg", 45.0),
        ]
        assert kept[0].path.read_bytes() == b"a3"
        assert kept[1].path.read_bytes() == b"b2"
        assert sorted(p.name for p in out.iterdir()) == ["frame-0000.jpg", "frame-0001.jpg"]

    def test_similarity_is_measured_from_run_start(self, tmp_path):
        """Drift accumulates against the anchor, so a run ends once it exceeds it."""
        kept, _ = self._run(tmp_path, 0.80, [("a", 0), ("a2", 8), ("a3", 14)])
        assert [f.timestamp for f in kept] == [0.0, 30.0]
        assert kept[0].path.read_bytes() == b"a2"

    def test_threshold_none_keeps_everything(self, tmp_path):
        kept, _ = self._run(tmp_path, None, [("a", 0), ("a2", 0), ("a3", 0)])
        assert [f.timestamp for f in kept] == [0.0, 15.0, 30.0]

    def test_source_files_are_consumed(self, tmp_path):
        _, _ = self._run(tmp_path, 0.80, [("a", 0), ("a2", 4)])
        assert not any((tmp_path / "src").iterdir())
