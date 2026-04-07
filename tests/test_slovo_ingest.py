"""Smoke tests for tools/slovo_ingest.py — no video files required."""

import csv
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "slovo_ingest", Path(__file__).parent.parent / "tools" / "slovo_ingest.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


si = _load_module()


class TestSignSlug(unittest.TestCase):
    def test_cyrillic(self):
        self.assertEqual(si._sign_to_slug("Привет!"), "привет")

    def test_latin(self):
        self.assertEqual(si._sign_to_slug("no_event"), "no_event")

    def test_single_letter(self):
        self.assertEqual(si._sign_to_slug("А"), "а")

    def test_empty_fallback(self):
        self.assertEqual(si._sign_to_slug("!!!"), "sign")


class TestHandCoverage(unittest.TestCase):
    def test_full_coverage(self):
        frames = [{"x": 1}, {"x": 2}]
        self.assertAlmostEqual(si._hand_coverage(frames), 1.0)

    def test_partial_coverage(self):
        frames = [{"x": 1}, None, None, None]
        self.assertAlmostEqual(si._hand_coverage(frames), 0.25)

    def test_empty(self):
        self.assertAlmostEqual(si._hand_coverage([]), 0.0)


class TestPassesQuality(unittest.TestCase):
    def test_no_index(self):
        self.assertTrue(si._passes_quality("any", None, 0.3))

    def test_not_in_index(self):
        self.assertTrue(si._passes_quality("missing", {"other": []}, 0.3))

    def test_passes(self):
        frames = [{"x": 1}] * 8 + [None] * 2  # 80% coverage
        self.assertTrue(si._passes_quality("v", {"v": frames}, 0.3))

    def test_fails(self):
        frames = [None] * 10  # 0% coverage
        self.assertFalse(si._passes_quality("v", {"v": frames}, 0.3))

    def test_zero_frames(self):
        self.assertFalse(si._passes_quality("v", {"v": []}, 0.3))


class TestManifest(unittest.TestCase):
    def test_record_and_ids(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "manifest.json"
            mf = si.Manifest(path)
            mf.record({"id": "aaa", "sign": "Привет!", "status": "ok"})
            mf.record({"id": "bbb", "sign": "А", "status": "failed"})

            self.assertEqual(mf.ids(), {"aaa", "bbb"})
            self.assertEqual(len(mf), 2)

            data = json.loads(path.read_text())
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0]["id"], "aaa")

    def test_resume_loads_existing(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "manifest.json"
            # Write initial manifest
            mf1 = si.Manifest(path)
            mf1.record({"id": "existing", "sign": "Б", "status": "ok"})

            # New instance should load existing entries
            mf2 = si.Manifest(path)
            self.assertIn("existing", mf2.ids())
            self.assertEqual(len(mf2), 1)

    def test_thread_safe(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "manifest.json"
            mf = si.Manifest(path)
            errors = []

            def write(i):
                try:
                    mf.record({"id": str(i), "sign": "А", "status": "ok"})
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=write, args=(i,)) for i in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errors, [])
            self.assertEqual(len(mf), 20)


class TestAnnotationFiltering(unittest.TestCase):
    """Test annotation CSV parsing and filtering using a synthetic dataset."""

    def _make_annotations(self, tmp_dir: Path) -> Path:
        slovo_dir = tmp_dir / "slovo"
        (slovo_dir / "train").mkdir(parents=True)
        (slovo_dir / "test").mkdir(parents=True)

        rows = [
            ("id-001", "Привет!", "user1", "1920", "1080", "63.0", "True"),
            ("id-002", "Привет!", "user2", "1280", "720", "45.0", "False"),
            ("id-003", "А",      "user1", "1920", "1080", "30.0", "True"),
            ("id-004", "Б",      "user1", "1920", "1080", "28.0", "True"),
            ("id-005", "no_event", "user1", "1920", "1080", "10.0", "True"),
        ]
        annotations = slovo_dir / "annotations.csv"
        with open(annotations, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["attachment_id", "text", "user_id", "height", "width", "length", "train"])
            writer.writerows(rows)

        # Create fake video files so _process_video doesn't bail early on missing file
        for vid_id, _, _, _, _, _, is_train in rows:
            split_dir = slovo_dir / "train" if is_train == "True" else slovo_dir / "test"
            (split_dir / f"{vid_id}.mp4").write_bytes(b"fake")

        return slovo_dir

    def _run_main(self, argv, fake_run_ok=True):
        """Run main() with mocked subprocess and no actual video files."""
        results = []

        def fake_run(cmd, **kw):
            r = MagicMock()
            r.returncode = 0 if fake_run_ok else 1
            results.append(cmd)
            return r

        with patch("sys.argv", argv), patch("subprocess.run", fake_run):
            try:
                si.main()
            except SystemExit:
                pass

        return results

    def test_filter_by_sign(self):
        with tempfile.TemporaryDirectory() as td:
            slovo_dir = self._make_annotations(Path(td))
            out_dir = Path(td) / "out"
            argv = [
                "slovo_ingest.py",
                "--slovo-dir", str(slovo_dir),
                "--sign", "Привет!",
                "--split", "train",
                "--count", "0",
                "--output-dir", str(out_dir),
                "--no-quality-filter",
            ]
            calls = self._run_main(argv)
            # Only id-001 is train + Привет!
            self.assertEqual(len(calls), 1)
            self.assertIn("id-001", " ".join(calls[0]))

    def test_filter_by_pattern(self):
        with tempfile.TemporaryDirectory() as td:
            slovo_dir = self._make_annotations(Path(td))
            out_dir = Path(td) / "out"
            argv = [
                "slovo_ingest.py",
                "--slovo-dir", str(slovo_dir),
                "--sign-pattern", "^[АБ]$",
                "--split", "train",
                "--count", "0",
                "--output-dir", str(out_dir),
                "--no-quality-filter",
            ]
            calls = self._run_main(argv)
            self.assertEqual(len(calls), 2)  # id-003 + id-004

    def test_count_limit(self):
        with tempfile.TemporaryDirectory() as td:
            slovo_dir = self._make_annotations(Path(td))
            out_dir = Path(td) / "out"
            argv = [
                "slovo_ingest.py",
                "--slovo-dir", str(slovo_dir),
                "--split", "train",
                "--count", "2",
                "--output-dir", str(out_dir),
                "--no-quality-filter",
            ]
            calls = self._run_main(argv)
            self.assertEqual(len(calls), 2)

    def test_resume_skips_existing(self):
        with tempfile.TemporaryDirectory() as td:
            slovo_dir = self._make_annotations(Path(td))
            out_dir = Path(td) / "out"
            out_dir.mkdir()

            # Pre-populate manifest with id-001
            mf = si.Manifest(out_dir / "manifest.json")
            mf.record({"id": "id-001", "sign": "Привет!", "status": "ok"})

            argv = [
                "slovo_ingest.py",
                "--slovo-dir", str(slovo_dir),
                "--sign", "Привет!",
                "--split", "all",
                "--count", "0",
                "--output-dir", str(out_dir),
                "--no-quality-filter",
            ]
            calls = self._run_main(argv)
            # id-001 in manifest → skipped; id-002 (test split, all) → processed
            ids_called = [c for c in calls]
            self.assertEqual(len(ids_called), 1)
            self.assertIn("id-002", " ".join(ids_called[0]))

    def test_manifest_written_on_success(self):
        with tempfile.TemporaryDirectory() as td:
            slovo_dir = self._make_annotations(Path(td))
            out_dir = Path(td) / "out"
            argv = [
                "slovo_ingest.py",
                "--slovo-dir", str(slovo_dir),
                "--sign", "А",
                "--split", "train",
                "--count", "0",
                "--output-dir", str(out_dir),
                "--no-quality-filter",
            ]
            self._run_main(argv, fake_run_ok=True)
            manifest_path = out_dir / "manifest.json"
            self.assertTrue(manifest_path.exists())
            data = json.loads(manifest_path.read_text())
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["id"], "id-003")
            self.assertEqual(data[0]["status"], "ok")


class TestPresetInCli(unittest.TestCase):
    def test_sign_language_preset_in_defaults(self):
        from vid2model_lib.cli import PRESET_DEFAULTS
        self.assertIn("sign_language", PRESET_DEFAULTS)
        p = PRESET_DEFAULTS["sign_language"]
        self.assertEqual(p["hand_tracking"], "auto")
        self.assertEqual(p["loop_mode"], "off")
        self.assertEqual(p["lower_body_rotation_mode"], "off")
        self.assertAlmostEqual(p["upper_body_rotation_scale"], 0.3)


if __name__ == "__main__":
    unittest.main()
