import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vid2model_lib import cli


class CliValidationTests(unittest.TestCase):
    def test_conflicting_output_flags_raise(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            video = Path(td) / "input.mp4"
            video.write_bytes(b"fake")
            argv = [
                "convert_video_to_bvh.py",
                "--input",
                str(video),
                "--output",
                str(Path(td) / "a.bvh"),
                "--output-bvh",
                str(Path(td) / "b.bvh"),
            ]
            with patch("sys.argv", argv):
                with self.assertRaisesRegex(ValueError, "Use either --output or --output-bvh"):
                    cli.main()

    def test_missing_output_targets_raise(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            video = Path(td) / "input.mp4"
            video.write_bytes(b"fake")
            argv = ["convert_video_to_bvh.py", "--input", str(video)]
            with patch("sys.argv", argv):
                with self.assertRaisesRegex(ValueError, "Specify at least one target"):
                    cli.main()

    def test_missing_input_file_raises(self) -> None:
        argv = [
            "convert_video_to_bvh.py",
            "--input",
            "/definitely/missing/input.mp4",
            "--output-bvh",
            "out.bvh",
        ]
        with patch("sys.argv", argv):
            with self.assertRaisesRegex(FileNotFoundError, "Input file not found"):
                cli.main()


if __name__ == "__main__":
    unittest.main()
