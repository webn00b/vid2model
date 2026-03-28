import json
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

    def test_check_tools_short_circuits_validation(self) -> None:
        argv = ["convert_video_to_bvh.py", "--check-tools"]
        with patch("sys.argv", argv):
            with patch("vid2model_lib.cli.check_tools", return_value=0) as mocked:
                rc = cli.main()
        self.assertEqual(rc, 0)
        mocked.assert_called_once()

    def test_missing_input_raises_when_not_check_tools(self) -> None:
        argv = ["convert_video_to_bvh.py", "--output-bvh", "out.bvh"]
        with patch("sys.argv", argv):
            with self.assertRaisesRegex(ValueError, "Specify --input"):
                cli.main()

    def test_json_config_used_and_cli_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            video = tmp / "input.mp4"
            video.write_bytes(b"fake")
            out_bvh = tmp / "out.bvh"
            config_path = tmp / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input": str(video),
                        "output_bvh": str(out_bvh),
                        "model_complexity": 2,
                        "progress_every": 77,
                    }
                ),
                encoding="utf-8",
            )

            argv = [
                "convert_video_to_bvh.py",
                "--config",
                str(config_path),
                "--model-complexity",
                "1",
            ]
            with patch("sys.argv", argv):
                with patch(
                    "vid2model_lib.cli.convert_video_to_bvh",
                    return_value=(30.0, {}, [[0.0] * 54], [0.0, 0.0, 0.0], []),
                ) as mocked_convert:
                    with patch("vid2model_lib.cli.write_bvh") as mocked_write_bvh:
                        rc = cli.main()

            self.assertEqual(rc, 0)
            mocked_convert.assert_called_once()
            kwargs = mocked_convert.call_args.kwargs
            self.assertEqual(kwargs["model_complexity"], 1)  # CLI override
            self.assertEqual(kwargs["progress_every"], 77)  # from config
            mocked_write_bvh.assert_called_once()

    def test_invalid_model_complexity_from_config_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            video = tmp / "input.mp4"
            video.write_bytes(b"fake")
            config_path = tmp / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input": str(video),
                        "output_bvh": str(tmp / "out.bvh"),
                        "model_complexity": 9,
                    }
                ),
                encoding="utf-8",
            )

            argv = ["convert_video_to_bvh.py", "--config", str(config_path)]
            with patch("sys.argv", argv):
                with self.assertRaisesRegex(ValueError, "model_complexity must be one of"):
                    cli.main()

    def test_negative_progress_every_from_config_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            video = tmp / "input.mp4"
            video.write_bytes(b"fake")
            config_path = tmp / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input": str(video),
                        "output_bvh": str(tmp / "out.bvh"),
                        "progress_every": -1,
                    }
                ),
                encoding="utf-8",
            )

            argv = ["convert_video_to_bvh.py", "--config", str(config_path)]
            with patch("sys.argv", argv):
                with self.assertRaisesRegex(ValueError, "progress_every must be >= 0"):
                    cli.main()


if __name__ == "__main__":
    unittest.main()
