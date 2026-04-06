import tempfile
import unittest
from pathlib import Path

from vid2model_lib.auto_pose_dataset import discover_labeled_video_inputs


class AutoPoseDatasetDirTests(unittest.TestCase):
    def test_discover_labeled_video_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "default").mkdir()
            (root / "mirrored").mkdir()
            (root / "default" / "a.mp4").write_bytes(b"video")
            (root / "default" / "ignore.txt").write_text("x", encoding="utf-8")
            (root / "mirrored" / "b.mov").write_bytes(b"video")
            (root / "mirrored" / "nested").mkdir()
            (root / "mirrored" / "nested" / "c.mp4").write_bytes(b"video")

            items = discover_labeled_video_inputs(root, recursive=False)
            self.assertEqual([(path.name, label) for path, label in items], [("a.mp4", "default"), ("b.mov", "mirrored")])

            recursive_items = discover_labeled_video_inputs(root, recursive=True)
            self.assertEqual(
                sorted((path.name, label) for path, label in recursive_items),
                [("a.mp4", "default"), ("b.mov", "mirrored"), ("c.mp4", "mirrored")],
            )


if __name__ == "__main__":
    unittest.main()
