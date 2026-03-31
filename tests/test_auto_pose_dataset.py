import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vid2model_lib.auto_pose_dataset import append_examples_jsonl, build_auto_pose_example, example_to_record


def make_sample(offset: float = 0.0) -> dict[str, np.ndarray]:
    base = {
        "left_shoulder": np.array([-10.0, 20.0, 0.0], dtype=np.float64),
        "right_shoulder": np.array([10.0, 20.0, 0.0], dtype=np.float64),
        "left_hip": np.array([-5.0, 0.0, 0.0], dtype=np.float64),
        "right_hip": np.array([5.0, 0.0, 0.0], dtype=np.float64),
        "nose": np.array([0.0, 40.0, 0.0], dtype=np.float64),
        "mid_hip": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "spine": np.array([0.0, 10.0, 0.0], dtype=np.float64),
        "chest": np.array([0.0, 20.0, 0.0], dtype=np.float64),
        "upper_chest": np.array([0.0, 25.0, 0.0], dtype=np.float64),
        "neck": np.array([0.0, 30.0, 0.0], dtype=np.float64),
        "head": np.array([0.0, 40.0, 0.0], dtype=np.float64),
    }
    return {key: value + np.array([offset, 0.0, 0.0], dtype=np.float64) for key, value in base.items()}


class AutoPoseDatasetTests(unittest.TestCase):
    def test_build_auto_pose_example_serializes(self) -> None:
        example = build_auto_pose_example(
            samples=[make_sample(), make_sample(1.0)],
            label="mirrored",
            source="think.mp4",
            meta={"fps": 30.0},
        )
        record = example_to_record(example)
        self.assertEqual(record["schema"], "vid2model.auto_pose_example.v1")
        self.assertEqual(record["label"], "mirrored")
        self.assertEqual(record["source"], "think.mp4")
        self.assertEqual(record["sample_count"], 2)
        self.assertGreater(len(record["features"]), 0)
        self.assertIn("head_to_torso", record["summary"])

    def test_append_examples_jsonl_writes_lines(self) -> None:
        example = build_auto_pose_example(
            samples=[make_sample()],
            label="default",
            source="think.mp4",
        )
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "dataset.jsonl"
            written = append_examples_jsonl(out_path, [example, example])
            self.assertEqual(written, 2)
            lines = out_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            payload = json.loads(lines[0])
            self.assertEqual(payload["label"], "default")


if __name__ == "__main__":
    unittest.main()
