import tempfile
import unittest
from pathlib import Path

import numpy as np

from vid2model_lib.auto_pose_dataset import append_examples_jsonl, build_auto_pose_example
from vid2model_lib.auto_pose_trainer import (
    load_auto_pose_jsonl,
    save_auto_pose_model,
    summarize_auto_pose_model,
    train_auto_pose_model,
)
from vid2model_lib.pipeline_auto_pose import _predict_auto_label


def make_sample(scale: float = 1.0, shift_x: float = 0.0) -> dict[str, np.ndarray]:
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
        "left_wrist": np.array([-30.0, 20.0, 0.0], dtype=np.float64),
        "right_wrist": np.array([30.0, 20.0, 0.0], dtype=np.float64),
        "left_ankle": np.array([-5.0, -30.0, 0.0], dtype=np.float64),
        "right_ankle": np.array([5.0, -30.0, 0.0], dtype=np.float64),
        "left_knee": np.array([-5.0, -15.0, 0.0], dtype=np.float64),
        "right_knee": np.array([5.0, -15.0, 0.0], dtype=np.float64),
    }
    out = {}
    for key, value in base.items():
        scaled = value * scale
        scaled[0] += shift_x
        out[key] = scaled
    return out


def mirror_sample(sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mirrored: dict[str, np.ndarray] = {}
    for key, value in sample.items():
        if key.startswith("left_"):
            new_key = "right_" + key[5:]
        elif key.startswith("right_"):
            new_key = "left_" + key[6:]
        else:
            new_key = key
        mirrored[new_key] = np.array([-value[0], value[1], value[2]], dtype=np.float64)
    return mirrored


class AutoPoseTrainerTests(unittest.TestCase):
    def test_train_and_predict_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            dataset_path = base / "dataset.jsonl"
            model_path = base / "model.npz"

            default_example = build_auto_pose_example([make_sample()], "default", "a.mp4")
            mirrored_example = build_auto_pose_example([mirror_sample(make_sample())], "mirrored", "b.mp4")
            append_examples_jsonl(dataset_path, [default_example, mirrored_example])

            data = load_auto_pose_jsonl(dataset_path)
            model = train_auto_pose_model(data, model_type="mlp", hidden_size=8, epochs=1200, learning_rate=0.05)
            save_auto_pose_model(model_path, model)
            summary = summarize_auto_pose_model(model)
            self.assertEqual(summary["model_type"], "mlp")
            self.assertEqual(summary["classes"], ["default", "mirrored"])
            self.assertEqual(summary["train_count"], 2)
            self.assertIn("W1", model)
            self.assertIn("W2", model)
            self.assertTrue(model_path.exists())

            label, meta = _predict_auto_label(
                np.asarray(default_example.features, dtype=np.float64),
                default_example.summary,
                str(model_path),
            )
            self.assertEqual(label, "default")
            self.assertIn("model_score", meta)

    def test_single_class_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dataset_path = Path(td) / "dataset.jsonl"
            append_examples_jsonl(dataset_path, [build_auto_pose_example([make_sample()], "default", "a.mp4")])
            data = load_auto_pose_jsonl(dataset_path)
            with self.assertRaises(ValueError):
                train_auto_pose_model(data, model_type="mlp")


if __name__ == "__main__":
    unittest.main()
