import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import convert_video_to_bvh as v2m


def make_pose_points() -> dict[str, np.ndarray]:
    return {
        "left_shoulder": np.array([-10.0, 20.0, 0.0], dtype=np.float64),
        "right_shoulder": np.array([10.0, 20.0, 0.0], dtype=np.float64),
        "left_elbow": np.array([-20.0, 20.0, 0.0], dtype=np.float64),
        "right_elbow": np.array([20.0, 20.0, 0.0], dtype=np.float64),
        "left_wrist": np.array([-30.0, 20.0, 0.0], dtype=np.float64),
        "right_wrist": np.array([30.0, 20.0, 0.0], dtype=np.float64),
        "left_hip": np.array([-5.0, 0.0, 0.0], dtype=np.float64),
        "right_hip": np.array([5.0, 0.0, 0.0], dtype=np.float64),
        "left_knee": np.array([-5.0, -15.0, 0.0], dtype=np.float64),
        "right_knee": np.array([5.0, -15.0, 0.0], dtype=np.float64),
        "left_ankle": np.array([-5.0, -30.0, 0.0], dtype=np.float64),
        "right_ankle": np.array([5.0, -30.0, 0.0], dtype=np.float64),
        "nose": np.array([0.0, 40.0, 0.0], dtype=np.float64),
        "mid_hip": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "spine": np.array([0.0, 10.0, 0.0], dtype=np.float64),
        "chest": np.array([0.0, 20.0, 0.0], dtype=np.float64),
        "neck": np.array([0.0, 30.0, 0.0], dtype=np.float64),
        "head": np.array([0.0, 40.0, 0.0], dtype=np.float64),
    }


class ConvertVideoToBvhTests(unittest.TestCase):
    def test_channel_headers_size_and_layout(self) -> None:
        headers = v2m.channel_headers()
        expected_len = 6 + (len(v2m.JOINTS) - 1) * 3
        self.assertEqual(len(headers), expected_len)
        self.assertEqual(headers[:6], [
            "Hips_Xposition",
            "Hips_Yposition",
            "Hips_Zposition",
            "Hips_Zrotation",
            "Hips_Xrotation",
            "Hips_Yrotation",
        ])
        self.assertEqual(headers[-3:], ["RightAnkle_Zrotation", "RightAnkle_Xrotation", "RightAnkle_Yrotation"])

    def test_frame_channels_rest_pose_is_near_zero(self) -> None:
        pts = make_pose_points()
        rest_offsets = v2m.build_rest_offsets([pts])
        channels = v2m.frame_channels(pts, rest_offsets, pts["mid_hip"])
        self.assertEqual(len(channels), 6 + (len(v2m.JOINTS) - 1) * 3)
        for value in channels:
            self.assertAlmostEqual(value, 0.0, places=6)

    def test_writers_create_valid_outputs(self) -> None:
        pts = make_pose_points()
        frames_pts = [pts, pts]
        rest_offsets = v2m.build_rest_offsets([pts])
        motion_values = [v2m.frame_channels(pts, rest_offsets, pts["mid_hip"]), v2m.frame_channels(pts, rest_offsets, pts["mid_hip"])]
        fps = 30.0
        ref_root = pts["mid_hip"]

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            out_bvh = base / "test.bvh"
            out_json = base / "test.json"
            out_csv = base / "test.csv"
            out_npz = base / "test.npz"
            out_trc = base / "test.trc"

            v2m.write_bvh(out_bvh, fps, rest_offsets, motion_values)
            v2m.write_json(out_json, Path("think.mp4"), fps, rest_offsets, motion_values, ref_root)
            v2m.write_csv(out_csv, motion_values)
            v2m.write_npz(out_npz, Path("think.mp4"), fps, rest_offsets, motion_values, ref_root)
            v2m.write_trc(out_trc, Path("think.mp4"), fps, frames_pts, ref_root)

            self.assertTrue(out_bvh.exists())
            self.assertTrue(out_json.exists())
            self.assertTrue(out_csv.exists())
            self.assertTrue(out_npz.exists())
            self.assertTrue(out_trc.exists())

            bvh_text = out_bvh.read_text(encoding="utf-8")
            self.assertIn("HIERARCHY", bvh_text)
            self.assertIn("MOTION", bvh_text)
            self.assertIn("Frames: 2", bvh_text)

            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"]["frame_count"], 2)
            self.assertEqual(len(payload["motion"]["frames"]), 2)

            csv_lines = out_csv.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(csv_lines), 3)
            self.assertIn("Hips_Xposition", csv_lines[0])

            npz = np.load(out_npz)
            self.assertEqual(int(npz["frame_count"]), 2)
            self.assertEqual(npz["motion"].shape[0], 2)

            trc_text = out_trc.read_text(encoding="utf-8")
            self.assertIn("PathFileType", trc_text)
            self.assertIn("Frame#", trc_text)


if __name__ == "__main__":
    unittest.main()
