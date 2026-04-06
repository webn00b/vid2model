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
        "left_shoulder_clavicle": np.array([-3.5, 20.0, 0.0], dtype=np.float64),
        "right_shoulder_clavicle": np.array([3.5, 20.0, 0.0], dtype=np.float64),
        "left_elbow": np.array([-20.0, 20.0, 0.0], dtype=np.float64),
        "right_elbow": np.array([20.0, 20.0, 0.0], dtype=np.float64),
        "left_wrist": np.array([-30.0, 20.0, 0.0], dtype=np.float64),
        "right_wrist": np.array([30.0, 20.0, 0.0], dtype=np.float64),
        "left_hand": np.array([-32.0, 20.0, 1.0], dtype=np.float64),
        "left_middle_proximal": np.array([-35.0, 20.0, 2.0], dtype=np.float64),
        "left_middle_intermediate": np.array([-38.0, 20.0, 3.0], dtype=np.float64),
        "left_middle_distal": np.array([-41.0, 20.0, 4.0], dtype=np.float64),
        "left_index_proximal": np.array([-35.0, 21.0, 2.0], dtype=np.float64),
        "left_index_intermediate": np.array([-38.0, 22.0, 3.0], dtype=np.float64),
        "left_index_distal": np.array([-41.0, 23.0, 4.0], dtype=np.float64),
        "left_ring_proximal": np.array([-35.0, 19.0, 2.0], dtype=np.float64),
        "left_ring_intermediate": np.array([-38.0, 18.0, 3.0], dtype=np.float64),
        "left_ring_distal": np.array([-41.0, 17.0, 4.0], dtype=np.float64),
        "left_little_proximal": np.array([-34.0, 18.0, 1.5], dtype=np.float64),
        "left_little_intermediate": np.array([-36.0, 17.0, 2.5], dtype=np.float64),
        "left_little_distal": np.array([-38.0, 16.0, 3.5], dtype=np.float64),
        "left_thumb_metacarpal": np.array([-33.5, 18.5, 1.0], dtype=np.float64),
        "left_thumb_proximal": np.array([-35.0, 17.5, 2.0], dtype=np.float64),
        "left_thumb_distal": np.array([-36.5, 16.5, 3.0], dtype=np.float64),
        "right_hand": np.array([32.0, 20.0, 1.0], dtype=np.float64),
        "right_middle_proximal": np.array([35.0, 20.0, 2.0], dtype=np.float64),
        "right_middle_intermediate": np.array([38.0, 20.0, 3.0], dtype=np.float64),
        "right_middle_distal": np.array([41.0, 20.0, 4.0], dtype=np.float64),
        "right_index_proximal": np.array([35.0, 21.0, 2.0], dtype=np.float64),
        "right_index_intermediate": np.array([38.0, 22.0, 3.0], dtype=np.float64),
        "right_index_distal": np.array([41.0, 23.0, 4.0], dtype=np.float64),
        "right_ring_proximal": np.array([35.0, 19.0, 2.0], dtype=np.float64),
        "right_ring_intermediate": np.array([38.0, 18.0, 3.0], dtype=np.float64),
        "right_ring_distal": np.array([41.0, 17.0, 4.0], dtype=np.float64),
        "right_little_proximal": np.array([34.0, 18.0, 1.5], dtype=np.float64),
        "right_little_intermediate": np.array([36.0, 17.0, 2.5], dtype=np.float64),
        "right_little_distal": np.array([38.0, 16.0, 3.5], dtype=np.float64),
        "right_thumb_metacarpal": np.array([33.5, 18.5, 1.0], dtype=np.float64),
        "right_thumb_proximal": np.array([35.0, 17.5, 2.0], dtype=np.float64),
        "right_thumb_distal": np.array([36.5, 16.5, 3.0], dtype=np.float64),
        "left_hip": np.array([-5.0, 0.0, 0.0], dtype=np.float64),
        "right_hip": np.array([5.0, 0.0, 0.0], dtype=np.float64),
        "left_knee": np.array([-5.0, -15.0, 0.0], dtype=np.float64),
        "right_knee": np.array([5.0, -15.0, 0.0], dtype=np.float64),
        "left_ankle": np.array([-5.0, -30.0, 0.0], dtype=np.float64),
        "right_ankle": np.array([5.0, -30.0, 0.0], dtype=np.float64),
        "left_toes": np.array([-5.0, -30.0, 8.0], dtype=np.float64),
        "right_toes": np.array([5.0, -30.0, 8.0], dtype=np.float64),
        "left_heel": np.array([-5.0, -30.0, -4.0], dtype=np.float64),
        "right_heel": np.array([5.0, -30.0, -4.0], dtype=np.float64),
        "nose": np.array([0.0, 40.0, 0.0], dtype=np.float64),
        "mid_hip": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "spine": np.array([0.0, 10.0, 0.0], dtype=np.float64),
        "chest": np.array([0.0, 20.0, 0.0], dtype=np.float64),
        "upper_chest": np.array([0.0, 25.0, 0.0], dtype=np.float64),
        "neck": np.array([0.0, 30.0, 0.0], dtype=np.float64),
        "head": np.array([0.0, 40.0, 0.0], dtype=np.float64),
    }


class ConvertVideoToBvhTests(unittest.TestCase):
    def test_extract_pose_points_supports_legacy_mediapipe_result_shape(self) -> None:
        class _LM:
            def __init__(self, x: float, y: float, z: float) -> None:
                self.x = x
                self.y = y
                self.z = z

        class _LandmarkList:
            def __init__(self, landmarks) -> None:
                self.landmark = landmarks

        class _Res:
            pass

        world = [_LM(0.0, 0.0, 0.0) for _ in range(33)]
        world[v2m.LM["nose"]] = _LM(0.0, -0.4, 0.0)
        world[v2m.LM["left_shoulder"]] = _LM(-0.2, -0.2, 0.0)
        world[v2m.LM["right_shoulder"]] = _LM(0.2, -0.2, 0.0)
        world[v2m.LM["left_elbow"]] = _LM(-0.35, -0.2, 0.0)
        world[v2m.LM["right_elbow"]] = _LM(0.35, -0.2, 0.0)
        world[v2m.LM["left_wrist"]] = _LM(-0.45, -0.2, 0.0)
        world[v2m.LM["right_wrist"]] = _LM(0.45, -0.2, 0.0)
        world[v2m.LM["left_pinky"]] = _LM(-0.5, -0.2, 0.02)
        world[v2m.LM["right_pinky"]] = _LM(0.5, -0.2, 0.02)
        world[v2m.LM["left_index"]] = _LM(-0.5, -0.18, 0.02)
        world[v2m.LM["right_index"]] = _LM(0.5, -0.18, 0.02)
        world[v2m.LM["left_thumb"]] = _LM(-0.48, -0.22, -0.01)
        world[v2m.LM["right_thumb"]] = _LM(0.48, -0.22, -0.01)
        world[v2m.LM["left_hip"]] = _LM(-0.1, 0.0, 0.0)
        world[v2m.LM["right_hip"]] = _LM(0.1, 0.0, 0.0)
        world[v2m.LM["left_knee"]] = _LM(-0.1, 0.2, 0.0)
        world[v2m.LM["right_knee"]] = _LM(0.1, 0.2, 0.0)
        world[v2m.LM["left_ankle"]] = _LM(-0.1, 0.4, 0.0)
        world[v2m.LM["right_ankle"]] = _LM(0.1, 0.4, 0.0)
        world[v2m.LM["left_heel"]] = _LM(-0.1, 0.4, -0.04)
        world[v2m.LM["right_heel"]] = _LM(0.1, 0.4, -0.04)
        world[v2m.LM["left_foot_index"]] = _LM(-0.1, 0.4, 0.08)
        world[v2m.LM["right_foot_index"]] = _LM(0.1, 0.4, 0.08)

        res = _Res()
        res.pose_world_landmarks = _LandmarkList(world)
        res.pose_landmarks = None

        extracted = v2m.extract_pose_points(res)
        self.assertIsNotNone(extracted)
        assert extracted is not None
        self.assertIn("left_toes", extracted)
        self.assertIn("right_toes", extracted)
        self.assertTrue(np.isfinite(extracted["left_toes"]).all())
        self.assertTrue(np.isfinite(extracted["right_toes"]).all())

    def test_extract_pose_points_contains_all_mapped_points(self) -> None:
        class _LM:
            def __init__(self, x: float, y: float, z: float) -> None:
                self.x = x
                self.y = y
                self.z = z

        class _Res:
            pass

        points = [_LM(0.0, 0.0, 0.0) for _ in range(33)]
        points[v2m.LM["nose"]] = _LM(0.0, 0.65, 0.0)
        points[v2m.LM["left_shoulder"]] = _LM(-0.20, 0.40, 0.0)
        points[v2m.LM["right_shoulder"]] = _LM(0.20, 0.40, 0.0)
        points[v2m.LM["left_elbow"]] = _LM(-0.32, 0.34, 0.0)
        points[v2m.LM["right_elbow"]] = _LM(0.32, 0.34, 0.0)
        points[v2m.LM["left_wrist"]] = _LM(-0.42, 0.30, 0.0)
        points[v2m.LM["right_wrist"]] = _LM(0.42, 0.30, 0.0)
        points[v2m.LM["left_pinky"]] = _LM(-0.47, 0.28, 0.02)
        points[v2m.LM["right_pinky"]] = _LM(0.47, 0.28, 0.02)
        points[v2m.LM["left_index"]] = _LM(-0.47, 0.33, 0.02)
        points[v2m.LM["right_index"]] = _LM(0.47, 0.33, 0.02)
        points[v2m.LM["left_thumb"]] = _LM(-0.45, 0.31, -0.01)
        points[v2m.LM["right_thumb"]] = _LM(0.45, 0.31, -0.01)
        points[v2m.LM["left_hip"]] = _LM(-0.10, 0.15, 0.0)
        points[v2m.LM["right_hip"]] = _LM(0.10, 0.15, 0.0)
        points[v2m.LM["left_knee"]] = _LM(-0.11, -0.05, 0.0)
        points[v2m.LM["right_knee"]] = _LM(0.11, -0.05, 0.0)
        points[v2m.LM["left_ankle"]] = _LM(-0.12, -0.25, 0.0)
        points[v2m.LM["right_ankle"]] = _LM(0.12, -0.25, 0.0)
        points[v2m.LM["left_heel"]] = _LM(-0.12, -0.25, -0.06)
        points[v2m.LM["right_heel"]] = _LM(0.12, -0.25, -0.06)
        points[v2m.LM["left_foot_index"]] = _LM(-0.12, -0.25, 0.09)
        points[v2m.LM["right_foot_index"]] = _LM(0.12, -0.25, 0.09)

        res = _Res()
        res.pose_world_landmarks = [points]
        res.pose_landmarks = []

        extracted = v2m.extract_pose_points(res)
        self.assertIsNotNone(extracted)
        assert extracted is not None

        required_point_keys = {names[0] for names in v2m.MAP_TO_POINTS.values()}
        missing = sorted(required_point_keys.difference(extracted.keys()))
        self.assertEqual(missing, [])
        for key in required_point_keys:
            self.assertTrue(np.isfinite(extracted[key]).all(), msg=f"non-finite point: {key}")

    def test_channel_headers_size_and_layout(self) -> None:
        headers = v2m.channel_headers()
        root_name = v2m.JOINTS[0].name
        expected_len = 6 + (len(v2m.JOINTS) - 1) * 3
        self.assertEqual(len(headers), expected_len)
        self.assertEqual(headers[:6], [
            f"{root_name}_Xposition",
            f"{root_name}_Yposition",
            f"{root_name}_Zposition",
            f"{root_name}_Zrotation",
            f"{root_name}_Xrotation",
            f"{root_name}_Yrotation",
        ])
        last_joint = v2m.JOINTS[-1].name
        self.assertEqual(headers[-3:], [f"{last_joint}_Zrotation", f"{last_joint}_Xrotation", f"{last_joint}_Yrotation"])

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
            out_diag = base / "test.diag.json"

            v2m.write_bvh(out_bvh, fps, rest_offsets, motion_values)
            v2m.write_json(out_json, Path("think.mp4"), fps, rest_offsets, motion_values, ref_root)
            v2m.write_csv(out_csv, motion_values)
            v2m.write_npz(out_npz, Path("think.mp4"), fps, rest_offsets, motion_values, ref_root)
            v2m.write_trc(out_trc, Path("think.mp4"), fps, frames_pts, ref_root)
            v2m.write_diagnostic_json(out_diag, {"cleanup": {"smooth_alpha": 0.35}})

            self.assertTrue(out_bvh.exists())
            self.assertTrue(out_json.exists())
            self.assertTrue(out_csv.exists())
            self.assertTrue(out_npz.exists())
            self.assertTrue(out_trc.exists())
            self.assertTrue(out_diag.exists())

            bvh_text = out_bvh.read_text(encoding="utf-8")
            self.assertIn("HIERARCHY", bvh_text)
            self.assertIn("MOTION", bvh_text)
            self.assertIn("Frames: 2", bvh_text)

            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"]["frame_count"], 2)
            self.assertEqual(len(payload["motion"]["frames"]), 2)

            csv_lines = out_csv.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(csv_lines), 3)
            self.assertIn(f"{v2m.JOINTS[0].name}_Xposition", csv_lines[0])

            npz = np.load(out_npz)
            self.assertEqual(int(npz["frame_count"]), 2)
            self.assertEqual(npz["motion"].shape[0], 2)

            trc_text = out_trc.read_text(encoding="utf-8")
            self.assertIn("PathFileType", trc_text)
            self.assertIn("Frame#", trc_text)

            diag_payload = json.loads(out_diag.read_text(encoding="utf-8"))
            self.assertEqual(diag_payload["cleanup"]["smooth_alpha"], 0.35)


if __name__ == "__main__":
    unittest.main()
