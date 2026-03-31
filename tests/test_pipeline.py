import unittest

import numpy as np

from vid2model_lib.pipeline import build_pose_correction_profile, fill_pose_gaps
from vid2model_lib import pipeline


def pts(v: float) -> dict[str, np.ndarray]:
    return {"mid_hip": np.array([v, 0.0, 0.0], dtype=np.float64)}


class PipelineGapFillingTests(unittest.TestCase):
    def test_mirror_detection_and_grounding(self) -> None:
        sample = {
            "left_shoulder": np.array([10.0, 20.0, 0.0], dtype=np.float64),
            "right_shoulder": np.array([-10.0, 20.0, 0.0], dtype=np.float64),
            "left_hip": np.array([5.0, 0.0, 0.0], dtype=np.float64),
            "right_hip": np.array([-5.0, 0.0, 0.0], dtype=np.float64),
            "left_wrist": np.array([18.0, 12.0, 0.0], dtype=np.float64),
            "right_wrist": np.array([-18.0, 12.0, 0.0], dtype=np.float64),
            "left_ankle": np.array([5.0, -30.0, 0.0], dtype=np.float64),
            "right_ankle": np.array([-5.0, -30.0, 0.0], dtype=np.float64),
            "left_heel": np.array([5.0, -31.0, 0.0], dtype=np.float64),
            "right_heel": np.array([-5.0, -31.0, 0.0], dtype=np.float64),
            "left_toes": np.array([5.0, -30.0, 4.0], dtype=np.float64),
            "right_toes": np.array([-5.0, -30.0, 4.0], dtype=np.float64),
            "nose": np.array([0.0, 30.0, 0.0], dtype=np.float64),
            "mid_hip": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            "spine": np.array([0.0, 10.0, 0.0], dtype=np.float64),
            "chest": np.array([0.0, 20.0, 0.0], dtype=np.float64),
            "upper_chest": np.array([0.0, 25.0, 0.0], dtype=np.float64),
            "neck": np.array([0.0, 28.0, 0.0], dtype=np.float64),
            "head": np.array([0.0, 30.0, 0.0], dtype=np.float64),
        }
        self.assertTrue(pipeline._looks_mirrored(sample))
        mirrored = pipeline._mirror_pose_points(sample)
        self.assertAlmostEqual(float(mirrored["left_shoulder"][0]), 10.0, places=6)
        self.assertAlmostEqual(float(mirrored["right_shoulder"][0]), -10.0, places=6)

        corrected = pipeline._apply_pose_corrections(mirrored, pipeline.DEFAULT_POSE_CORRECTIONS)
        ground_values = [corrected[k][1] for k in ("left_ankle", "right_ankle", "left_heel", "right_heel")]
        self.assertAlmostEqual(float(min(ground_values)), 0.0, places=6)

    def test_build_pose_correction_profile_maps_sa_style_keys(self) -> None:
        profile = build_pose_correction_profile(
            {
                "shoulder_tracking": False,
                "hip_camera": True,
                "auto_grounding": False,
                "use_arm_ik": False,
                "use_leg_ik": False,
                "upper_rotation_offset": 12.0,
                "body_collider_mode": 1,
                "body_collider_head_size_percent": 140,
            }
        )
        self.assertFalse(profile.shoulder_tracking)
        self.assertTrue(profile.hip_camera)
        self.assertFalse(profile.auto_grounding)
        self.assertFalse(profile.use_arm_ik)
        self.assertFalse(profile.use_leg_ik)
        self.assertEqual(profile.upper_rotation_offset_deg, 12.0)
        self.assertEqual(profile.body_collider_mode, 1)
        self.assertEqual(profile.body_collider_head_size_percent, 140.0)

    def test_interpolates_short_gap(self) -> None:
        frames = [pts(0.0), None, pts(2.0)]
        filled, interpolated, carried = fill_pose_gaps(frames, max_gap_interpolate=2)
        self.assertEqual(len(filled), 3)
        self.assertEqual(interpolated, 1)
        self.assertEqual(carried, 0)
        self.assertAlmostEqual(float(filled[1]["mid_hip"][0]), 1.0, places=6)

    def test_carries_long_gap(self) -> None:
        frames = [pts(0.0), None, None, pts(3.0)]
        filled, interpolated, carried = fill_pose_gaps(frames, max_gap_interpolate=1)
        self.assertEqual(len(filled), 4)
        self.assertEqual(interpolated, 0)
        self.assertEqual(carried, 2)
        self.assertAlmostEqual(float(filled[1]["mid_hip"][0]), 0.0, places=6)
        self.assertAlmostEqual(float(filled[2]["mid_hip"][0]), 0.0, places=6)

    def test_fills_leading_and_trailing_gaps(self) -> None:
        frames = [None, pts(1.0), None]
        filled, interpolated, carried = fill_pose_gaps(frames, max_gap_interpolate=2)
        self.assertEqual(len(filled), 3)
        self.assertEqual(interpolated, 0)
        self.assertEqual(carried, 2)
        self.assertAlmostEqual(float(filled[0]["mid_hip"][0]), 1.0, places=6)
        self.assertAlmostEqual(float(filled[2]["mid_hip"][0]), 1.0, places=6)

    def test_raises_when_no_detections(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "No detectable human pose frames found"):
            fill_pose_gaps([None, None], max_gap_interpolate=3)


if __name__ == "__main__":
    unittest.main()
