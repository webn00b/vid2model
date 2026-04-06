import unittest

import numpy as np

from vid2model_lib.pipeline_auto_pose import DEFAULT_POSE_CORRECTIONS
from vid2model_lib.pipeline_cleanup import _copy_pose_frame, _smooth_pose_frames
from vid2model_lib.pipeline_retarget import apply_pose_corrections
from vid2model_lib.pipeline_video_scan import _should_fallback_to_legacy_pose
from vid2model_lib.pipeline import build_pose_correction_profile, fill_pose_gaps
from vid2model_lib import pipeline
from vid2model_lib.pose_points import extract_pose_points, LM


def pts(v: float) -> dict[str, np.ndarray]:
    return {"mid_hip": np.array([v, 0.0, 0.0], dtype=np.float64)}


class PipelineGapFillingTests(unittest.TestCase):
    def test_should_fallback_to_legacy_pose_for_opengl_backend_error(self) -> None:
        exc = RuntimeError(
            'Service "kGpuService" ... Could not create an NSOpenGLPixelFormat'
        )
        self.assertTrue(_should_fallback_to_legacy_pose(exc))
        self.assertFalse(_should_fallback_to_legacy_pose(RuntimeError("plain failure")))

    def test_extract_pose_points_builds_compact_neck_and_head_from_face_cluster(self) -> None:
        class Landmark:
            def __init__(self, x: float, y: float, z: float) -> None:
                self.x = x
                self.y = y
                self.z = z

        coords = [Landmark(0.5, 0.5, 0.0) for _ in range(33)]
        coords[LM["left_shoulder"]] = Landmark(0.40, 0.45, 0.0)
        coords[LM["right_shoulder"]] = Landmark(0.60, 0.45, 0.0)
        coords[LM["left_hip"]] = Landmark(0.44, 0.62, 0.0)
        coords[LM["right_hip"]] = Landmark(0.56, 0.62, 0.0)
        coords[LM["nose"]] = Landmark(0.50, 0.30, 0.0)
        coords[LM["left_eye"]] = Landmark(0.47, 0.31, 0.0)
        coords[LM["right_eye"]] = Landmark(0.53, 0.31, 0.0)
        coords[LM["left_ear"]] = Landmark(0.44, 0.33, 0.0)
        coords[LM["right_ear"]] = Landmark(0.56, 0.33, 0.0)
        coords[LM["left_elbow"]] = Landmark(0.34, 0.52, 0.0)
        coords[LM["right_elbow"]] = Landmark(0.66, 0.52, 0.0)
        coords[LM["left_wrist"]] = Landmark(0.28, 0.60, 0.0)
        coords[LM["right_wrist"]] = Landmark(0.72, 0.60, 0.0)
        coords[LM["left_pinky"]] = Landmark(0.26, 0.62, 0.0)
        coords[LM["right_pinky"]] = Landmark(0.74, 0.62, 0.0)
        coords[LM["left_index"]] = Landmark(0.27, 0.60, 0.0)
        coords[LM["right_index"]] = Landmark(0.73, 0.60, 0.0)
        coords[LM["left_thumb"]] = Landmark(0.27, 0.58, 0.0)
        coords[LM["right_thumb"]] = Landmark(0.73, 0.58, 0.0)
        coords[LM["left_knee"]] = Landmark(0.44, 0.78, 0.0)
        coords[LM["right_knee"]] = Landmark(0.56, 0.78, 0.0)
        coords[LM["left_ankle"]] = Landmark(0.44, 0.93, 0.0)
        coords[LM["right_ankle"]] = Landmark(0.56, 0.93, 0.0)
        coords[LM["left_heel"]] = Landmark(0.44, 0.94, -0.02)
        coords[LM["right_heel"]] = Landmark(0.56, 0.94, -0.02)
        coords[LM["left_foot_index"]] = Landmark(0.44, 0.93, 0.04)
        coords[LM["right_foot_index"]] = Landmark(0.56, 0.93, 0.04)

        class LandmarkList:
            def __init__(self, landmark):
                self.landmark = landmark

        class Result:
            pose_world_landmarks = None
            pose_landmarks = LandmarkList(coords)

        pts = extract_pose_points(Result())
        self.assertIsNotNone(pts)
        neck_len = float(np.linalg.norm(pts["neck"] - pts["upper_chest"]))
        head_len = float(np.linalg.norm(pts["head"] - pts["neck"]))
        self.assertLess(neck_len, head_len)
        self.assertLess(neck_len, 6.0)

    def test_extract_pose_points_uses_torso_and_face_to_push_neck_forward(self) -> None:
        class Landmark:
            def __init__(self, x: float, y: float, z: float) -> None:
                self.x = x
                self.y = y
                self.z = z

        coords = [Landmark(0.5, 0.5, 0.0) for _ in range(33)]
        coords[LM["left_shoulder"]] = Landmark(0.40, 0.45, 0.0)
        coords[LM["right_shoulder"]] = Landmark(0.60, 0.45, 0.0)
        coords[LM["left_hip"]] = Landmark(0.44, 0.62, 0.0)
        coords[LM["right_hip"]] = Landmark(0.56, 0.62, 0.0)
        coords[LM["nose"]] = Landmark(0.50, 0.30, -0.10)
        coords[LM["left_eye"]] = Landmark(0.47, 0.31, -0.08)
        coords[LM["right_eye"]] = Landmark(0.53, 0.31, -0.08)
        coords[LM["left_ear"]] = Landmark(0.44, 0.33, -0.04)
        coords[LM["right_ear"]] = Landmark(0.56, 0.33, -0.04)
        coords[LM["left_elbow"]] = Landmark(0.34, 0.52, 0.0)
        coords[LM["right_elbow"]] = Landmark(0.66, 0.52, 0.0)
        coords[LM["left_wrist"]] = Landmark(0.28, 0.60, 0.0)
        coords[LM["right_wrist"]] = Landmark(0.72, 0.60, 0.0)
        coords[LM["left_pinky"]] = Landmark(0.26, 0.62, 0.0)
        coords[LM["right_pinky"]] = Landmark(0.74, 0.62, 0.0)
        coords[LM["left_index"]] = Landmark(0.27, 0.60, 0.0)
        coords[LM["right_index"]] = Landmark(0.73, 0.60, 0.0)
        coords[LM["left_thumb"]] = Landmark(0.27, 0.58, 0.0)
        coords[LM["right_thumb"]] = Landmark(0.73, 0.58, 0.0)
        coords[LM["left_knee"]] = Landmark(0.44, 0.78, 0.0)
        coords[LM["right_knee"]] = Landmark(0.56, 0.78, 0.0)
        coords[LM["left_ankle"]] = Landmark(0.44, 0.93, 0.0)
        coords[LM["right_ankle"]] = Landmark(0.56, 0.93, 0.0)
        coords[LM["left_heel"]] = Landmark(0.44, 0.94, -0.02)
        coords[LM["right_heel"]] = Landmark(0.56, 0.94, -0.02)
        coords[LM["left_foot_index"]] = Landmark(0.44, 0.93, 0.04)
        coords[LM["right_foot_index"]] = Landmark(0.56, 0.93, 0.04)

        class LandmarkList:
            def __init__(self, landmark):
                self.landmark = landmark

        class Result:
            pose_world_landmarks = None
            pose_landmarks = LandmarkList(coords)

        pose = extract_pose_points(Result())
        self.assertIsNotNone(pose)
        upper_to_neck = pose["neck"] - pose["upper_chest"]
        neck_to_head = pose["head"] - pose["neck"]
        self.assertGreater(float(upper_to_neck[1]), 0.0)
        self.assertGreater(float(upper_to_neck[2]), 0.0)
        self.assertGreater(float(neck_to_head[2]), 0.0)

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

        corrected = apply_pose_corrections(mirrored, DEFAULT_POSE_CORRECTIONS)
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

    def test_cleanup_pose_frames_constrains_segment_lengths(self) -> None:
        base = {
            "mid_hip": np.array([0.0, 0.0, 0.0], dtype=np.float64),
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
            "left_shoulder": np.array([-8.0, 18.0, 0.0], dtype=np.float64),
            "right_shoulder": np.array([8.0, 18.0, 0.0], dtype=np.float64),
            "spine": np.array([0.0, 8.0, 0.0], dtype=np.float64),
            "chest": np.array([0.0, 18.0, 0.0], dtype=np.float64),
            "upper_chest": np.array([0.0, 22.0, 0.0], dtype=np.float64),
            "neck": np.array([0.0, 25.0, 0.0], dtype=np.float64),
            "head": np.array([0.0, 30.0, 0.0], dtype=np.float64),
        }
        anchor_samples = [_copy_pose_frame(base) for _ in range(3)]

        distorted = []
        for ankle_y in (-24.0, -38.0, -27.0):
            frame = _copy_pose_frame(base)
            frame["left_ankle"] = np.array([-5.0, ankle_y, 0.0], dtype=np.float64)
            distorted.append(frame)

        cleaned, stats = pipeline.cleanup_pose_frames(distorted, anchor_samples)
        self.assertGreater(stats["length_constraints"], 0.0)
        target_len = float(np.linalg.norm(base["left_ankle"] - base["left_knee"]))
        cleaned_lengths = [
            float(np.linalg.norm(frame["left_ankle"] - frame["left_knee"]))
            for frame in cleaned
        ]
        for length in cleaned_lengths:
            self.assertAlmostEqual(length, target_len, places=4)

    def test_cleanup_pose_frames_stabilizes_foot_contact_window(self) -> None:
        frames = []
        for shift in (0.0, 1.5, -1.0, 0.5):
            frame = {
                "mid_hip": np.array([0.0, 0.0, 0.0], dtype=np.float64),
                "left_hip": np.array([-5.0, 0.0, 0.0], dtype=np.float64),
                "right_hip": np.array([5.0, 0.0, 0.0], dtype=np.float64),
                "left_knee": np.array([-5.0, -15.0, 0.0], dtype=np.float64),
                "right_knee": np.array([5.0, -15.0, 0.0], dtype=np.float64),
                "left_ankle": np.array([-5.0 + shift, -30.0, 0.0 + shift], dtype=np.float64),
                "right_ankle": np.array([5.0, -30.0, 0.0], dtype=np.float64),
                "left_toes": np.array([-5.0 + shift, -30.0, 8.0 + shift], dtype=np.float64),
                "right_toes": np.array([5.0, -30.0, 8.0], dtype=np.float64),
                "left_heel": np.array([-5.0 + shift, -30.0, -4.0 + shift], dtype=np.float64),
                "right_heel": np.array([5.0, -30.0, -4.0], dtype=np.float64),
                "left_shoulder": np.array([-8.0, 18.0, 0.0], dtype=np.float64),
                "right_shoulder": np.array([8.0, 18.0, 0.0], dtype=np.float64),
                "spine": np.array([0.0, 8.0, 0.0], dtype=np.float64),
                "chest": np.array([0.0, 18.0, 0.0], dtype=np.float64),
                "upper_chest": np.array([0.0, 22.0, 0.0], dtype=np.float64),
                "neck": np.array([0.0, 25.0, 0.0], dtype=np.float64),
                "head": np.array([0.0, 30.0, 0.0], dtype=np.float64),
            }
            frames.append(frame)

        anchors = [_copy_pose_frame(frames[0]) for _ in range(3)]
        before_ankle_x = [float(frame["left_ankle"][0]) for frame in frames]
        before_ankle_z = [float(frame["left_ankle"][2]) for frame in frames]
        cleaned, stats = pipeline.cleanup_pose_frames(frames, anchors)
        self.assertGreaterEqual(stats["contact_windows"], 1.0)

        ankle_x = [float(frame["left_ankle"][0]) for frame in cleaned]
        ankle_z = [float(frame["left_ankle"][2]) for frame in cleaned]
        self.assertLess(max(ankle_x) - min(ankle_x), (max(before_ankle_x) - min(before_ankle_x)) * 0.25)
        self.assertLess(max(ankle_z) - min(ankle_z), (max(before_ankle_z) - min(before_ankle_z)) * 0.25)

    def test_cleanup_pose_frames_keeps_support_foot_shape_more_consistent(self) -> None:
        frames = []
        for shift, toe_spread, heel_spread in (
            (0.0, 0.0, 0.0),
            (1.2, 1.5, -1.0),
            (-0.8, -1.3, 0.9),
            (0.6, 0.8, -0.6),
        ):
            frames.append(
                {
                    "mid_hip": np.array([0.0, 0.0, 0.0], dtype=np.float64),
                    "left_hip": np.array([-5.0, 0.0, 0.0], dtype=np.float64),
                    "right_hip": np.array([5.0, 0.0, 0.0], dtype=np.float64),
                    "left_knee": np.array([-5.0, -15.0, 0.0], dtype=np.float64),
                    "right_knee": np.array([5.0, -15.0, 0.0], dtype=np.float64),
                    "left_ankle": np.array([-5.0 + shift, -30.0, shift], dtype=np.float64),
                    "right_ankle": np.array([5.0, -30.0, 0.0], dtype=np.float64),
                    "left_toes": np.array([-5.0 + shift + toe_spread, -30.0, 8.0 + shift + toe_spread], dtype=np.float64),
                    "right_toes": np.array([5.0, -30.0, 8.0], dtype=np.float64),
                    "left_heel": np.array([-5.0 + shift + heel_spread, -30.0, -4.0 + shift + heel_spread], dtype=np.float64),
                    "right_heel": np.array([5.0, -30.0, -4.0], dtype=np.float64),
                    "left_shoulder": np.array([-8.0, 18.0, 0.0], dtype=np.float64),
                    "right_shoulder": np.array([8.0, 18.0, 0.0], dtype=np.float64),
                    "spine": np.array([0.0, 8.0, 0.0], dtype=np.float64),
                    "chest": np.array([0.0, 18.0, 0.0], dtype=np.float64),
                    "upper_chest": np.array([0.0, 22.0, 0.0], dtype=np.float64),
                    "neck": np.array([0.0, 25.0, 0.0], dtype=np.float64),
                    "head": np.array([0.0, 30.0, 0.0], dtype=np.float64),
                }
            )

        anchors = [_copy_pose_frame(frames[0]) for _ in range(3)]
        before_spans = [
            float(np.linalg.norm(frame["left_toes"] - frame["left_ankle"])) +
            float(np.linalg.norm(frame["left_heel"] - frame["left_ankle"]))
            for frame in frames
        ]
        cleaned, _stats = pipeline.cleanup_pose_frames(frames, anchors)
        after_spans = [
            float(np.linalg.norm(frame["left_toes"] - frame["left_ankle"])) +
            float(np.linalg.norm(frame["left_heel"] - frame["left_ankle"]))
            for frame in cleaned
        ]

        self.assertLess(max(after_spans) - min(after_spans), (max(before_spans) - min(before_spans)) * 0.6)

    def test_adaptive_smoothing_preserves_fast_wrist_accent(self) -> None:
        frames = []
        for wrist_z, hip_x in (
            (0.0, 0.0),
            (0.4, 0.6),
            (12.0, -0.5),
            (0.5, 0.7),
            (0.0, 0.0),
        ):
            frames.append(
                {
                    "mid_hip": np.array([hip_x, 0.0, 0.0], dtype=np.float64),
                    "left_hip": np.array([-5.0 + hip_x, 0.0, 0.0], dtype=np.float64),
                    "right_hip": np.array([5.0 + hip_x, 0.0, 0.0], dtype=np.float64),
                    "left_knee": np.array([-5.0, -15.0, 0.0], dtype=np.float64),
                    "right_knee": np.array([5.0, -15.0, 0.0], dtype=np.float64),
                    "left_ankle": np.array([-5.0, -30.0, 0.0], dtype=np.float64),
                    "right_ankle": np.array([5.0, -30.0, 0.0], dtype=np.float64),
                    "left_toes": np.array([-5.0, -30.0, 8.0], dtype=np.float64),
                    "right_toes": np.array([5.0, -30.0, 8.0], dtype=np.float64),
                    "left_heel": np.array([-5.0, -30.0, -4.0], dtype=np.float64),
                    "right_heel": np.array([5.0, -30.0, -4.0], dtype=np.float64),
                    "left_shoulder": np.array([-8.0, 18.0, 0.0], dtype=np.float64),
                    "right_shoulder": np.array([8.0, 18.0, 0.0], dtype=np.float64),
                    "left_elbow": np.array([-12.0, 16.0, wrist_z * 0.4], dtype=np.float64),
                    "right_elbow": np.array([12.0, 16.0, 0.0], dtype=np.float64),
                    "left_wrist": np.array([-16.0, 14.0, wrist_z], dtype=np.float64),
                    "right_wrist": np.array([16.0, 14.0, 0.0], dtype=np.float64),
                    "spine": np.array([hip_x, 8.0, 0.0], dtype=np.float64),
                    "chest": np.array([hip_x, 18.0, 0.0], dtype=np.float64),
                    "upper_chest": np.array([hip_x, 22.0, 0.0], dtype=np.float64),
                    "neck": np.array([hip_x, 25.0, 0.0], dtype=np.float64),
                    "head": np.array([hip_x, 30.0, 0.0], dtype=np.float64),
                }
            )

        smoothed = _smooth_pose_frames(frames, alpha=0.35)
        wrist_peak = float(smoothed[2]["left_wrist"][2])
        hip_span = max(float(frame["mid_hip"][0]) for frame in smoothed) - min(float(frame["mid_hip"][0]) for frame in smoothed)

        self.assertGreater(wrist_peak, 5.5)
        self.assertLess(hip_span, 0.9)

    def test_extract_motion_loop_finds_cyclic_window(self) -> None:
        def make_frame(phase: float, offset_x: float = 0.0) -> dict[str, np.ndarray]:
            swing = np.sin(phase)
            lift = np.cos(phase) * 0.5
            return {
                "mid_hip": np.array([offset_x, 0.0, 0.0], dtype=np.float64),
                "spine": np.array([offset_x, 8.0, 0.0], dtype=np.float64),
                "chest": np.array([offset_x, 18.0, 0.0], dtype=np.float64),
                "upper_chest": np.array([offset_x, 22.0, 0.0], dtype=np.float64),
                "neck": np.array([offset_x, 25.0, 0.0], dtype=np.float64),
                "head": np.array([offset_x, 30.0, 0.0], dtype=np.float64),
                "left_shoulder": np.array([offset_x - 7.0, 18.0 + lift, swing], dtype=np.float64),
                "right_shoulder": np.array([offset_x + 7.0, 18.0 - lift, -swing], dtype=np.float64),
                "left_elbow": np.array([offset_x - 12.0, 16.0 + lift, swing * 1.5], dtype=np.float64),
                "right_elbow": np.array([offset_x + 12.0, 16.0 - lift, -swing * 1.5], dtype=np.float64),
                "left_wrist": np.array([offset_x - 16.0, 14.0 + lift, swing * 2.0], dtype=np.float64),
                "right_wrist": np.array([offset_x + 16.0, 14.0 - lift, -swing * 2.0], dtype=np.float64),
                "left_hip": np.array([offset_x - 5.0, 0.0, 0.0], dtype=np.float64),
                "right_hip": np.array([offset_x + 5.0, 0.0, 0.0], dtype=np.float64),
                "left_knee": np.array([offset_x - 5.0, -14.0, swing], dtype=np.float64),
                "right_knee": np.array([offset_x + 5.0, -14.0, -swing], dtype=np.float64),
                "left_ankle": np.array([offset_x - 5.0, -28.0 + lift, swing], dtype=np.float64),
                "right_ankle": np.array([offset_x + 5.0, -28.0 - lift, -swing], dtype=np.float64),
                "left_toes": np.array([offset_x - 5.0, -28.0 + lift, 8.0 + swing], dtype=np.float64),
                "right_toes": np.array([offset_x + 5.0, -28.0 - lift, 8.0 - swing], dtype=np.float64),
                "left_heel": np.array([offset_x - 5.0, -28.0 + lift, -4.0 + swing], dtype=np.float64),
                "right_heel": np.array([offset_x + 5.0, -28.0 - lift, -4.0 - swing], dtype=np.float64),
            }

        phases = [0.9, 1.7, 0.0, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6, 0.0, 0.8]
        frames = [make_frame(phase, offset_x=(0.0 if 2 <= idx <= 10 else idx * 2.0)) for idx, phase in enumerate(phases)]

        trimmed, stats = pipeline.extract_motion_loop(frames, fps=12.0, mode="auto")
        self.assertGreater(stats["applied"], 0.5)
        self.assertEqual(int(stats["start"]), 2)
        self.assertEqual(int(stats["end"]), 10)
        self.assertEqual(len(trimmed), 9)

    def test_analyze_motion_loopability_distinguishes_cyclic_and_oneshot(self) -> None:
        def make_cycle_frame(phase: float) -> dict[str, np.ndarray]:
            swing = np.sin(phase)
            return {
                "mid_hip": np.array([0.0, 0.0, 0.0], dtype=np.float64),
                "spine": np.array([0.0, 8.0, 0.0], dtype=np.float64),
                "chest": np.array([0.0, 18.0, 0.0], dtype=np.float64),
                "upper_chest": np.array([0.0, 22.0, 0.0], dtype=np.float64),
                "neck": np.array([0.0, 25.0, 0.0], dtype=np.float64),
                "head": np.array([0.0, 30.0, 0.0], dtype=np.float64),
                "left_shoulder": np.array([-7.0, 18.0, swing], dtype=np.float64),
                "right_shoulder": np.array([7.0, 18.0, -swing], dtype=np.float64),
                "left_elbow": np.array([-12.0, 16.0, swing * 1.5], dtype=np.float64),
                "right_elbow": np.array([12.0, 16.0, -swing * 1.5], dtype=np.float64),
                "left_wrist": np.array([-16.0, 14.0, swing * 2.0], dtype=np.float64),
                "right_wrist": np.array([16.0, 14.0, -swing * 2.0], dtype=np.float64),
                "left_hip": np.array([-5.0, 0.0, 0.0], dtype=np.float64),
                "right_hip": np.array([5.0, 0.0, 0.0], dtype=np.float64),
                "left_knee": np.array([-5.0, -14.0, swing], dtype=np.float64),
                "right_knee": np.array([5.0, -14.0, -swing], dtype=np.float64),
                "left_ankle": np.array([-5.0, -28.0, swing], dtype=np.float64),
                "right_ankle": np.array([5.0, -28.0, -swing], dtype=np.float64),
                "left_toes": np.array([-5.0, -28.0, 8.0 + swing], dtype=np.float64),
                "right_toes": np.array([5.0, -28.0, 8.0 - swing], dtype=np.float64),
                "left_heel": np.array([-5.0, -28.0, -4.0], dtype=np.float64),
                "right_heel": np.array([5.0, -28.0, -4.0], dtype=np.float64),
            }

        cyclic = [make_cycle_frame(phase) for phase in (0.0, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6, 0.0, 0.8, 1.6, 2.4)]
        oneshot = [make_cycle_frame(phase) for phase in np.linspace(0.0, 3.0, 12)]
        for idx, frame in enumerate(oneshot):
            drift = idx * 1.5
            frame["mid_hip"] = frame["mid_hip"] + np.array([drift, 0.0, drift * 0.3], dtype=np.float64)
            for key in frame.keys():
                if key != "mid_hip":
                    frame[key] = frame[key] + np.array([drift, 0.0, drift * 0.3], dtype=np.float64)
            frame["left_wrist"] = frame["left_wrist"] + np.array([0.0, idx * 1.2, idx * 0.9], dtype=np.float64)
            frame["right_wrist"] = frame["right_wrist"] + np.array([0.0, idx * 0.8, -idx * 0.7], dtype=np.float64)
            frame["head"] = frame["head"] + np.array([0.0, idx * 0.6, idx * 0.5], dtype=np.float64)

        cyclic_stats = pipeline.analyze_motion_loopability(cyclic, fps=12.0)
        oneshot_stats = pipeline.analyze_motion_loopability(oneshot, fps=12.0)
        self.assertEqual(str(cyclic_stats["label"]), "cyclic")
        self.assertEqual(str(oneshot_stats["label"]), "oneshot")

    def test_unwrap_motion_rotation_channels_reduces_large_branch_flips(self) -> None:
        motion = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 170.0, 0.0, 0.0, 170.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, -175.0, 0.0, 0.0, -175.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, -170.0, 0.0, 0.0, -170.0],
        ]

        unwrapped, stats = pipeline.unwrap_motion_rotation_channels(motion)
        self.assertGreater(stats["applied"], 0.5)
        self.assertGreater(stats["changed_values"], 0.0)
        self.assertGreater(stats["max_step_before"], stats["max_step_after"])
        self.assertAlmostEqual(float(unwrapped[1][5]), 185.0, places=6)
        self.assertAlmostEqual(float(unwrapped[2][5]), 190.0, places=6)

    def test_skeleton_profile_overrides_rest_offsets(self) -> None:
        rest_offsets = {
            "hips": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            "spine": np.array([0.0, 30.0, 0.0], dtype=np.float64),
            "leftUpperLeg": np.array([-6.0, -16.0, 2.0], dtype=np.float64),
        }
        profile = {
            "joint_blend": {
                "spine": 0.5,
                "leftUpperLeg": 1.0,
            },
            "joint_offsets": {
                "spine": [0.0, 2.0, 0.0],
                "leftUpperLeg": [-0.6, -1.6, 0.2],
            }
        }

        updated, stats = pipeline.apply_skeleton_profile_to_rest_offsets(rest_offsets, profile)
        self.assertGreater(stats["applied"], 0.5)
        self.assertEqual(int(stats["overridden"]), 2)
        self.assertAlmostEqual(float(stats["scale_ratio"]), 12.5, places=6)
        self.assertAlmostEqual(float(stats["group_scale_ratios"]["torso"]), 15.0, places=6)
        self.assertAlmostEqual(float(stats["group_scale_ratios"]["left_leg"]), 10.0, places=6)
        self.assertAlmostEqual(float(stats["avg_blend"]), 0.75, places=6)
        self.assertAlmostEqual(float(updated["spine"][1]), 30.0, places=6)
        self.assertAlmostEqual(float(updated["leftUpperLeg"][0]), -6.0, places=6)
        self.assertAlmostEqual(float(updated["leftUpperLeg"][2]), 2.0, places=6)

    def test_skeleton_profile_uses_chain_aware_scale_ratios(self) -> None:
        rest_offsets = {
            "hips": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            "spine": np.array([0.0, 20.0, 0.0], dtype=np.float64),
            "chest": np.array([0.0, 10.0, 0.0], dtype=np.float64),
            "leftUpperLeg": np.array([-12.0, -24.0, 0.0], dtype=np.float64),
            "rightUpperLeg": np.array([12.0, -24.0, 0.0], dtype=np.float64),
        }
        profile = {
            "joint_blend": {
                "spine": 1.0,
                "chest": 1.0,
                "leftUpperLeg": 1.0,
                "rightUpperLeg": 1.0,
            },
            "joint_offsets": {
                "spine": [0.0, 2.0, 0.0],
                "chest": [0.0, 1.0, 0.0],
                "leftUpperLeg": [-1.0, -2.0, 0.0],
                "rightUpperLeg": [1.0, -2.0, 0.0],
            },
        }

        updated, stats = pipeline.apply_skeleton_profile_to_rest_offsets(rest_offsets, profile)
        self.assertGreater(stats["applied"], 0.5)
        self.assertAlmostEqual(float(stats["group_scale_ratios"]["torso"]), 10.0, places=6)
        self.assertAlmostEqual(float(stats["group_scale_ratios"]["left_leg"]), 12.0, places=6)
        self.assertAlmostEqual(float(stats["group_scale_ratios"]["right_leg"]), 12.0, places=6)
        self.assertAlmostEqual(float(updated["spine"][1]), 20.0, places=6)
        self.assertAlmostEqual(float(updated["chest"][1]), 10.0, places=6)
        self.assertAlmostEqual(float(updated["leftUpperLeg"][0]), -12.0, places=6)
        self.assertAlmostEqual(float(updated["leftUpperLeg"][1]), -24.0, places=6)

    def test_skeleton_profile_can_normalize_final_rest_offsets_to_vrm_humanoid_baseline(self) -> None:
        rest_offsets = {
            "hips": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            "spine": np.array([0.0, 20.0, 0.0], dtype=np.float64),
            "chest": np.array([0.0, 10.0, 0.0], dtype=np.float64),
            "upperChest": np.array([0.0, 6.0, 0.0], dtype=np.float64),
            "neck": np.array([0.0, 12.0, 0.0], dtype=np.float64),
            "head": np.array([0.0, 2.0, 0.0], dtype=np.float64),
        }
        profile = {"normalize_to_vrm_humanoid": True}

        updated, stats = pipeline.apply_skeleton_profile_to_rest_offsets(rest_offsets, profile)
        self.assertGreater(stats["humanoid_baseline_applied"], 0.5)
        self.assertIn("torso", stats["humanoid_baseline_chains"])
        total = sum(float(np.linalg.norm(updated[name])) for name in ("spine", "chest", "upperChest", "neck", "head"))
        self.assertAlmostEqual(float(np.linalg.norm(updated["neck"])) / total, 0.14, places=3)
        self.assertAlmostEqual(float(np.linalg.norm(updated["head"])) / total, 0.28, places=3)

    def test_blend_motion_loop_edges_reduces_first_last_gap(self) -> None:
        def make_frame(phase: float, edge_bias: float = 0.0) -> dict[str, np.ndarray]:
            swing = np.sin(phase)
            lift = np.cos(phase) * 0.4
            return {
                "mid_hip": np.array([0.0, 0.0, 0.0], dtype=np.float64),
                "spine": np.array([0.0, 8.0, 0.0], dtype=np.float64),
                "chest": np.array([0.0, 18.0, 0.0], dtype=np.float64),
                "upper_chest": np.array([0.0, 22.0, 0.0], dtype=np.float64),
                "neck": np.array([0.0, 25.0, 0.0], dtype=np.float64),
                "head": np.array([0.0, 30.0, edge_bias], dtype=np.float64),
                "left_shoulder": np.array([-7.0, 18.0 + lift, swing + edge_bias], dtype=np.float64),
                "right_shoulder": np.array([7.0, 18.0 - lift, -swing - edge_bias], dtype=np.float64),
                "left_elbow": np.array([-12.0, 16.0 + lift, swing * 1.5 + edge_bias], dtype=np.float64),
                "right_elbow": np.array([12.0, 16.0 - lift, -swing * 1.5 - edge_bias], dtype=np.float64),
                "left_wrist": np.array([-16.0, 14.0 + lift, swing * 2.0 + edge_bias], dtype=np.float64),
                "right_wrist": np.array([16.0, 14.0 - lift, -swing * 2.0 - edge_bias], dtype=np.float64),
                "left_hip": np.array([-5.0, 0.0, 0.0], dtype=np.float64),
                "right_hip": np.array([5.0, 0.0, 0.0], dtype=np.float64),
                "left_knee": np.array([-5.0, -14.0, swing], dtype=np.float64),
                "right_knee": np.array([5.0, -14.0, -swing], dtype=np.float64),
                "left_ankle": np.array([-5.0, -28.0 + lift, swing], dtype=np.float64),
                "right_ankle": np.array([5.0, -28.0 - lift, -swing], dtype=np.float64),
                "left_toes": np.array([-5.0, -28.0 + lift, 8.0 + swing + edge_bias], dtype=np.float64),
                "right_toes": np.array([5.0, -28.0 - lift, 8.0 - swing - edge_bias], dtype=np.float64),
                "left_heel": np.array([-5.0, -28.0 + lift, -4.0 + swing], dtype=np.float64),
                "right_heel": np.array([5.0, -28.0 - lift, -4.0 - swing], dtype=np.float64),
            }

        frames = [make_frame(idx * 0.6, edge_bias=(1.0 if idx == 0 else (-1.0 if idx == 11 else 0.0))) for idx in range(12)]
        before = float(np.linalg.norm(pipeline._loop_feature_vector(frames[0]) - pipeline._loop_feature_vector(frames[-1])))
        blended, stats = pipeline.blend_motion_loop_edges(frames, fps=24.0, mode="auto")
        after = float(np.linalg.norm(pipeline._loop_feature_vector(blended[0]) - pipeline._loop_feature_vector(blended[-1])))

        self.assertGreater(stats["applied"], 0.5)
        self.assertGreaterEqual(stats["blend_frames"], 2.0)
        self.assertLess(after, before)

    def test_cleanup_pose_frames_stabilizes_pelvis_during_foot_contact(self) -> None:
        frames = []
        for hip_shift in (0.0, 1.8, -1.4, 1.1):
            frames.append(
                {
                    "mid_hip": np.array([hip_shift, 0.0, hip_shift * 0.5], dtype=np.float64),
                    "left_hip": np.array([-5.0 + hip_shift, 0.0, hip_shift * 0.5], dtype=np.float64),
                    "right_hip": np.array([5.0 + hip_shift, 0.0, hip_shift * 0.5], dtype=np.float64),
                    "left_knee": np.array([-5.0, -15.0, 0.0], dtype=np.float64),
                    "right_knee": np.array([5.0 + hip_shift * 0.5, -15.0, hip_shift * 0.25], dtype=np.float64),
                    "left_ankle": np.array([-5.0, -30.0, 0.0], dtype=np.float64),
                    "right_ankle": np.array([5.0 + hip_shift * 0.5, -30.0, hip_shift * 0.25], dtype=np.float64),
                    "left_toes": np.array([-5.0, -30.0, 8.0], dtype=np.float64),
                    "right_toes": np.array([5.0 + hip_shift * 0.5, -30.0, 8.0 + hip_shift * 0.25], dtype=np.float64),
                    "left_heel": np.array([-5.0, -30.0, -4.0], dtype=np.float64),
                    "right_heel": np.array([5.0 + hip_shift * 0.5, -30.0, -4.0 + hip_shift * 0.25], dtype=np.float64),
                    "spine": np.array([hip_shift, 8.0, hip_shift * 0.5], dtype=np.float64),
                    "chest": np.array([hip_shift, 18.0, hip_shift * 0.5], dtype=np.float64),
                    "upper_chest": np.array([hip_shift, 22.0, hip_shift * 0.5], dtype=np.float64),
                    "neck": np.array([hip_shift, 25.0, hip_shift * 0.5], dtype=np.float64),
                    "head": np.array([hip_shift, 30.0, hip_shift * 0.5], dtype=np.float64),
                    "left_shoulder": np.array([-8.0 + hip_shift, 18.0, hip_shift * 0.5], dtype=np.float64),
                    "right_shoulder": np.array([8.0 + hip_shift, 18.0, hip_shift * 0.5], dtype=np.float64),
                }
            )

        anchors = [_copy_pose_frame(frames[0]) for _ in range(3)]
        before_x = [float(frame["mid_hip"][0]) for frame in frames]
        before_z = [float(frame["mid_hip"][2]) for frame in frames]
        before_steps = [
            float(np.linalg.norm(frames[idx]["mid_hip"][[0, 2]] - frames[idx - 1]["mid_hip"][[0, 2]]))
            for idx in range(1, len(frames))
        ]
        cleaned, stats = pipeline.cleanup_pose_frames(frames, anchors)
        self.assertGreater(stats["pelvis_contact_frames"], 0.0)
        self.assertGreater(stats["root_stabilized_frames"], 0.0)

        after_x = [float(frame["mid_hip"][0]) for frame in cleaned]
        after_z = [float(frame["mid_hip"][2]) for frame in cleaned]
        after_steps = [
            float(np.linalg.norm(cleaned[idx]["mid_hip"][[0, 2]] - cleaned[idx - 1]["mid_hip"][[0, 2]]))
            for idx in range(1, len(cleaned))
        ]
        self.assertLess(max(after_x) - min(after_x), (max(before_x) - min(before_x)) * 0.6)
        self.assertLess(max(after_z) - min(after_z), (max(before_z) - min(before_z)) * 0.6)
        self.assertLess(max(after_steps), max(before_steps) * 0.65)

    def test_cleanup_pose_frames_stabilizes_root_height_during_support_window(self) -> None:
        frames = []
        for idx, hip_y in enumerate((0.0, 3.5, -2.8, 2.2, -1.7)):
            sway = (-1.2 if idx % 2 else 1.0) * 0.9
            frames.append(
                {
                    "mid_hip": np.array([sway, hip_y, sway * 0.4], dtype=np.float64),
                    "left_hip": np.array([-5.0 + sway, hip_y, sway * 0.4], dtype=np.float64),
                    "right_hip": np.array([5.0 + sway, hip_y, sway * 0.4], dtype=np.float64),
                    "left_knee": np.array([-5.0, -15.0, 0.0], dtype=np.float64),
                    "right_knee": np.array([5.0, -15.0, 0.0], dtype=np.float64),
                    "left_ankle": np.array([-5.0, -30.0, 0.0], dtype=np.float64),
                    "right_ankle": np.array([5.0, -30.0, 0.0], dtype=np.float64),
                    "left_toes": np.array([-5.0, -30.0, 8.0], dtype=np.float64),
                    "right_toes": np.array([5.0, -30.0, 8.0], dtype=np.float64),
                    "left_heel": np.array([-5.0, -30.0, -4.0], dtype=np.float64),
                    "right_heel": np.array([5.0, -30.0, -4.0], dtype=np.float64),
                    "spine": np.array([sway, 8.0 + hip_y, sway * 0.4], dtype=np.float64),
                    "chest": np.array([sway, 18.0 + hip_y, sway * 0.4], dtype=np.float64),
                    "upper_chest": np.array([sway, 22.0 + hip_y, sway * 0.4], dtype=np.float64),
                    "neck": np.array([sway, 25.0 + hip_y, sway * 0.4], dtype=np.float64),
                    "head": np.array([sway, 30.0 + hip_y, sway * 0.4], dtype=np.float64),
                    "left_shoulder": np.array([-8.0 + sway, 18.0 + hip_y, sway * 0.4], dtype=np.float64),
                    "right_shoulder": np.array([8.0 + sway, 18.0 + hip_y, sway * 0.4], dtype=np.float64),
                }
            )

        anchors = [_copy_pose_frame(frames[0]) for _ in range(3)]
        before_y = [float(frame["mid_hip"][1]) for frame in frames]
        cleaned, stats = pipeline.cleanup_pose_frames(frames, anchors)

        self.assertGreater(stats["root_stabilized_frames"], 0.0)
        after_y = [float(frame["mid_hip"][1]) for frame in cleaned]
        self.assertLess(max(after_y) - min(after_y), (max(before_y) - min(before_y)) * 0.7)

    def test_cleanup_pose_frames_applies_leg_ik_on_contact_frames(self) -> None:
        base_hip = np.array([-5.0, 0.0, 0.0], dtype=np.float64)
        base_knee = np.array([-5.0, -15.0, 0.0], dtype=np.float64)
        base_ankle = np.array([-5.0, -30.0, 0.0], dtype=np.float64)
        frames = []
        for hip_shift, knee_x in ((0.0, -9.0), (1.5, -1.0), (-1.2, -10.5), (0.8, -2.0)):
            frames.append(
                {
                    "mid_hip": np.array([hip_shift, 0.0, 0.0], dtype=np.float64),
                    "left_hip": np.array([base_hip[0] + hip_shift, base_hip[1], base_hip[2]], dtype=np.float64),
                    "right_hip": np.array([5.0 + hip_shift, 0.0, 0.0], dtype=np.float64),
                    "left_knee": np.array([knee_x, -15.0, 4.0], dtype=np.float64),
                    "right_knee": np.array([5.0, -15.0, 0.0], dtype=np.float64),
                    "left_ankle": np.array(base_ankle, dtype=np.float64),
                    "right_ankle": np.array([5.0, -30.0, 0.0], dtype=np.float64),
                    "left_toes": np.array([-5.0, -30.0, 8.0], dtype=np.float64),
                    "right_toes": np.array([5.0, -30.0, 8.0], dtype=np.float64),
                    "left_heel": np.array([-5.0, -30.0, -4.0], dtype=np.float64),
                    "right_heel": np.array([5.0, -30.0, -4.0], dtype=np.float64),
                    "spine": np.array([hip_shift, 8.0, 0.0], dtype=np.float64),
                    "chest": np.array([hip_shift, 18.0, 0.0], dtype=np.float64),
                    "upper_chest": np.array([hip_shift, 22.0, 0.0], dtype=np.float64),
                    "neck": np.array([hip_shift, 25.0, 0.0], dtype=np.float64),
                    "head": np.array([hip_shift, 30.0, 0.0], dtype=np.float64),
                    "left_shoulder": np.array([-8.0 + hip_shift, 18.0, 0.0], dtype=np.float64),
                    "right_shoulder": np.array([8.0 + hip_shift, 18.0, 0.0], dtype=np.float64),
                }
            )

        anchors = [
            {
                **_copy_pose_frame(frames[0]),
                "left_hip": np.array(base_hip, dtype=np.float64),
                "left_knee": np.array(base_knee, dtype=np.float64),
                "left_ankle": np.array(base_ankle, dtype=np.float64),
            }
            for _ in range(3)
        ]

        cleaned, stats = pipeline.cleanup_pose_frames(frames, anchors)
        self.assertGreater(stats["leg_ik_frames"], 0.0)

        target_upper = float(np.linalg.norm(base_knee - base_hip))
        target_lower = float(np.linalg.norm(base_ankle - base_knee))
        for frame in cleaned:
            upper = float(np.linalg.norm(frame["left_knee"] - frame["left_hip"]))
            lower = float(np.linalg.norm(frame["left_ankle"] - frame["left_knee"]))
            self.assertAlmostEqual(upper, target_upper, places=3)
            self.assertAlmostEqual(lower, target_lower, places=3)

    def test_cleanup_pose_frames_can_skip_contact_cleanup_for_oneshot_motion(self) -> None:
        frames = []
        for shift in (0.0, 1.5, -1.0, 0.5):
            frame = {
                "mid_hip": np.array([0.0, 0.0, 0.0], dtype=np.float64),
                "left_hip": np.array([-5.0, 0.0, 0.0], dtype=np.float64),
                "right_hip": np.array([5.0, 0.0, 0.0], dtype=np.float64),
                "left_knee": np.array([-5.0, -15.0, 0.0], dtype=np.float64),
                "right_knee": np.array([5.0, -15.0, 0.0], dtype=np.float64),
                "left_ankle": np.array([-5.0 + shift, -30.0, shift], dtype=np.float64),
                "right_ankle": np.array([5.0, -30.0, 0.0], dtype=np.float64),
                "left_toes": np.array([-5.0 + shift, -30.0, 8.0 + shift], dtype=np.float64),
                "right_toes": np.array([5.0, -30.0, 8.0], dtype=np.float64),
                "left_heel": np.array([-5.0 + shift, -30.0, -4.0 + shift], dtype=np.float64),
                "right_heel": np.array([5.0, -30.0, -4.0], dtype=np.float64),
                "left_shoulder": np.array([-8.0, 18.0, 0.0], dtype=np.float64),
                "right_shoulder": np.array([8.0, 18.0, 0.0], dtype=np.float64),
                "spine": np.array([0.0, 8.0, 0.0], dtype=np.float64),
                "chest": np.array([0.0, 18.0, 0.0], dtype=np.float64),
                "upper_chest": np.array([0.0, 22.0, 0.0], dtype=np.float64),
                "neck": np.array([0.0, 25.0, 0.0], dtype=np.float64),
                "head": np.array([0.0, 30.0, 0.0], dtype=np.float64),
            }
            frames.append(frame)

        anchors = [_copy_pose_frame(frames[0]) for _ in range(3)]
        cleaned, stats = pipeline.cleanup_pose_frames(frames, anchors, use_contact_cleanup=False)

        self.assertEqual(stats["contact_windows"], 0.0)
        self.assertEqual(stats["contact_frames"], 0.0)
        self.assertEqual(stats["pelvis_contact_frames"], 0.0)
        self.assertEqual(stats["leg_ik_frames"], 0.0)
        target_len = float(np.linalg.norm(anchors[0]["left_ankle"] - anchors[0]["left_knee"]))
        for frame in cleaned:
            self.assertAlmostEqual(
                float(np.linalg.norm(frame["left_ankle"] - frame["left_knee"])),
                target_len,
                places=4,
            )


if __name__ == "__main__":
    unittest.main()
