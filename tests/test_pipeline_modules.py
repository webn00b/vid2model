import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from vid2model_lib import pipeline
from vid2model_lib.pipeline import DEFAULT_POSE_CORRECTIONS
from vid2model_lib.pipeline_auto_pose import AUTO_POSE_PRESETS, build_pose_correction_profile
from vid2model_lib.pipeline_channels import frame_channels as module_frame_channels
from vid2model_lib.pipeline_mirror import looks_mirrored, mirror_pose_points
from vid2model_lib.pipeline_retarget import (
    apply_pose_corrections,
    build_reference_basis,
    canonicalize_pose_points,
    median_pose_sample,
)
from vid2model_lib.pipeline_rest_offsets import build_rest_offsets


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


def rotate_pose_y(pts: dict[str, np.ndarray], angle_deg: float) -> dict[str, np.ndarray]:
    angle_rad = np.radians(angle_deg)
    cos_y = float(np.cos(angle_rad))
    sin_y = float(np.sin(angle_rad))
    rot = np.array(
        [
            [cos_y, 0.0, sin_y],
            [0.0, 1.0, 0.0],
            [-sin_y, 0.0, cos_y],
        ],
        dtype=np.float64,
    )
    pivot = pts["mid_hip"]
    return {
        key: pivot + rot @ (value - pivot)
        for key, value in pts.items()
    }


class PipelineRetargetModuleTests(unittest.TestCase):
    def test_pipeline_public_contract_is_explicit(self) -> None:
        expected = [
            "AUTO_FEATURE_NAMES",
            "AUTO_POSE_PRESETS",
            "DEFAULT_POSE_CORRECTIONS",
            "LOOP_FEATURE_KEYS",
            "PoseCorrectionProfile",
            "analyze_motion_loopability",
            "apply_lower_body_rotation_mode",
            "apply_manual_root_yaw_offset",
            "apply_skeleton_profile_to_rest_offsets",
            "apply_upper_body_rotation_scale",
            "blend_motion_loop_edges",
            "build_pose_correction_profile",
            "build_rest_offsets",
            "bvh_hierarchy_lines",
            "clamp_roi_box",
            "cleanup_pose_frames",
            "collect_detected_pose_samples",
            "convert_video_to_bvh",
            "extract_motion_loop",
            "extract_pose_bbox_pixels",
            "fill_pose_gaps",
            "frame_channels",
            "gamma_lut",
            "interpolate_pose_points",
            "normalize_motion_root_yaw",
            "preprocess_video_frame",
            "resolve_auto_pose_corrections",
            "resize_frame_for_detection",
            "unwrap_motion_rotation_channels",
            "update_tracking_roi",
        ]

        self.assertEqual(pipeline.__all__, expected)
        self.assertTrue(all(not name.startswith("_") for name in pipeline.__all__))
        self.assertTrue(all(hasattr(pipeline, name) for name in pipeline.__all__))

    def test_median_pose_sample_raises_on_empty_samples(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "No valid pose frames detected"):
            median_pose_sample([])

    def test_build_reference_basis_keeps_up_axis_aligned_with_torso(self) -> None:
        sample = make_pose_points()
        sample["chest"] = np.array([0.0, -20.0, 0.0], dtype=np.float64)
        sample["upper_chest"] = np.array([0.0, -25.0, 0.0], dtype=np.float64)
        sample["neck"] = np.array([0.0, -30.0, 0.0], dtype=np.float64)
        sample["head"] = np.array([0.0, -40.0, 0.0], dtype=np.float64)

        basis, origin = build_reference_basis(sample)
        up_hint = sample["chest"] - origin

        self.assertGreater(float(np.dot(basis[:, 1], up_hint)), 0.0)
        self.assertAlmostEqual(float(np.linalg.det(basis)), 1.0, places=6)

    def test_canonicalize_pose_points_subtracts_origin_in_basis_space(self) -> None:
        pts = make_pose_points()
        basis, origin = build_reference_basis(pts)

        canonical = canonicalize_pose_points(pts, basis, origin)

        self.assertTrue(np.allclose(canonical["mid_hip"], np.zeros(3), atol=1e-6))
        self.assertLess(float(canonical["left_shoulder"][0]), 0.0)
        self.assertGreater(float(canonical["right_shoulder"][0]), 0.0)

    def test_apply_pose_corrections_auto_grounding_lifts_feet_to_ground(self) -> None:
        pts = make_pose_points()
        for key in pts:
            pts[key] = pts[key] + np.array([0.0, -12.0, 0.0], dtype=np.float64)

        corrected = apply_pose_corrections(pts, DEFAULT_POSE_CORRECTIONS)
        foot_y = [
            corrected["left_ankle"][1],
            corrected["right_ankle"][1],
            corrected["left_heel"][1],
            corrected["right_heel"][1],
        ]

        self.assertAlmostEqual(float(min(foot_y)), 0.0, places=6)

    def test_apply_pose_corrections_preserves_expressive_torso_depth(self) -> None:
        pts = make_pose_points()
        pts["spine"] = np.array([0.0, 10.0, 8.0], dtype=np.float64)
        pts["chest"] = np.array([0.0, 20.0, 14.0], dtype=np.float64)
        pts["upper_chest"] = np.array([0.0, 24.0, 18.0], dtype=np.float64)
        pts["neck"] = np.array([0.0, 30.0, 20.0], dtype=np.float64)
        pts["head"] = np.array([0.0, 40.0, 22.0], dtype=np.float64)

        corrected = apply_pose_corrections(
            pts,
            replace(DEFAULT_POSE_CORRECTIONS, shoulder_tracking=False),
        )

        self.assertGreater(float(corrected["chest"][2]), 9.0)
        self.assertGreater(float(corrected["head"][2]), 16.0)

    def test_apply_pose_corrections_preserves_preset_specific_body_bend_strength(self) -> None:
        pts = make_pose_points()
        pts["spine"] = np.array([0.0, 10.0, 8.0], dtype=np.float64)
        pts["chest"] = np.array([0.0, 20.0, 14.0], dtype=np.float64)
        pts["upper_chest"] = np.array([0.0, 24.0, 18.0], dtype=np.float64)
        pts["neck"] = np.array([0.0, 30.0, 20.0], dtype=np.float64)
        pts["head"] = np.array([0.0, 40.0, 22.0], dtype=np.float64)

        def preset_bend_profile(name: str):
            return replace(
                build_pose_correction_profile(AUTO_POSE_PRESETS[name]),
                shoulder_tracking=False,
                auto_grounding=False,
                use_arm_ik=False,
                use_leg_ik=False,
                body_collider_mode=0,
                arm_horizontal_offset_percent=0.0,
                arm_vertical_offset_percent=0.0,
                hip_depth_scale_percent=100.0,
                hip_y_position_offset_percent=0.0,
                hip_z_position_offset_percent=0.0,
            )

        low_camera = apply_pose_corrections(pts, preset_bend_profile("low_camera"))
        wide_arms = apply_pose_corrections(pts, preset_bend_profile("wide_arms"))
        crouched = apply_pose_corrections(pts, preset_bend_profile("crouched"))

        self.assertLess(float(low_camera["chest"][2]), float(wide_arms["chest"][2]))
        self.assertLess(float(wide_arms["chest"][2]), float(crouched["chest"][2]))
        self.assertGreater(float(wide_arms["chest"][2] - low_camera["chest"][2]), 0.5)
        self.assertGreater(float(crouched["chest"][2] - wide_arms["chest"][2]), 0.5)

    def test_apply_pose_corrections_softens_near_body_arm_and_leg_ik(self) -> None:
        pts = make_pose_points()
        pts["left_elbow"] = np.array([-1.0, 18.0, 0.0], dtype=np.float64)
        pts["right_elbow"] = np.array([1.0, 18.0, 0.0], dtype=np.float64)
        pts["left_wrist"] = np.array([-0.5, 14.0, 0.0], dtype=np.float64)
        pts["right_wrist"] = np.array([0.5, 14.0, 0.0], dtype=np.float64)
        pts["left_knee"] = np.array([-0.4, -15.0, 0.0], dtype=np.float64)
        pts["right_knee"] = np.array([0.4, -15.0, 0.0], dtype=np.float64)
        pts["left_ankle"] = np.array([-0.2, -30.0, 0.0], dtype=np.float64)
        pts["right_ankle"] = np.array([0.2, -30.0, 0.0], dtype=np.float64)

        corrected = apply_pose_corrections(pts, DEFAULT_POSE_CORRECTIONS)

        self.assertLess(abs(float(corrected["left_wrist"][0])), 3.0)
        self.assertLess(abs(float(corrected["right_wrist"][0])), 3.0)
        self.assertLess(abs(float(corrected["left_elbow"][0])), 2.5)
        self.assertLess(abs(float(corrected["right_elbow"][0])), 2.5)
        self.assertLess(abs(float(corrected["left_ankle"][0])), 1.0)
        self.assertLess(abs(float(corrected["right_ankle"][0])), 1.0)

    def test_apply_pose_corrections_preserves_compact_limbs_without_forcing_full_width(self) -> None:
        pts = make_pose_points()
        pts["left_elbow"] = np.array([-0.6, 18.0, 0.0], dtype=np.float64)
        pts["right_elbow"] = np.array([0.6, 18.0, 0.0], dtype=np.float64)
        pts["left_wrist"] = np.array([-0.3, 14.0, 0.0], dtype=np.float64)
        pts["right_wrist"] = np.array([0.3, 14.0, 0.0], dtype=np.float64)
        pts["left_hand"] = np.array([-0.2, 13.0, 0.0], dtype=np.float64)
        pts["right_hand"] = np.array([0.2, 13.0, 0.0], dtype=np.float64)
        pts["left_knee"] = np.array([-0.25, -15.0, 0.0], dtype=np.float64)
        pts["right_knee"] = np.array([0.25, -15.0, 0.0], dtype=np.float64)
        pts["left_ankle"] = np.array([-0.15, -30.0, 0.0], dtype=np.float64)
        pts["right_ankle"] = np.array([0.15, -30.0, 0.0], dtype=np.float64)

        corrected = apply_pose_corrections(pts, DEFAULT_POSE_CORRECTIONS)

        self.assertLess(abs(float(corrected["left_elbow"][0])), 1.5)
        self.assertLess(abs(float(corrected["right_elbow"][0])), 1.5)
        self.assertLess(abs(float(corrected["left_wrist"][0])), 1.4)
        self.assertLess(abs(float(corrected["right_wrist"][0])), 1.4)
        self.assertLess(abs(float(corrected["left_hand"][0])), 1.4)
        self.assertLess(abs(float(corrected["right_hand"][0])), 1.4)
        self.assertLess(abs(float(corrected["left_ankle"][0])), 0.5)
        self.assertLess(abs(float(corrected["right_ankle"][0])), 0.5)

    def test_apply_pose_corrections_keeps_crossed_limbs_separated_on_bad_input(self) -> None:
        pts = make_pose_points()
        pts["left_elbow"] = np.array([1.5, 18.0, 0.0], dtype=np.float64)
        pts["right_elbow"] = np.array([-1.5, 18.0, 0.0], dtype=np.float64)
        pts["left_wrist"] = np.array([2.5, 14.0, 0.0], dtype=np.float64)
        pts["right_wrist"] = np.array([-2.5, 14.0, 0.0], dtype=np.float64)
        pts["left_hand"] = np.array([3.0, 13.0, 0.0], dtype=np.float64)
        pts["right_hand"] = np.array([-3.0, 13.0, 0.0], dtype=np.float64)
        pts["left_knee"] = np.array([0.8, -15.0, 0.0], dtype=np.float64)
        pts["right_knee"] = np.array([-0.8, -15.0, 0.0], dtype=np.float64)
        pts["left_ankle"] = np.array([0.6, -30.0, 0.0], dtype=np.float64)
        pts["right_ankle"] = np.array([-0.6, -30.0, 0.0], dtype=np.float64)
        pts["left_toes"] = np.array([0.5, -30.0, 8.0], dtype=np.float64)
        pts["right_toes"] = np.array([-0.5, -30.0, 8.0], dtype=np.float64)

        corrected = apply_pose_corrections(pts, DEFAULT_POSE_CORRECTIONS)

        self.assertLess(float(corrected["left_wrist"][0]), -2.0)
        self.assertGreater(float(corrected["right_wrist"][0]), 2.0)
        self.assertLess(float(corrected["left_hand"][0]), -2.0)
        self.assertGreater(float(corrected["right_hand"][0]), 2.0)
        self.assertLess(float(corrected["left_ankle"][0]), -0.6)
        self.assertGreater(float(corrected["right_ankle"][0]), 0.6)
        self.assertLess(float(corrected["left_toes"][0]), -0.6)
        self.assertGreater(float(corrected["right_toes"][0]), 0.6)

    def test_quality_summary_flags_risky_tracking(self) -> None:
        quality = pipeline._build_quality_summary(
            source_frame_count=100,
            detected_count=58,
            interpolated_frames=18,
            carried_frames=16,
            cleanup_stats={
                "side_swaps": 8.0,
                "contact_windows": 0.0,
                "contact_frames": 0.0,
            },
            yaw_norm_stats={"normalized_applied": 1.0},
            unwrap_stats={"changed_values": 64.0},
            loopability={},
            pre_cleanup_loopability={"label": "oneshot"},
            skeleton_profile_stats={"applied": 0.0},
            pose_backend="solutions",
            contact_cleanup_enabled=True,
        )

        self.assertEqual(quality["rating"], "poor")
        self.assertFalse(quality["tracking_ok"])
        self.assertTrue(quality["retarget_risk"])
        self.assertIn("low_detect_ratio", quality["reasons"])
        self.assertIn("high_carried_ratio", quality["reasons"])

    def test_cleanup_evaluation_reports_before_after_improvements(self) -> None:
        before = []
        after = []
        for hip_x, hip_y, foot_shift, wrist_z in (
            (0.0, 0.0, 0.0, 0.0),
            (1.8, 2.5, 1.4, 2.0),
            (-1.4, -2.0, -1.1, 7.0),
            (1.1, 1.6, 0.8, 1.5),
        ):
            before.append(
                {
                    "mid_hip": np.array([hip_x, hip_y, hip_x * 0.4], dtype=np.float64),
                    "left_ankle": np.array([-5.0 + foot_shift, -30.0, foot_shift], dtype=np.float64),
                    "right_ankle": np.array([5.0, -30.0, 0.0], dtype=np.float64),
                    "left_toes": np.array([-5.0 + foot_shift, -30.0, 8.0 + foot_shift], dtype=np.float64),
                    "right_toes": np.array([5.0, -30.0, 8.0], dtype=np.float64),
                    "left_heel": np.array([-5.0 + foot_shift, -30.0, -4.0 + foot_shift], dtype=np.float64),
                    "right_heel": np.array([5.0, -30.0, -4.0], dtype=np.float64),
                    "left_wrist": np.array([-16.0, 14.0, wrist_z], dtype=np.float64),
                    "right_wrist": np.array([16.0, 14.0, 0.0], dtype=np.float64),
                }
            )
        for hip_x, hip_y, foot_shift, wrist_z in (
            (0.2, 0.3, 0.1, 0.5),
            (0.5, 0.4, 0.2, 1.8),
            (0.1, 0.2, 0.0, 5.8),
            (0.3, 0.3, 0.1, 1.0),
        ):
            after.append(
                {
                    "mid_hip": np.array([hip_x, hip_y, hip_x * 0.2], dtype=np.float64),
                    "left_ankle": np.array([-5.0 + foot_shift, -30.0, foot_shift], dtype=np.float64),
                    "right_ankle": np.array([5.0, -30.0, 0.0], dtype=np.float64),
                    "left_toes": np.array([-5.0 + foot_shift, -30.0, 8.0 + foot_shift], dtype=np.float64),
                    "right_toes": np.array([5.0, -30.0, 8.0], dtype=np.float64),
                    "left_heel": np.array([-5.0 + foot_shift, -30.0, -4.0 + foot_shift], dtype=np.float64),
                    "right_heel": np.array([5.0, -30.0, -4.0], dtype=np.float64),
                    "left_wrist": np.array([-16.0, 14.0, wrist_z], dtype=np.float64),
                    "right_wrist": np.array([16.0, 14.0, 0.0], dtype=np.float64),
                }
            )

        evaluation = pipeline._build_cleanup_evaluation(before, after)
        self.assertLess(
            evaluation["root_position_jitter"]["after"],
            evaluation["root_position_jitter"]["before"],
        )
        self.assertLess(
            evaluation["left_foot_contact_spread"]["after"],
            evaluation["left_foot_contact_spread"]["before"],
        )
        self.assertGreater(
            evaluation["left_wrist_motion_energy"]["after"],
            0.0,
        )
        self.assertGreater(
            evaluation["root_position_jitter"]["improvement_ratio"],
            0.0,
        )

    def test_convert_video_to_bvh_reports_source_stages_in_diagnostics(self) -> None:
        base = make_pose_points()
        frames = []
        for dx in (0.0, 0.3, -0.2, 0.1):
            frame = {key: np.array(value, dtype=np.float64) + np.array([dx, 0.0, 0.0], dtype=np.float64) for key, value in base.items()}
            frames.append(frame)

        scan_stats = {
            "detected": len(frames),
            "roi_used": 0,
            "roi_fallback": 0,
            "roi_resets": 0,
            "pose_backend": "tasks",
        }

        with patch(
            "vid2model_lib.pipeline.collect_detected_pose_samples",
            return_value=(30.0, frames, frames[:2], scan_stats),
        ):
            *_unused, diagnostics = pipeline.convert_video_to_bvh(
                input_path=Path("synthetic.mp4"),
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                include_source_stage_diagnostics=True,
            )

        source_stages = diagnostics.get("source_stages")
        self.assertIsInstance(source_stages, dict)
        assert isinstance(source_stages, dict)
        self.assertIn("pose", source_stages)
        self.assertIn("motion", source_stages)

        pose = source_stages["pose"]
        self.assertIn("stages", pose)
        self.assertIn("comparisons", pose)
        self.assertIn("flags", pose)
        self.assertIn("filled_source", pose["stages"])
        self.assertIn("final_source", pose["stages"])
        self.assertIn("source_pipeline_risk", pose["flags"])
        self.assertIn("suspected_issue_stage", pose["flags"])

        motion = source_stages["motion"]
        self.assertIn("stages", motion)
        self.assertIn("comparisons", motion)
        self.assertIn("flags", motion)
        self.assertIn("pre_root_yaw", motion["stages"])
        self.assertIn("post_root_yaw", motion["stages"])
        self.assertIn("final_motion", motion["stages"])
        self.assertIn("source_motion_risk", motion["flags"])
        self.assertIn("suspected_issue_stage", motion["flags"])

    def test_source_motion_stage_diagnostics_flags_finalize_spikes(self) -> None:
        pre_yaw = [
            [0.0, 0.0, 0.0, 2.0, 1.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.2, 1.1, 6.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.4, 1.2, 7.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.5, 1.3, 8.0, 0.0, 0.0],
        ]
        post_yaw = [
            [0.0, 0.0, 0.0, 2.1, 1.1, 5.2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.3, 1.1, 6.1, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.6, 1.2, 7.2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.7, 1.3, 8.0, 0.0, 0.0],
        ]
        final = [
            [0.0, 0.0, 0.0, 3.0, 1.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 170.0, 1.1, 6.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 20.0, 1.2, 7.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 190.0, 1.3, 8.0, 0.0, 0.0],
        ]

        diagnostics = pipeline._build_source_motion_stage_diagnostics(pre_yaw, post_yaw, final)

        self.assertTrue(diagnostics["flags"]["rotation_jump_increase_after_finalize"])
        self.assertTrue(diagnostics["flags"]["source_motion_risk"])
        self.assertEqual(diagnostics["flags"]["suspected_issue_stage"], "motion_finalize")

    def test_source_motion_stage_diagnostics_ignores_wraparound_steps(self) -> None:
        pre_yaw = [
            [0.0, 0.0, 0.0, 10.0, 1.0, 170.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 11.0, 1.1, 175.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 12.0, 1.2, 179.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 13.0, 1.3, -179.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 14.0, 1.4, -174.0, 0.0, 0.0],
        ]
        post_yaw = [
            [0.0, 0.0, 0.0, 10.5, 1.0, 171.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 11.5, 1.1, 176.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 12.5, 1.2, -179.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 13.5, 1.3, -176.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 14.5, 1.4, -170.0, 0.0, 0.0],
        ]
        final = [list(row) for row in post_yaw]

        diagnostics = pipeline._build_source_motion_stage_diagnostics(pre_yaw, post_yaw, final)

        self.assertFalse(diagnostics["flags"]["yaw_normalization_spike"])
        self.assertFalse(diagnostics["flags"]["rotation_jump_increase_after_finalize"])
        self.assertFalse(diagnostics["flags"]["final_rotation_jump_risk"])
        self.assertFalse(diagnostics["flags"]["source_motion_risk"])

    def test_convert_video_to_bvh_skips_source_stages_by_default(self) -> None:
        base = make_pose_points()
        frames = [
            {key: np.array(value, dtype=np.float64) for key, value in base.items()}
            for _ in range(4)
        ]
        scan_stats = {
            "detected": len(frames),
            "roi_used": 0,
            "roi_fallback": 0,
            "roi_resets": 0,
            "pose_backend": "tasks",
        }

        with patch(
            "vid2model_lib.pipeline.collect_detected_pose_samples",
            return_value=(30.0, frames, frames[:2], scan_stats),
        ):
            *_unused, diagnostics = pipeline.convert_video_to_bvh(
                input_path=Path("synthetic.mp4"),
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

        self.assertNotIn("source_stages", diagnostics)


class PipelineMirrorModuleTests(unittest.TestCase):
    def test_mirror_helpers_detect_and_flip_mirrored_pose(self) -> None:
        sample = make_pose_points()
        mirrored_input = {
            "left_shoulder": np.array([10.0, 20.0, 0.0], dtype=np.float64),
            "right_shoulder": np.array([-10.0, 20.0, 0.0], dtype=np.float64),
            "left_hip": np.array([5.0, 0.0, 0.0], dtype=np.float64),
            "right_hip": np.array([-5.0, 0.0, 0.0], dtype=np.float64),
            "left_wrist": np.array([18.0, 12.0, 0.0], dtype=np.float64),
            "right_wrist": np.array([-18.0, 12.0, 0.0], dtype=np.float64),
            **{k: v for k, v in sample.items() if k not in {"left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_wrist", "right_wrist"}},
        }

        self.assertTrue(looks_mirrored(mirrored_input))
        corrected = mirror_pose_points(mirrored_input)
        self.assertAlmostEqual(float(corrected["left_shoulder"][0]), 10.0, places=6)
        self.assertAlmostEqual(float(corrected["right_shoulder"][0]), -10.0, places=6)


class PipelineChannelsModuleTests(unittest.TestCase):
    def test_frame_channels_rest_pose_is_near_zero(self) -> None:
        pts = make_pose_points()
        rest_offsets = build_rest_offsets([pts])

        channels = module_frame_channels(pts, rest_offsets, pts["mid_hip"])

        self.assertTrue(np.allclose(channels, np.zeros(len(channels)), atol=1e-6))

    def test_frame_channels_rotated_pose_reports_root_yaw(self) -> None:
        pts = make_pose_points()
        rotated = rotate_pose_y(pts, 90.0)
        rest_offsets = build_rest_offsets([pts])

        channels = module_frame_channels(rotated, rest_offsets, rotated["mid_hip"])

        self.assertAlmostEqual(abs(float(channels[5])), 90.0, places=4)


if __name__ == "__main__":
    unittest.main()
