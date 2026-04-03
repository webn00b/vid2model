from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .math3d import normalize
from .pipeline_auto_pose import PoseCorrectionProfile


def median_pose_sample(samples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if not samples:
        raise RuntimeError("No valid pose frames detected in video.")

    sample_keys = list(samples[0].keys())
    median_sample: Dict[str, np.ndarray] = {}
    for key in sample_keys:
        stacked = np.stack([sample[key] for sample in samples], axis=0)
        median_sample[key] = np.median(stacked, axis=0)
    return median_sample


def build_reference_basis(
    sample: Dict[str, np.ndarray],
    corrections: Optional[PoseCorrectionProfile] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    origin = np.array(sample["mid_hip"], dtype=np.float64)
    left_shoulder = np.array(sample["left_shoulder"], dtype=np.float64)
    right_shoulder = np.array(sample["right_shoulder"], dtype=np.float64)
    left_hip = np.array(sample["left_hip"], dtype=np.float64)
    right_hip = np.array(sample["right_hip"], dtype=np.float64)
    chest = np.array(sample["chest"], dtype=np.float64)

    use_shoulders = True if corrections is None else corrections.shoulder_tracking
    x_axis = normalize(right_shoulder - left_shoulder) if use_shoulders else np.array([0.0, 0.0, 0.0], dtype=np.float64)
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = normalize(right_hip - left_hip)
    if np.linalg.norm(x_axis) < 1e-8 and use_shoulders:
        x_axis = normalize(right_shoulder - left_shoulder)
    if np.linalg.norm(x_axis) < 1e-8:
        return np.eye(3), origin

    up_hint = chest - origin
    if np.linalg.norm(up_hint) < 1e-8:
        up_hint = ((left_shoulder + right_shoulder) * 0.5) - origin
    y_axis = normalize(up_hint)
    if np.linalg.norm(y_axis) < 1e-8:
        return np.eye(3), origin

    z_axis = np.cross(x_axis, y_axis)
    if np.linalg.norm(z_axis) < 1e-8:
        return np.eye(3), origin
    z_axis = normalize(z_axis)

    nose = sample.get("nose")
    if nose is not None:
        nose_vec = np.array(nose, dtype=np.float64) - origin
        if float(np.dot(nose_vec, z_axis)) < 0.0:
            z_axis = -z_axis

    y_axis = np.cross(z_axis, x_axis)
    if np.linalg.norm(y_axis) < 1e-8:
        return np.eye(3), origin
    y_axis = normalize(y_axis)
    if float(np.dot(y_axis, up_hint)) < 0.0:
        y_axis = -y_axis
        z_axis = -z_axis

    basis = np.column_stack((x_axis, y_axis, z_axis))
    return basis, origin


def rotate_points_about_y(
    pts: Dict[str, np.ndarray],
    keys: List[str],
    angle_deg: float,
    pivot: np.ndarray,
) -> None:
    if not keys or abs(float(angle_deg)) < 1e-8:
        return
    angle_rad = np.radians(float(angle_deg))
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
    for key in keys:
        value = pts.get(key)
        if value is None:
            continue
        pts[key] = pivot + rot @ (value - pivot)


def apply_pose_corrections(
    pts: Dict[str, np.ndarray],
    corrections: PoseCorrectionProfile,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {key: np.array(value, dtype=np.float64) for key, value in pts.items()}
    if not out:
        return out

    mid_hip = out.get("mid_hip")
    if mid_hip is None:
        mid_hip = (out["left_hip"] + out["right_hip"]) * 0.5
        out["mid_hip"] = mid_hip

    shoulder_span = float(np.linalg.norm(out["left_shoulder"] - out["right_shoulder"]))
    if shoulder_span < 1e-6:
        shoulder_span = float(np.linalg.norm(out["left_hip"] - out["right_hip"]))
    if shoulder_span < 1e-6:
        shoulder_span = 1.0

    hip_span = float(np.linalg.norm(out["left_hip"] - out["right_hip"]))
    if hip_span < 1e-6:
        hip_span = shoulder_span

    if corrections.auto_grounding:
        ground_keys = [
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_toes",
            "right_toes",
        ]
        ground_values = [out[key][1] for key in ground_keys if key in out and np.isfinite(out[key][1])]
        if ground_values:
            ground_y = float(min(ground_values))
            if np.isfinite(ground_y):
                offset = np.array([0.0, -ground_y, 0.0], dtype=np.float64)
                for key in out:
                    out[key] = out[key] + offset
                mid_hip = out["mid_hip"]

    if corrections.hip_y_position_offset_percent:
        dy = shoulder_span * 0.06 * (float(corrections.hip_y_position_offset_percent) / 100.0)
        for key in out:
            out[key][1] += dy
        mid_hip = out["mid_hip"]

    if corrections.hip_z_position_offset_percent:
        dz = shoulder_span * 0.06 * (float(corrections.hip_z_position_offset_percent) / 100.0)
        for key in out:
            out[key][2] += dz
        mid_hip = out["mid_hip"]

    if corrections.hip_depth_scale_percent and abs(float(corrections.hip_depth_scale_percent) - 100.0) > 1e-6:
        scale_z = float(corrections.hip_depth_scale_percent) / 100.0
        lower_body_keys = {
            "mid_hip",
            "hips",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_toes",
            "right_toes",
        }
        for key in lower_body_keys:
            if key not in out:
                continue
            vec = out[key] - mid_hip
            vec[2] *= scale_z
            out[key] = mid_hip + vec

    torso_keys = ["spine", "chest", "upper_chest", "neck", "head", "left_shoulder_clavicle", "right_shoulder_clavicle"]
    if corrections.body_bend_reduction_power:
        bend = float(np.clip(corrections.body_bend_reduction_power, 0.0, 1.0))
        if bend > 1e-6:
            for key in torso_keys:
                if key not in out:
                    continue
                vec = out[key] - mid_hip
                vec[2] *= (1.0 - bend)
                out[key] = mid_hip + vec

    upper_keys = [
        "spine",
        "chest",
        "upper_chest",
        "neck",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_shoulder_clavicle",
        "right_shoulder_clavicle",
        "left_upper_arm",
        "right_upper_arm",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hand",
        "right_hand",
        "left_thumb_metacarpal",
        "left_thumb_proximal",
        "left_thumb_distal",
        "right_thumb_metacarpal",
        "right_thumb_proximal",
        "right_thumb_distal",
        "left_index_proximal",
        "left_index_intermediate",
        "left_index_distal",
        "right_index_proximal",
        "right_index_intermediate",
        "right_index_distal",
        "left_middle_proximal",
        "left_middle_intermediate",
        "left_middle_distal",
        "right_middle_proximal",
        "right_middle_intermediate",
        "right_middle_distal",
        "left_ring_proximal",
        "left_ring_intermediate",
        "left_ring_distal",
        "right_ring_proximal",
        "right_ring_intermediate",
        "right_ring_distal",
        "left_little_proximal",
        "left_little_intermediate",
        "left_little_distal",
        "right_little_proximal",
        "right_little_intermediate",
        "right_little_distal",
    ]
    if corrections.upper_rotation_offset_deg:
        rotate_points_about_y(out, upper_keys, corrections.upper_rotation_offset_deg, mid_hip)

    arm_dx = shoulder_span * 0.08 * (float(corrections.arm_horizontal_offset_percent) / 100.0)
    arm_dy = shoulder_span * 0.05 * (float(corrections.arm_vertical_offset_percent) / 100.0)
    if abs(arm_dx) > 1e-8 or abs(arm_dy) > 1e-8:
        arm_keys = [
            "left_shoulder",
            "left_shoulder_clavicle",
            "left_upper_arm",
            "left_elbow",
            "left_wrist",
            "left_hand",
            "left_thumb_metacarpal",
            "left_thumb_proximal",
            "left_thumb_distal",
            "left_index_proximal",
            "left_index_intermediate",
            "left_index_distal",
            "left_middle_proximal",
            "left_middle_intermediate",
            "left_middle_distal",
            "left_ring_proximal",
            "left_ring_intermediate",
            "left_ring_distal",
            "left_little_proximal",
            "left_little_intermediate",
            "left_little_distal",
        ]
        for key in arm_keys:
            if key in out:
                out[key][0] -= arm_dx
                out[key][1] += arm_dy
        for key in [k.replace("left_", "right_") for k in arm_keys]:
            if key in out:
                out[key][0] += arm_dx
                out[key][1] += arm_dy

    if corrections.body_collider_mode > 0:
        body_height = float(np.linalg.norm(out["head"] - mid_hip))
        if body_height < 1e-6:
            body_height = shoulder_span * 2.1
        if body_height < 1e-6:
            body_height = 1.0

        zone_defs = [
            (
                "head",
                corrections.body_collider_head_size_percent,
                0.76,
                [
                    "head",
                    "nose",
                    "neck",
                    "upper_chest",
                ],
            ),
            (
                "chest",
                corrections.body_collider_chest_size_percent,
                0.50,
                [
                    "chest",
                    "spine",
                    "left_shoulder_clavicle",
                    "right_shoulder_clavicle",
                    "left_shoulder",
                    "right_shoulder",
                    "left_upper_arm",
                    "right_upper_arm",
                    "left_elbow",
                    "right_elbow",
                    "left_wrist",
                    "right_wrist",
                    "left_hand",
                    "right_hand",
                ],
            ),
            (
                "waist",
                corrections.body_collider_waist_size_percent,
                0.24,
                [
                    "mid_hip",
                    "hips",
                    "left_hip",
                    "right_hip",
                    "left_upper_leg",
                    "right_upper_leg",
                ],
            ),
            (
                "hip",
                corrections.body_collider_hip_size_percent,
                0.00,
                [
                    "left_lower_leg",
                    "right_lower_leg",
                    "left_ankle",
                    "right_ankle",
                    "left_heel",
                    "right_heel",
                    "left_toes",
                    "right_toes",
                ],
            ),
        ]
        for zone_name, size_percent, min_height, zone_keys in zone_defs:
            if corrections.body_collider_mode == 1 and zone_name == "hip":
                continue
            if size_percent is None:
                continue
            size_scale = max(0.25, float(size_percent) / 100.0)
            zone_lateral_base = shoulder_span * (0.12 if zone_name == "head" else 0.24 if zone_name == "chest" else 0.20 if zone_name == "waist" else 0.17)
            zone_depth_base = shoulder_span * (0.10 if zone_name == "head" else 0.22 if zone_name == "chest" else 0.18 if zone_name == "waist" else 0.15)
            zone_height_base = min_height * body_height
            for key in zone_keys:
                if key not in out:
                    continue
                point = out[key]
                rel = point - mid_hip
                height = rel[1]
                if height < zone_height_base - body_height * 0.04:
                    continue
                target_lateral = zone_lateral_base * size_scale
                current_lateral = abs(rel[0])
                if current_lateral < target_lateral:
                    sign = -1.0 if rel[0] < 0.0 else 1.0
                    blend = 0.60
                    point[0] = point[0] * (1.0 - blend) + (mid_hip[0] + sign * target_lateral) * blend
                target_depth = zone_depth_base * size_scale
                current_depth = abs(rel[2])
                if current_depth < target_depth:
                    sign = -1.0 if rel[2] < 0.0 else 1.0
                    blend = 0.45
                    point[2] = point[2] * (1.0 - blend) + (mid_hip[2] + sign * target_depth) * blend

    if corrections.use_arm_ik:
        arm_floor = shoulder_span * 0.16
        forearm_floor = shoulder_span * 0.24
        for side, sign in (("left", -1.0), ("right", 1.0)):
            for key, floor in [
                (f"{side}_shoulder_clavicle", arm_floor),
                (f"{side}_shoulder", arm_floor * 1.05),
                (f"{side}_elbow", forearm_floor),
                (f"{side}_wrist", forearm_floor * 1.15),
                (f"{side}_hand", forearm_floor * 1.2),
            ]:
                point = out.get(key)
                if point is None:
                    continue
                rel_x = (point[0] - mid_hip[0]) * sign
                if rel_x < floor:
                    point[0] = mid_hip[0] + sign * floor

    if corrections.use_leg_ik:
        leg_floor = hip_span * 0.16
        lower_leg_floor = hip_span * 0.12
        for side, sign in (("left", -1.0), ("right", 1.0)):
            for key, floor in [
                (f"{side}_hip", leg_floor),
                (f"{side}_knee", lower_leg_floor),
                (f"{side}_ankle", lower_leg_floor * 0.95),
                (f"{side}_heel", lower_leg_floor * 0.9),
                (f"{side}_toes", lower_leg_floor * 0.9),
            ]:
                point = out.get(key)
                if point is None:
                    continue
                rel_x = (point[0] - mid_hip[0]) * sign
                if rel_x < floor:
                    point[0] = mid_hip[0] + sign * floor

    if corrections.shoulder_tracking and "left_shoulder_clavicle" in out and "right_shoulder_clavicle" in out:
        span = out["right_shoulder_clavicle"] - out["left_shoulder_clavicle"]
        if np.linalg.norm(span) > 1e-6:
            out["chest"] = (out["left_shoulder_clavicle"] + out["right_shoulder_clavicle"]) * 0.5
            out["spine"] = out["mid_hip"] + (out["chest"] - out["mid_hip"]) * 0.5
            out["upper_chest"] = out["chest"] + (out["neck"] - out["chest"]) * 0.5 if "neck" in out else out["chest"]

    return out


def canonicalize_pose_points(
    pts: Dict[str, np.ndarray],
    basis: np.ndarray,
    origin: np.ndarray,
) -> Dict[str, np.ndarray]:
    inv_basis = basis.T
    out: Dict[str, np.ndarray] = {}
    for key, value in pts.items():
        out[key] = inv_basis @ (np.array(value, dtype=np.float64) - origin)
    return out
