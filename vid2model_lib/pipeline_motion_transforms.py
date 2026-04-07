from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .skeleton import JOINTS


def _wrap_angle_deg(angle: float) -> float:
    wrapped = (float(angle) + 180.0) % 360.0 - 180.0
    if wrapped == -180.0:
        return 180.0
    return wrapped


def normalize_motion_root_yaw(
    motion_values: List[List[float]],
) -> Tuple[List[List[float]], Dict[str, float]]:
    if not motion_values:
        return motion_values, {"applied": 0.0, "offset_deg": 0.0, "near0": 0.0, "near180": 0.0}

    yaw_values = [float(frame[5]) for frame in motion_values if len(frame) > 5 and np.isfinite(frame[5])]
    if not yaw_values:
        return motion_values, {"applied": 0.0, "offset_deg": 0.0, "near0": 0.0, "near180": 0.0}

    near0 = 0
    near180 = 0
    for yaw in yaw_values:
        wrapped = _wrap_angle_deg(yaw)
        if abs(wrapped) < 45.0:
            near0 += 1
        elif abs(abs(wrapped) - 180.0) < 45.0:
            near180 += 1

    if near180 <= near0 or near180 < max(8, int(len(yaw_values) * 0.35)):
        return motion_values, {
            "applied": 0.0,
            "offset_deg": 0.0,
            "near0": float(near0),
            "near180": float(near180),
        }

    normalized: List[List[float]] = []
    for frame in motion_values:
        updated = list(frame)
        if len(updated) > 5 and np.isfinite(updated[5]):
            updated[5] = _wrap_angle_deg(updated[5] - 180.0)
        normalized.append(updated)
    return normalized, {
        "applied": 1.0,
        "offset_deg": 180.0,
        "near0": float(near0),
        "near180": float(near180),
    }


def apply_manual_root_yaw_offset(
    motion_values: List[List[float]],
    offset_deg: float,
) -> List[List[float]]:
    if not motion_values or not np.isfinite(offset_deg) or abs(float(offset_deg)) < 1e-6:
        return motion_values
    adjusted: List[List[float]] = []
    for frame in motion_values:
        updated = list(frame)
        if len(updated) > 5 and np.isfinite(updated[5]):
            updated[5] = _wrap_angle_deg(updated[5] + float(offset_deg))
        adjusted.append(updated)
    return adjusted


def unwrap_motion_rotation_channels(
    motion_values: List[List[float]],
) -> Tuple[List[List[float]], Dict[str, float]]:
    if not motion_values:
        return motion_values, {
            "applied": 0.0,
            "changed_values": 0.0,
            "max_step_before": 0.0,
            "max_step_after": 0.0,
        }

    adjusted: List[List[float]] = [list(frame) for frame in motion_values]
    rotation_indices: List[int] = []
    channel_index = 0
    for joint in JOINTS:
        if joint.channels == 6:
            rotation_indices.extend([channel_index + 3, channel_index + 4, channel_index + 5])
            channel_index += 6
        elif joint.channels == 3:
            rotation_indices.extend([channel_index, channel_index + 1, channel_index + 2])
            channel_index += 3

    changed_values = 0
    max_step_before = 0.0
    max_step_after = 0.0
    for channel_idx in rotation_indices:
        prev_value = None
        for frame_idx, frame in enumerate(adjusted):
            if channel_idx >= len(frame):
                break
            value = frame[channel_idx]
            if not np.isfinite(value):
                continue
            if prev_value is None:
                prev_value = float(value)
                continue

            value_before = float(value)
            step_before = abs(value_before - prev_value)
            max_step_before = max(max_step_before, step_before)

            value_after = value_before
            while value_after - prev_value > 180.0:
                value_after -= 360.0
            while value_after - prev_value < -180.0:
                value_after += 360.0

            step_after = abs(value_after - prev_value)
            max_step_after = max(max_step_after, step_after)
            if abs(value_after - value_before) > 1e-6:
                adjusted[frame_idx][channel_idx] = value_after
                changed_values += 1
            prev_value = float(adjusted[frame_idx][channel_idx])

    return adjusted, {
        "applied": 1.0 if changed_values > 0 else 0.0,
        "changed_values": float(changed_values),
        "max_step_before": float(max_step_before),
        "max_step_after": float(max_step_after),
    }


def apply_lower_body_rotation_mode(
    motion_values: List[List[float]],
    mode: str,
) -> List[List[float]]:
    normalized_mode = str(mode or "off").strip().lower()
    if normalized_mode == "off" or not motion_values:
        return motion_values

    joint_names = [joint.name for joint in JOINTS]
    target_joints = {"leftUpperLeg", "leftLowerLeg", "leftFoot", "rightUpperLeg", "rightLowerLeg", "rightFoot"}
    target_indices = {name for name in target_joints if name in joint_names}
    adjusted: List[List[float]] = []

    for frame in motion_values:
        updated = list(frame)
        for joint_index, joint_name in enumerate(joint_names[1:], start=1):
            if joint_name not in target_indices:
                continue
            base = 6 + (joint_index - 1) * 3
            if base + 2 >= len(updated):
                continue
            rz, rx, ry = updated[base], updated[base + 1], updated[base + 2]
            if normalized_mode == "invert":
                updated[base] = _wrap_angle_deg(-float(rz))
                updated[base + 1] = _wrap_angle_deg(-float(rx))
                updated[base + 2] = _wrap_angle_deg(-float(ry))
            elif normalized_mode == "yaw180":
                updated[base + 2] = _wrap_angle_deg(float(ry) + 180.0)
        adjusted.append(updated)
    return adjusted


def apply_upper_body_rotation_scale(
    motion_values: List[List[float]],
    scale: float,
    arm_scale: Optional[float] = None,
) -> List[List[float]]:
    if not motion_values:
        return motion_values
    if not np.isfinite(scale):
        return motion_values
    scale = float(scale)
    arm_scale = scale if arm_scale is None or not np.isfinite(arm_scale) else float(arm_scale)
    if abs(scale - 1.0) < 1e-6 and abs(arm_scale - 1.0) < 1e-6:
        return motion_values

    joint_names = [joint.name for joint in JOINTS]
    torso_joints = {
        "spine",
        "chest",
        "upperChest",
        "neck",
        "head",
    }
    arm_joints = {
        "leftShoulder",
        "rightShoulder",
        "leftUpperArm",
        "rightUpperArm",
        "leftLowerArm",
        "rightLowerArm",
        "leftHand",
        "rightHand",
        "leftThumbMetacarpal",
        "leftThumbProximal",
        "leftThumbDistal",
        "leftIndexProximal",
        "leftIndexIntermediate",
        "leftIndexDistal",
        "leftMiddleProximal",
        "leftMiddleIntermediate",
        "leftMiddleDistal",
        "leftRingProximal",
        "leftRingIntermediate",
        "leftRingDistal",
        "leftLittleProximal",
        "leftLittleIntermediate",
        "leftLittleDistal",
        "rightThumbMetacarpal",
        "rightThumbProximal",
        "rightThumbDistal",
        "rightIndexProximal",
        "rightIndexIntermediate",
        "rightIndexDistal",
        "rightMiddleProximal",
        "rightMiddleIntermediate",
        "rightMiddleDistal",
        "rightRingProximal",
        "rightRingIntermediate",
        "rightRingDistal",
        "rightLittleProximal",
        "rightLittleIntermediate",
        "rightLittleDistal",
    }
    adjusted: List[List[float]] = []
    for frame in motion_values:
        updated = list(frame)
        for joint_index, joint_name in enumerate(joint_names[1:], start=1):
            if joint_name in torso_joints:
                joint_scale = scale
            elif joint_name in arm_joints:
                joint_scale = arm_scale
            else:
                continue
            base = 6 + (joint_index - 1) * 3
            if base + 2 >= len(updated):
                continue
            updated[base] = float(updated[base]) * joint_scale
            updated[base + 1] = float(updated[base + 1]) * joint_scale
            # Spine local-Y encodes the canonical rest orientation (~±180°) that results
            # from normalize_motion_root_yaw flipping hips. Scaling this value would shift
            # the apparent facing direction of the source skeleton and break viewer retarget
            # facing detection, which relies on spine Y staying near ±180°. Only Z and X
            # channels of spine carry real motion (forward lean / side tilt) and are scaled.
            if joint_name != "spine":
                updated[base + 2] = float(updated[base + 2]) * joint_scale
        adjusted.append(updated)
    return adjusted
