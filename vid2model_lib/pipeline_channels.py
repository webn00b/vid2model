from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .math3d import euler_zxy_from_matrix, normalize, rotation_align, rotation_align_with_secondary
from .pipeline_auto_pose import PoseCorrectionProfile
from .skeleton import CHILDREN, JOINTS, MAP_TO_POINTS


def _body_forward_vector(pts: Dict[str, np.ndarray]) -> np.ndarray:
    mid_hip = pts.get("mid_hip", np.zeros(3, dtype=np.float64))
    chest = pts.get("chest", mid_hip + np.array([0.0, 1.0, 0.0], dtype=np.float64))
    nose = pts.get("nose", chest)
    shoulder_side = pts.get("right_shoulder", chest) - pts.get("left_shoulder", chest)
    hip_side = pts.get("right_hip", mid_hip) - pts.get("left_hip", mid_hip)
    side = shoulder_side if np.linalg.norm(shoulder_side) > 1e-6 else hip_side
    if np.linalg.norm(side) < 1e-6:
        side = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    up = chest - mid_hip
    if np.linalg.norm(up) < 1e-6:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    forward = np.cross(normalize(side), normalize(up))
    if np.linalg.norm(forward) < 1e-6:
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    nose_dir = nose - mid_hip
    if np.linalg.norm(nose_dir) > 1e-6 and float(np.dot(forward, nose_dir)) < 0.0:
        forward = -forward
    return normalize(forward)


def _secondary_vectors_for_joint(
    name: str,
    pts: Dict[str, np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if name == "hips":
        chest = pts.get("chest")
        mid_hip = pts.get("mid_hip")
        if chest is not None and mid_hip is not None:
            return chest - mid_hip, np.array([0.0, 1.0, 0.0], dtype=np.float64)
        return None, None

    if name in {"leftUpperLeg", "rightUpperLeg", "leftLowerLeg", "rightLowerLeg", "leftFoot", "rightFoot"}:
        body_forward = _body_forward_vector(pts)
        if np.linalg.norm(body_forward) > 1e-6:
            return body_forward, np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return None, None


def _root_joint_rotation(
    pts: Dict[str, np.ndarray],
    corrections: Optional[PoseCorrectionProfile],
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    left_hip = pts["left_hip"]
    right_hip = pts["right_hip"]
    left_shoulder = pts["left_shoulder"]
    right_shoulder = pts["right_shoulder"]
    mid_hip = pts["mid_hip"]
    chest = pts["chest"]
    nose = pts.get("nose", chest)

    hip_weight = 0.82 if corrections and corrections.hip_camera else 0.58
    shoulder_weight = 1.0 - hip_weight if (corrections is None or corrections.shoulder_tracking) else 0.0
    if shoulder_weight <= 0.0:
        side_vec = right_hip - left_hip
    else:
        side_vec = hip_weight * (right_hip - left_hip) + shoulder_weight * (right_shoulder - left_shoulder)
    side_xz = normalize(np.array([side_vec[0], 0.0, side_vec[2]], dtype=np.float64))

    up_vec = chest - mid_hip
    if np.linalg.norm(up_vec) < 1e-8:
        up_vec = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    up_vec = normalize(up_vec)

    forward_vec = np.cross(side_xz, up_vec)
    forward_xz = np.array([forward_vec[0], 0.0, forward_vec[2]], dtype=np.float64)
    forward_norm = np.linalg.norm(forward_xz)
    if forward_norm < 1e-7:
        yaw_deg = 0.0
    else:
        forward_xz /= forward_norm
        nose_dir = np.array([nose[0] - mid_hip[0], 0.0, nose[2] - mid_hip[2]], dtype=np.float64)
        if np.linalg.norm(nose_dir) > 1e-7 and float(np.dot(forward_xz, nose_dir)) < 0.0:
            forward_xz = -forward_xz
        yaw_deg = float(np.degrees(np.arctan2(forward_xz[0], forward_xz[2])))

    yaw_rad = np.radians(yaw_deg)
    cos_y = float(np.cos(yaw_rad))
    sin_y = float(np.sin(yaw_rad))
    rotation = np.array(
        [
            [cos_y, 0.0, sin_y],
            [0.0, 1.0, 0.0],
            [-sin_y, 0.0, cos_y],
        ],
        dtype=np.float64,
    )
    return rotation, (0.0, 0.0, yaw_deg)


def frame_channels(
    pts: Dict[str, np.ndarray],
    rest_offsets: Dict[str, np.ndarray],
    ref_root: np.ndarray,
    corrections: Optional[PoseCorrectionProfile] = None,
) -> List[float]:
    channels: List[float] = []
    root_name = JOINTS[0].name
    root_point_name = MAP_TO_POINTS[root_name][0]
    root_pos = pts[root_point_name] - ref_root
    global_rot: Dict[str, np.ndarray] = {root_name: np.eye(3)}

    def solve_joint(name: str) -> Tuple[float, float, float]:
        if name == root_name:
            root_rotation, root_angles = _root_joint_rotation(pts, corrections)
            global_rot[name] = root_rotation
            return root_angles

        children = CHILDREN.get(name, [])
        if not children:
            return (0.0, 0.0, 0.0)

        child = children[0]
        parent_point_name = MAP_TO_POINTS[name][0]
        child_point_name = MAP_TO_POINTS[child][0]

        cur_vec = pts[child_point_name] - pts[parent_point_name]
        rest_vec = rest_offsets[child]

        if np.linalg.norm(cur_vec) < 1e-7 or np.linalg.norm(rest_vec) < 1e-7:
            return (0.0, 0.0, 0.0)

        parent = next(j for j in JOINTS if j.name == name).parent
        parent_global = np.eye(3) if parent is None else global_rot[parent]

        cur_secondary, rest_secondary = _secondary_vectors_for_joint(name, pts)
        if cur_secondary is not None and rest_secondary is not None:
            r_align = rotation_align_with_secondary(rest_vec, cur_vec, rest_secondary, cur_secondary)
        else:
            r_align = rotation_align(rest_vec, cur_vec)
        r_local = parent_global.T @ r_align

        global_rot[name] = parent_global @ r_local
        return euler_zxy_from_matrix(r_local)

    # Compute hips height above ground from leg chain rest offsets.
    # rest_offsets store leg vectors as (0, -length, 0), so summing their Y
    # gives the foot Y relative to hips (negative). Negating gives hip height.
    leg_chain_y = (
        float(rest_offsets.get("leftUpperLeg", np.zeros(3))[1])
        + float(rest_offsets.get("leftLowerLeg", np.zeros(3))[1])
        + float(rest_offsets.get("leftFoot", np.zeros(3))[1])
    )
    hip_height = -leg_chain_y  # positive: how high hips sit above foot level

    rz, rx, ry = solve_joint(root_name)
    channels.extend([float(root_pos[0]), float(root_pos[1]) + hip_height, float(root_pos[2]), rz, rx, ry])

    for joint in JOINTS[1:]:
        rz, rx, ry = solve_joint(joint.name)
        channels.extend([rz, rx, ry])

    return channels
