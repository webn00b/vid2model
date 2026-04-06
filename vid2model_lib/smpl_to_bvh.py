"""Convert SMPL body parameters (axis-angle rotations) to BVH motion channels.

SMPL uses 24 joints with axis-angle rotations (72 values per frame).
This module maps them to the project's VRM-compatible BVH skeleton
and converts axis-angle → ZXY euler angles.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from .skeleton import JOINTS


# SMPL joint indices → project BVH joint names.
# SMPL has 24 joints; we map the ones that correspond to our skeleton.
SMPL_JOINT_TO_BVH: Dict[int, str] = {
    0: "hips",
    1: "leftUpperLeg",
    2: "rightUpperLeg",
    3: "spine",
    4: "leftLowerLeg",
    5: "rightLowerLeg",
    6: "chest",
    7: "leftFoot",
    8: "rightFoot",
    9: "upperChest",
    10: "leftToes",
    11: "rightToes",
    12: "neck",
    13: "leftShoulder",
    14: "rightShoulder",
    15: "head",
    16: "leftUpperArm",
    17: "rightUpperArm",
    18: "leftLowerArm",
    19: "rightLowerArm",
    20: "leftHand",
    21: "rightHand",
    # 22: left fingers (not mapped individually)
    # 23: right fingers (not mapped individually)
}

# Reverse: BVH joint name → SMPL index
BVH_TO_SMPL_JOINT: Dict[str, int] = {v: k for k, v in SMPL_JOINT_TO_BVH.items()}


def axis_angle_to_euler_zxy(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle (3,) to ZXY euler angles in degrees.

    This matches the BVH channel order used by the project.
    """
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float64)

    rot = Rotation.from_rotvec(axis_angle)
    # BVH uses ZXY intrinsic order
    euler = rot.as_euler("ZXY", degrees=True)
    return euler


def _normalize_foot_level(
    motion_values: List[List[float]],
    rest_offsets: Dict[str, np.ndarray],
) -> List[List[float]]:
    """Normalize Y position so feet are at consistent height across all frames.

    Computes foot level from first frame and offsets all Y positions to place
    feet at Y=0, ensuring consistent animation heights across different videos.

    Args:
        motion_values: Motion channel values (hips has 6 channels: Xpos, Ypos, Zpos, Zrot, Xrot, Yrot)
        rest_offsets: Rest pose bone offsets for skeleton structure

    Returns:
        motion_values with normalized Y positions
    """
    if not motion_values or len(motion_values[0]) < 2:
        return motion_values

    # Get foot bone heights from rest offsets
    leftFoot_offset = rest_offsets.get("leftFoot", np.zeros(3))
    rightFoot_offset = rest_offsets.get("rightFoot", np.zeros(3))
    hips_offset = rest_offsets.get("hips", np.zeros(3))

    # Approximate foot level: hips Y + path to foot (simplified as hips height)
    # In practice, feet are slightly below hips in the kinematic chain
    # We'll use the hips Y position from first frame as reference
    first_frame_hips_y = float(motion_values[0][1])  # Ypos of hips is second value

    # Target feet level at Y=0; hips will be at (first_frame_hips_y - offset)
    target_hips_y = 0.0
    y_offset = target_hips_y - first_frame_hips_y

    # Apply Y offset to all frames' hips Y position (channel index 1)
    result = []
    for frame_channels in motion_values:
        frame_copy = list(frame_channels)
        frame_copy[1] += y_offset  # Adjust hips Ypos
        result.append(frame_copy)

    return result


def smpl_poses_to_bvh_channels(
    smpl_poses: np.ndarray,
    smpl_trans: np.ndarray,
    fps: float = 30.0,
    scale: float = 100.0,
) -> Tuple[List[List[float]], Dict[str, np.ndarray], float]:
    """Convert SMPL parameters to BVH motion channel values.

    Args:
        smpl_poses: (N, 72) or (N, 24, 3) axis-angle rotations per frame.
        smpl_trans: (N, 3) root translation per frame (meters).
        fps: Frame rate.
        scale: Scale factor for translation (SMPL meters → BVH centimeters).

    Returns:
        (motion_values, rest_offsets, fps) matching the project's BVH format.
    """
    n_frames = smpl_poses.shape[0]

    # Reshape poses to (N, 24, 3) if flat
    if smpl_poses.ndim == 2 and smpl_poses.shape[1] == 72:
        smpl_poses = smpl_poses.reshape(n_frames, 24, 3)
    elif smpl_poses.ndim == 2 and smpl_poses.shape[1] != 72:
        raise ValueError(f"Expected 72 pose values per frame, got {smpl_poses.shape[1]}")

    # Build rest offsets from SMPL default skeleton proportions (T-pose).
    # These are approximate and will be overridden if a skeleton profile is used.
    rest_offsets = _build_smpl_rest_offsets(scale)

    # SMPL → BVH coordinate conversion.
    #
    # SMPL camera space: X=right, Y=down, Z=into-screen
    # BVH world space:   X=right, Y=up,   Z=forward
    #
    # HMR2 global_orient includes a ~180° X rotation (person faces camera
    # in SMPL camera coords). We need to undo this so the skeleton stands
    # upright in BVH Y-up space.
    #
    # Root correction: pre-multiply each frame's root rotation by Rx(π) to
    # flip from camera-Y-down to world-Y-up.
    #
    # Body joints: SMPL body pose rotations are parent-relative in the body's
    # anatomical frame. After correcting the root with Rx(π) the kinematic
    # chain propagates correctly — body joint rotations do NOT need a
    # separate coordinate-frame correction.
    _Rx_pi = Rotation.from_euler("X", 180, degrees=True)
    _root_correction = _Rx_pi

    # Center translation and convert coordinate system
    centered_trans = smpl_trans.copy()
    median_trans = np.median(centered_trans, axis=0)
    centered_trans -= median_trans

    # Convert each frame
    motion_values: List[List[float]] = []

    for frame_idx in range(n_frames):
        frame_channels: List[float] = []

        # Root joint (hips): 6 channels — Xpos, Ypos, Zpos, Zrot, Xrot, Yrot
        ct = centered_trans[frame_idx]
        # Transform translation: flip Y, zero out Z (camera depth)
        hip_height = rest_offsets.get("hips", np.zeros(3))[1]
        root_trans = np.array([
            ct[0] * scale,
            -ct[1] * scale + hip_height,
            0.0,
        ])

        # Correct root rotation: undo SMPL camera-space 180° X flip
        smpl_root_rot = Rotation.from_rotvec(smpl_poses[frame_idx, 0])
        bvh_root_rot = _root_correction * smpl_root_rot
        root_rot = bvh_root_rot.as_euler("ZXY", degrees=True)
        frame_channels.extend([
            float(root_trans[0]),
            float(root_trans[1]),
            float(root_trans[2]),
            float(root_rot[0]),  # Z
            float(root_rot[1]),  # X
            float(root_rot[2]),  # Y
        ])

        # Remaining joints: 3 channels each — Zrot, Xrot, Yrot
        for joint in JOINTS[1:]:
            smpl_idx = BVH_TO_SMPL_JOINT.get(joint.name)
            if smpl_idx is not None:
                euler = axis_angle_to_euler_zxy(smpl_poses[frame_idx, smpl_idx])
                frame_channels.extend([
                    float(euler[0]),  # Z
                    float(euler[1]),  # X
                    float(euler[2]),  # Y
                ])
            else:
                # Joint not in SMPL (e.g. fingers) — zero rotation
                frame_channels.extend([0.0, 0.0, 0.0])

        motion_values.append(frame_channels)

    # Normalize Y position: find foot level from first frame and offset all frames
    # so feet rest at Y=0 for consistent animation heights across different videos
    motion_values = _normalize_foot_level(motion_values, rest_offsets)

    return motion_values, rest_offsets, fps


def _build_smpl_rest_offsets(scale: float = 100.0) -> Dict[str, np.ndarray]:
    """Build approximate rest offsets based on SMPL default skeleton.

    Values are in centimeters (scaled from SMPL meters).
    These define the T-pose bone positions relative to parent.
    """
    # Approximate SMPL bone offsets in meters, then scale
    offsets_m: Dict[str, np.ndarray] = {
        "hips": np.array([0.0, 0.95, 0.0]),
        "spine": np.array([0.0, 0.1, 0.0]),
        "chest": np.array([0.0, 0.12, 0.0]),
        "upperChest": np.array([0.0, 0.12, 0.0]),
        "neck": np.array([0.0, 0.1, 0.0]),
        "head": np.array([0.0, 0.08, 0.0]),
        "leftShoulder": np.array([0.04, 0.08, 0.0]),
        "leftUpperArm": np.array([0.12, 0.0, 0.0]),
        "leftLowerArm": np.array([0.25, 0.0, 0.0]),
        "leftHand": np.array([0.22, 0.0, 0.0]),
        "rightShoulder": np.array([-0.04, 0.08, 0.0]),
        "rightUpperArm": np.array([-0.12, 0.0, 0.0]),
        "rightLowerArm": np.array([-0.25, 0.0, 0.0]),
        "rightHand": np.array([-0.22, 0.0, 0.0]),
        "leftUpperLeg": np.array([0.09, -0.05, 0.0]),
        "leftLowerLeg": np.array([0.0, -0.42, 0.0]),
        "leftFoot": np.array([0.0, -0.40, 0.0]),
        "leftToes": np.array([0.0, -0.04, 0.1]),
        "rightUpperLeg": np.array([-0.09, -0.05, 0.0]),
        "rightLowerLeg": np.array([0.0, -0.42, 0.0]),
        "rightFoot": np.array([0.0, -0.40, 0.0]),
        "rightToes": np.array([0.0, -0.04, 0.1]),
    }

    rest_offsets: Dict[str, np.ndarray] = {}
    for joint in JOINTS:
        if joint.name in offsets_m:
            rest_offsets[joint.name] = offsets_m[joint.name] * scale
        else:
            # Finger bones — small default offset
            rest_offsets[joint.name] = np.array([0.02, 0.0, 0.0]) * scale

    return rest_offsets


def load_smpl_output(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load SMPL output from NPZ or PKL file.

    Supports formats from 4D-Humans, WHAM, and smpl2bvh.

    Returns:
        (smpl_poses, smpl_trans) arrays.
    """
    from pathlib import Path as P

    p = P(path)

    if p.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        # Try common key names
        poses = None
        trans = None
        for key in ("smpl_poses", "poses", "body_pose", "pose", "rotations"):
            if key in data:
                poses = data[key]
                break
        for key in ("smpl_trans", "trans", "transl", "translation"):
            if key in data:
                trans = data[key]
                break
        if poses is None:
            raise ValueError(
                f"Could not find pose data in {path}. "
                f"Available keys: {list(data.keys())}"
            )
        if trans is None:
            trans = np.zeros((poses.shape[0], 3), dtype=np.float64)
        return np.array(poses, dtype=np.float64), np.array(trans, dtype=np.float64)

    elif p.suffix == ".pkl":
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            poses = data.get("smpl_poses", data.get("body_pose", data.get("poses")))
            trans = data.get("smpl_trans", data.get("transl", data.get("trans")))
            if poses is None:
                raise ValueError(
                    f"Could not find pose data in {path}. "
                    f"Available keys: {list(data.keys())}"
                )
            if trans is None:
                trans = np.zeros((len(poses), 3), dtype=np.float64)
            return np.array(poses, dtype=np.float64), np.array(trans, dtype=np.float64)
        raise ValueError(f"Unexpected PKL format in {path}")

    else:
        raise ValueError(f"Unsupported file format: {p.suffix} (expected .npz or .pkl)")
