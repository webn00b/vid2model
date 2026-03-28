from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_vision

from .math3d import euler_zxy_from_matrix, rotation_align
from .pose_model import ensure_pose_model
from .pose_points import extract_pose_points
from .skeleton import CHILDREN, JOINTS, MAP_TO_POINTS


def build_rest_offsets(samples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    rest: Dict[str, np.ndarray] = {}
    if not samples:
        raise RuntimeError("No valid pose frames detected in video.")

    all_keys = list(samples[0].keys())
    median_points = {}
    for key in all_keys:
        stacked = np.stack([sample[key] for sample in samples], axis=0)
        median_points[key] = np.median(stacked, axis=0)

    for joint in JOINTS:
        if joint.parent is None:
            rest[joint.name] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            continue

        parent_point = median_points[MAP_TO_POINTS[joint.parent][0]]
        child_point = median_points[MAP_TO_POINTS[joint.name][0]]
        offset = child_point - parent_point
        if np.linalg.norm(offset) < 1e-6:
            offset = np.array([0.0, 5.0, 0.0], dtype=np.float64)
        rest[joint.name] = offset

    return rest


def bvh_hierarchy_lines(rest_offsets: Dict[str, np.ndarray]) -> List[str]:
    lines: List[str] = ["HIERARCHY"]

    def write_joint(name: str, depth: int) -> None:
        indent = "  " * depth
        joint = next(j for j in JOINTS if j.name == name)

        if joint.parent is None:
            lines.append(f"{indent}ROOT {name}")
        else:
            lines.append(f"{indent}JOINT {name}")

        lines.append(f"{indent}{{")
        offset = rest_offsets.get(name, np.zeros(3))
        lines.append(f"{indent}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}")

        if joint.channels == 6:
            lines.append(f"{indent}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
        elif joint.channels == 3:
            lines.append(f"{indent}  CHANNELS 3 Zrotation Xrotation Yrotation")

        children = CHILDREN.get(name, [])
        if not children:
            lines.append(f"{indent}  End Site")
            lines.append(f"{indent}  {{")
            lines.append(f"{indent}    OFFSET 0.000000 3.000000 0.000000")
            lines.append(f"{indent}  }}")
        else:
            for child in children:
                write_joint(child, depth + 1)

        lines.append(f"{indent}}}")

    write_joint("Hips", 0)
    return lines


def frame_channels(
    pts: Dict[str, np.ndarray],
    rest_offsets: Dict[str, np.ndarray],
    ref_root: np.ndarray,
) -> List[float]:
    channels: List[float] = []
    root_pos = pts["mid_hip"] - ref_root
    global_rot: Dict[str, np.ndarray] = {"Hips": np.eye(3)}

    def solve_joint(name: str) -> Tuple[float, float, float]:
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

        r_align = rotation_align(rest_vec, cur_vec)
        r_local = parent_global.T @ r_align

        global_rot[name] = parent_global @ r_local
        return euler_zxy_from_matrix(r_local)

    rz, rx, ry = solve_joint("Hips")
    channels.extend([float(root_pos[0]), float(root_pos[1]), float(root_pos[2]), rz, rx, ry])

    for joint in JOINTS[1:]:
        rz, rx, ry = solve_joint(joint.name)
        channels.extend([rz, rx, ry])

    return channels


def convert_video_to_bvh(
    input_path: Path,
    model_complexity: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
) -> Tuple[float, Dict[str, np.ndarray], List[List[float]], np.ndarray, List[Dict[str, np.ndarray]]]:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-6:
        fps = 30.0

    model_path = ensure_pose_model(model_complexity)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=False,
    )
    pose = mp_vision.PoseLandmarker.create_from_options(options)

    frames_pts: List[Dict[str, np.ndarray]] = []
    warmup_samples: List[Dict[str, np.ndarray]] = []
    prev: Optional[Dict[str, np.ndarray]] = None

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((frame_idx * 1000.0) / fps)
        res = pose.detect_for_video(mp_image, ts_ms)
        frame_idx += 1

        pts = extract_pose_points(res)
        if pts is None:
            if prev is None:
                continue
            pts = prev
        else:
            prev = pts
            if len(warmup_samples) < 60:
                warmup_samples.append(pts)

        frames_pts.append(pts)

    cap.release()
    pose.close()

    if not frames_pts:
        raise RuntimeError("No detectable human pose frames found.")

    rest_offsets = build_rest_offsets(warmup_samples if warmup_samples else frames_pts[:20])
    ref_root = frames_pts[0]["mid_hip"].copy()

    motion_values = []
    for pts in frames_pts:
        motion_values.append(frame_channels(pts, rest_offsets, ref_root))

    return fps, rest_offsets, motion_values, ref_root, frames_pts
