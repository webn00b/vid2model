from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def swap_lr_name(name: str) -> str:
    if name.startswith("left_"):
        return "right_" + name[5:]
    if name.startswith("right_"):
        return "left_" + name[6:]
    return name


def mirror_pose_points(pts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    mirrored: Dict[str, np.ndarray] = {}
    for key, value in pts.items():
        mirrored[swap_lr_name(key)] = np.array([-value[0], value[1], value[2]], dtype=np.float64)
    return mirrored


def swap_pose_sides(pts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    swapped: Dict[str, np.ndarray] = {}
    for key, value in pts.items():
        swapped[swap_lr_name(key)] = np.array(value, dtype=np.float64)
    return swapped


def looks_mirrored(sample: Dict[str, np.ndarray]) -> bool:
    votes = []
    for left_key, right_key in [
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "right_hip"),
        ("left_wrist", "right_wrist"),
    ]:
        left = sample.get(left_key)
        right = sample.get(right_key)
        if left is None or right is None:
            continue
        if not np.isfinite(left[0]) or not np.isfinite(right[0]):
            continue
        votes.append(float(left[0]) > float(right[0]))

    if not votes:
        return False
    return sum(votes) >= (len(votes) / 2.0)


def looks_side_swapped(sample: Dict[str, np.ndarray]) -> bool:
    votes = []
    for left_key, right_key in [
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "right_hip"),
        ("left_elbow", "right_elbow"),
        ("left_wrist", "right_wrist"),
        ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle"),
    ]:
        left = sample.get(left_key)
        right = sample.get(right_key)
        if left is None or right is None:
            continue
        if not np.isfinite(left[0]) or not np.isfinite(right[0]):
            continue
        votes.append(float(left[0]) > float(right[0]))
    if not votes:
        return False
    return sum(votes) >= (len(votes) / 2.0)


def pose_distance(
    prev_pts: Dict[str, np.ndarray],
    cur_pts: Dict[str, np.ndarray],
) -> float:
    total = 0.0
    count = 0
    for key in [
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]:
        prev = prev_pts.get(key)
        cur = cur_pts.get(key)
        if prev is None or cur is None:
            continue
        total += float(np.linalg.norm(cur - prev))
        count += 1
    return total / max(count, 1)


def fix_temporal_side_swaps(
    frames_pts: List[Dict[str, np.ndarray]],
) -> Tuple[List[Dict[str, np.ndarray]], int]:
    if not frames_pts:
        return [], 0

    fixed: List[Dict[str, np.ndarray]] = []
    swaps = 0
    prev = None
    for pts in frames_pts:
        current = {key: np.array(value, dtype=np.float64) for key, value in pts.items()}
        swapped = swap_pose_sides(current)
        choose_swap = looks_side_swapped(current)
        if prev is not None:
            keep_cost = pose_distance(prev, current)
            swap_cost = pose_distance(prev, swapped)
            if swap_cost + 1e-6 < keep_cost * 0.9:
                choose_swap = True
        if choose_swap:
            current = swapped
            swaps += 1
        fixed.append(current)
        prev = current
    return fixed, swaps
