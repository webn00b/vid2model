from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def interpolate_pose_points(
    prev_pts: Dict[str, np.ndarray],
    next_pts: Dict[str, np.ndarray],
    alpha: float,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key in prev_pts:
        out[key] = prev_pts[key] * (1.0 - alpha) + next_pts[key] * alpha
    return out


def fill_pose_gaps(
    frames_pts: List[Optional[Dict[str, np.ndarray]]],
    max_gap_interpolate: int,
) -> Tuple[List[Dict[str, np.ndarray]], int, int]:
    known_indices = [idx for idx, pts in enumerate(frames_pts) if pts is not None]
    if not known_indices:
        raise RuntimeError("No detectable human pose frames found.")

    first_idx = known_indices[0]
    last_idx = known_indices[-1]
    interpolated_frames = 0
    carried_frames = 0

    filled: List[Optional[Dict[str, np.ndarray]]] = list(frames_pts)

    for idx in range(0, first_idx):
        filled[idx] = filled[first_idx]
        carried_frames += 1
    for idx in range(last_idx + 1, len(filled)):
        filled[idx] = filled[last_idx]
        carried_frames += 1

    for prev_idx, next_idx in zip(known_indices, known_indices[1:]):
        gap = next_idx - prev_idx - 1
        if gap <= 0:
            continue

        prev_pts = filled[prev_idx]
        next_pts = filled[next_idx]
        if prev_pts is None or next_pts is None:
            continue

        if gap <= max_gap_interpolate:
            for step in range(1, gap + 1):
                alpha = step / (gap + 1)
                filled[prev_idx + step] = interpolate_pose_points(prev_pts, next_pts, alpha)
                interpolated_frames += 1
        else:
            for step in range(1, gap + 1):
                filled[prev_idx + step] = prev_pts
                carried_frames += 1

    if any(pts is None for pts in filled):
        raise RuntimeError("Internal error: pose gap filling produced unresolved frames.")

    return [pts for pts in filled if pts is not None], interpolated_frames, carried_frames
