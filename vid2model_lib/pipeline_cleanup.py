from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .skeleton import JOINTS, MAP_TO_POINTS


def _smooth_pose_frames(
    frames_pts: List[Dict[str, np.ndarray]],
    alpha: float = 0.35,
) -> List[Dict[str, np.ndarray]]:
    if not frames_pts:
        return frames_pts
    alpha = max(0.0, min(1.0, float(alpha)))
    if alpha <= 1e-6:
        return [{key: np.array(value, dtype=np.float64) for key, value in pts.items()} for pts in frames_pts]

    keys = list(frames_pts[0].keys())

    forward: List[Dict[str, np.ndarray]] = []
    prev = {key: np.array(frames_pts[0][key], dtype=np.float64) for key in keys}
    forward.append(prev)
    for pts in frames_pts[1:]:
        current: Dict[str, np.ndarray] = {}
        for key in keys:
            current[key] = prev[key] * (1.0 - alpha) + np.array(pts[key], dtype=np.float64) * alpha
        forward.append(current)
        prev = current

    backward: List[Optional[Dict[str, np.ndarray]]] = [None] * len(frames_pts)
    prev = {key: np.array(frames_pts[-1][key], dtype=np.float64) for key in keys}
    backward[-1] = prev
    for idx in range(len(frames_pts) - 2, -1, -1):
        pts = frames_pts[idx]
        current = {}
        for key in keys:
            current[key] = prev[key] * (1.0 - alpha) + np.array(pts[key], dtype=np.float64) * alpha
        backward[idx] = current
        prev = current

    smoothed: List[Dict[str, np.ndarray]] = []
    for idx in range(len(frames_pts)):
        combined = {}
        for key in keys:
            combined[key] = (forward[idx][key] + backward[idx][key]) * 0.5
        smoothed.append(combined)
    return smoothed


def _build_target_segment_lengths(samples: List[Dict[str, np.ndarray]]) -> Dict[str, float]:
    lengths: Dict[str, float] = {}
    if not samples:
        return lengths
    for joint in JOINTS:
        if joint.parent is None:
            continue
        parent_point = MAP_TO_POINTS[joint.parent][0]
        child_point = MAP_TO_POINTS[joint.name][0]
        segment_lengths = []
        for sample in samples:
            parent = sample.get(parent_point)
            child = sample.get(child_point)
            if parent is None or child is None:
                continue
            length = float(np.linalg.norm(child - parent))
            if np.isfinite(length) and length > 1e-6:
                segment_lengths.append(length)
        if segment_lengths:
            lengths[joint.name] = float(np.median(segment_lengths))
    return lengths


def _apply_segment_length_constraints(
    pts: Dict[str, np.ndarray],
    target_lengths: Dict[str, float],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {key: np.array(value, dtype=np.float64) for key, value in pts.items()}
    for joint in JOINTS:
        if joint.parent is None:
            continue
        target_length = target_lengths.get(joint.name)
        if target_length is None or target_length <= 1e-6:
            continue
        parent_point = MAP_TO_POINTS[joint.parent][0]
        child_point = MAP_TO_POINTS[joint.name][0]
        parent = out.get(parent_point)
        child = out.get(child_point)
        if parent is None or child is None:
            continue
        direction = child - parent
        length = float(np.linalg.norm(direction))
        if not np.isfinite(length) or length <= 1e-8:
            continue
        out[child_point] = parent + (direction / length) * target_length
    return out


def _copy_pose_frame(pts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {key: np.array(value, dtype=np.float64) for key, value in pts.items()}


def _foot_keys(side: str) -> Tuple[str, str, str]:
    return (f"{side}_ankle", f"{side}_toes", f"{side}_heel")


def _leg_keys(side: str) -> Tuple[str, str, str]:
    return (f"{side}_hip", f"{side}_knee", f"{side}_ankle")


def _empty_cleanup_stats(smooth_alpha: float = 0.0, length_constraints: float = 0.0) -> Dict[str, float]:
    return {
        "side_swaps": 0.0,
        "smooth_alpha": smooth_alpha,
        "length_constraints": length_constraints,
        "contact_windows": 0.0,
        "contact_frames": 0.0,
        "pelvis_contact_frames": 0.0,
        "leg_ik_frames": 0.0,
    }


def _apply_length_constraints_to_frames(
    frames_pts: List[Dict[str, np.ndarray]],
    target_lengths: Dict[str, float],
) -> List[Dict[str, np.ndarray]]:
    return [_apply_segment_length_constraints(frame, target_lengths) for frame in frames_pts]


def _contact_cleanup_stats(
    smooth_alpha: float,
    target_lengths: Dict[str, float],
    contact_stats: Dict[str, float] | None = None,
    pelvis_stats: Dict[str, float] | None = None,
    leg_ik_stats: Dict[str, float] | None = None,
) -> Dict[str, float]:
    stats = _empty_cleanup_stats(smooth_alpha=smooth_alpha, length_constraints=float(len(target_lengths)))
    if contact_stats:
        stats["contact_windows"] = contact_stats["contact_windows"]
        stats["contact_frames"] = contact_stats["contact_frames"]
    if pelvis_stats:
        stats["pelvis_contact_frames"] = pelvis_stats["pelvis_contact_frames"]
    if leg_ik_stats:
        stats["leg_ik_frames"] = leg_ik_stats["leg_ik_frames"]
    return stats


def _detect_foot_contact_mask(
    frames_pts: List[Dict[str, np.ndarray]],
    side: str,
) -> np.ndarray:
    if len(frames_pts) < 2:
        return np.zeros(len(frames_pts), dtype=bool)

    ankle_key, toes_key, heel_key = _foot_keys(side)

    support_heights = []
    foot_speeds = [0.0]
    for idx, pts in enumerate(frames_pts):
        ankle = np.array(pts[ankle_key], dtype=np.float64)
        toes = np.array(pts[toes_key], dtype=np.float64)
        heel = np.array(pts[heel_key], dtype=np.float64)
        support_heights.append(float(min(ankle[1], toes[1], heel[1])))
        if idx > 0:
            prev = frames_pts[idx - 1]
            prev_ankle = np.array(prev[ankle_key], dtype=np.float64)
            prev_toes = np.array(prev[toes_key], dtype=np.float64)
            ankle_speed = float(np.linalg.norm((ankle - prev_ankle)[[0, 2]]))
            toes_speed = float(np.linalg.norm((toes - prev_toes)[[0, 2]]))
            foot_speeds.append((ankle_speed + toes_speed) * 0.5)

    height_floor = float(np.percentile(support_heights, 25))
    height_tol = max(float(np.std(support_heights)) * 0.75, 0.8)
    speed_tol = max(float(np.percentile(foot_speeds, 40)), 0.35)

    mask = np.zeros(len(frames_pts), dtype=bool)
    for idx, (height, speed) in enumerate(zip(support_heights, foot_speeds)):
        if height <= height_floor + height_tol and speed <= speed_tol:
            mask[idx] = True

    if len(mask) >= 3:
        refined = mask.copy()
        for idx in range(1, len(mask) - 1):
            if not mask[idx] and mask[idx - 1] and mask[idx + 1]:
                refined[idx] = True
        mask = refined
    return mask


def _mask_runs(mask: np.ndarray, min_len: int = 2) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            if idx - start >= min_len:
                runs.append((start, idx))
            start = None
    if start is not None and len(mask) - start >= min_len:
        runs.append((start, len(mask)))
    return runs


def _stabilize_foot_contacts(
    frames_pts: List[Dict[str, np.ndarray]],
) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, float]]:
    if len(frames_pts) < 2:
        return [_copy_pose_frame(pts) for pts in frames_pts], {"contact_windows": 0.0, "contact_frames": 0.0}

    adjusted = [_copy_pose_frame(pts) for pts in frames_pts]
    contact_windows = 0
    contact_frames = 0

    for side in ("left", "right"):
        ankle_key, toes_key, heel_key = _foot_keys(side)
        mask = _detect_foot_contact_mask(adjusted, side)
        for start, end in _mask_runs(mask, min_len=2):
            window = adjusted[start:end]
            if not window:
                continue
            contact_windows += 1
            contact_frames += end - start

            ankle_xz = np.median(np.stack([frame[ankle_key][[0, 2]] for frame in window], axis=0), axis=0)
            toes_xz = np.median(np.stack([frame[toes_key][[0, 2]] for frame in window], axis=0), axis=0)
            heel_xz = np.median(np.stack([frame[heel_key][[0, 2]] for frame in window], axis=0), axis=0)
            support_target = float(
                np.median(
                    [
                        min(float(frame[ankle_key][1]), float(frame[toes_key][1]), float(frame[heel_key][1]))
                        for frame in window
                    ]
                )
            )

            for idx in range(start, end):
                frame = adjusted[idx]
                support_height = min(
                    float(frame[ankle_key][1]),
                    float(frame[toes_key][1]),
                    float(frame[heel_key][1]),
                )
                delta_y = support_target - support_height
                frame[ankle_key] = np.array(
                    [ankle_xz[0], float(frame[ankle_key][1]) + delta_y, ankle_xz[1]],
                    dtype=np.float64,
                )
                frame[toes_key] = np.array(
                    [toes_xz[0], float(frame[toes_key][1]) + delta_y, toes_xz[1]],
                    dtype=np.float64,
                )
                frame[heel_key] = np.array(
                    [heel_xz[0], float(frame[heel_key][1]) + delta_y, heel_xz[1]],
                    dtype=np.float64,
                )

    return adjusted, {
        "contact_windows": float(contact_windows),
        "contact_frames": float(contact_frames),
    }


def _stabilize_pelvis_during_contacts(
    frames_pts: List[Dict[str, np.ndarray]],
) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, float]]:
    if len(frames_pts) < 2:
        return [_copy_pose_frame(pts) for pts in frames_pts], {"pelvis_contact_frames": 0.0}

    adjusted = [_copy_pose_frame(pts) for pts in frames_pts]
    desired_mid_hip_xz: List[List[np.ndarray]] = [[] for _ in adjusted]
    pelvis_contact_frames = 0

    for side in ("left", "right"):
        ankle_key, _, _ = _foot_keys(side)
        mask = _detect_foot_contact_mask(adjusted, side)
        for start, end in _mask_runs(mask, min_len=2):
            window = adjusted[start:end]
            if not window:
                continue
            pelvis_contact_frames += end - start
            target_offset = np.median(
                np.stack(
                    [
                        np.array(frame["mid_hip"][[0, 2]], dtype=np.float64)
                        - np.array(frame[ankle_key][[0, 2]], dtype=np.float64)
                        for frame in window
                    ],
                    axis=0,
                ),
                axis=0,
            )
            for idx in range(start, end):
                ankle_xz = np.array(adjusted[idx][ankle_key][[0, 2]], dtype=np.float64)
                desired_mid_hip_xz[idx].append(ankle_xz + target_offset)

    if pelvis_contact_frames <= 0:
        return adjusted, {"pelvis_contact_frames": 0.0}

    movable_keys = [
        key
        for key in adjusted[0].keys()
        if key
        not in {
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_toes",
            "right_toes",
            "left_heel",
            "right_heel",
        }
    ]

    for idx, candidates in enumerate(desired_mid_hip_xz):
        if not candidates:
            continue
        current_mid_hip = np.array(adjusted[idx]["mid_hip"][[0, 2]], dtype=np.float64)
        target_mid_hip = np.mean(np.stack(candidates, axis=0), axis=0)
        delta_xz = (target_mid_hip - current_mid_hip) * 0.7
        if float(np.linalg.norm(delta_xz)) < 1e-8:
            continue
        for key in movable_keys:
            point = np.array(adjusted[idx][key], dtype=np.float64)
            point[0] += float(delta_xz[0])
            point[2] += float(delta_xz[1])
            adjusted[idx][key] = point

    return adjusted, {"pelvis_contact_frames": float(pelvis_contact_frames)}


def _apply_leg_ik_during_contacts(
    frames_pts: List[Dict[str, np.ndarray]],
    target_lengths: Dict[str, float],
) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, float]]:
    if len(frames_pts) < 2:
        return [_copy_pose_frame(pts) for pts in frames_pts], {"leg_ik_frames": 0.0}

    adjusted = [_copy_pose_frame(pts) for pts in frames_pts]
    leg_ik_frames = 0

    for side in ("left", "right"):
        hip_key, knee_key, ankle_key = _leg_keys(side)
        upper_len = float(target_lengths.get(f"{side}LowerLeg", 0.0))
        lower_len = float(target_lengths.get(f"{side}Foot", 0.0))
        if upper_len <= 1e-6 or lower_len <= 1e-6:
            continue
        mask = _detect_foot_contact_mask(adjusted, side)
        for start, end in _mask_runs(mask, min_len=2):
            for idx in range(start, end):
                frame = adjusted[idx]
                hip = np.array(frame[hip_key], dtype=np.float64)
                ankle = np.array(frame[ankle_key], dtype=np.float64)
                current_knee = np.array(frame[knee_key], dtype=np.float64)

                root_to_end = ankle - hip
                distance = float(np.linalg.norm(root_to_end))
                if not np.isfinite(distance) or distance <= 1e-8:
                    continue
                direction = root_to_end / distance

                bend_ref = current_knee - hip
                bend_ref = bend_ref - direction * float(np.dot(bend_ref, direction))
                bend_norm = float(np.linalg.norm(bend_ref))
                if bend_norm <= 1e-8:
                    lateral_sign = -1.0 if side == "left" else 1.0
                    bend_ref = np.array([lateral_sign, 0.0, 0.0], dtype=np.float64)
                    bend_ref = bend_ref - direction * float(np.dot(bend_ref, direction))
                    bend_norm = float(np.linalg.norm(bend_ref))
                    if bend_norm <= 1e-8:
                        bend_ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                        bend_ref = bend_ref - direction * float(np.dot(bend_ref, direction))
                        bend_norm = float(np.linalg.norm(bend_ref))
                        if bend_norm <= 1e-8:
                            continue
                bend_dir = bend_ref / bend_norm

                clamped_dist = min(distance, upper_len + lower_len - 1e-6)
                along = (clamped_dist * clamped_dist + upper_len * upper_len - lower_len * lower_len) / (2.0 * clamped_dist)
                height_sq = max(0.0, upper_len * upper_len - along * along)
                height = float(np.sqrt(height_sq))
                new_knee = hip + direction * along + bend_dir * height
                frame[knee_key] = np.array(new_knee, dtype=np.float64)
                leg_ik_frames += 1

    return adjusted, {"leg_ik_frames": float(leg_ik_frames)}


def cleanup_pose_frames(
    frames_pts: List[Dict[str, np.ndarray]],
    anchor_samples: List[Dict[str, np.ndarray]],
    use_contact_cleanup: bool = True,
) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, float]]:
    smooth_alpha = 0.35
    if not frames_pts:
        return frames_pts, _empty_cleanup_stats()

    smoothed = _smooth_pose_frames(frames_pts, alpha=smooth_alpha)
    target_lengths = _build_target_segment_lengths(anchor_samples or smoothed[:20])
    constrained = _apply_length_constraints_to_frames(smoothed, target_lengths)
    if not use_contact_cleanup:
        return constrained, _contact_cleanup_stats(smooth_alpha, target_lengths)

    stabilized, contact_stats = _stabilize_foot_contacts(constrained)
    stabilized, pelvis_stats = _stabilize_pelvis_during_contacts(stabilized)
    stabilized = _apply_length_constraints_to_frames(stabilized, target_lengths)
    stabilized, leg_ik_stats = _apply_leg_ik_during_contacts(stabilized, target_lengths)
    return stabilized, _contact_cleanup_stats(
        smooth_alpha,
        target_lengths,
        contact_stats=contact_stats,
        pelvis_stats=pelvis_stats,
        leg_ik_stats=leg_ik_stats,
    )
