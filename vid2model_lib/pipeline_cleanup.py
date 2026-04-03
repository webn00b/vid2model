from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .skeleton import JOINTS, MAP_TO_POINTS


_ADAPTIVE_SMOOTHING_EXPRESSIVE_KEYS = {
    "head",
    "neck",
    "left_wrist",
    "right_wrist",
    "left_elbow",
    "right_elbow",
    "left_ankle",
    "right_ankle",
    "left_toes",
    "right_toes",
}


def _adaptive_smoothing_alpha_schedule(
    frames_pts: List[Dict[str, np.ndarray]],
    keys: List[str],
    base_alpha: float,
) -> Dict[str, np.ndarray]:
    if len(frames_pts) < 2:
        return {key: np.full(len(frames_pts), base_alpha, dtype=np.float64) for key in keys}

    schedules: Dict[str, np.ndarray] = {}
    for key in keys:
        speeds = np.zeros(len(frames_pts), dtype=np.float64)
        for idx in range(1, len(frames_pts)):
            current = np.array(frames_pts[idx][key], dtype=np.float64)
            previous = np.array(frames_pts[idx - 1][key], dtype=np.float64)
            speeds[idx] = float(np.linalg.norm(current - previous))
        reference = max(float(np.percentile(speeds, 65)), 1e-4)
        expressive_boost = 1.18 if key in _ADAPTIVE_SMOOTHING_EXPRESSIVE_KEYS else 1.0
        schedule = np.full(len(frames_pts), base_alpha, dtype=np.float64)
        for idx, speed in enumerate(speeds):
            motion_ratio = min(max(speed / reference, 0.0), 2.0)
            alpha_scale = 0.55 + motion_ratio * 0.45
            schedule[idx] = max(
                0.12,
                min(0.82, base_alpha * alpha_scale * expressive_boost),
            )
        schedules[key] = schedule
    return schedules


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
    adaptive_alphas = _adaptive_smoothing_alpha_schedule(frames_pts, keys, alpha)

    forward: List[Dict[str, np.ndarray]] = []
    prev = {key: np.array(frames_pts[0][key], dtype=np.float64) for key in keys}
    forward.append(prev)
    for idx, pts in enumerate(frames_pts[1:], start=1):
        current: Dict[str, np.ndarray] = {}
        for key in keys:
            frame_alpha = float(adaptive_alphas[key][idx])
            current[key] = prev[key] * (1.0 - frame_alpha) + np.array(pts[key], dtype=np.float64) * frame_alpha
        forward.append(current)
        prev = current

    backward: List[Optional[Dict[str, np.ndarray]]] = [None] * len(frames_pts)
    prev = {key: np.array(frames_pts[-1][key], dtype=np.float64) for key in keys}
    backward[-1] = prev
    for idx in range(len(frames_pts) - 2, -1, -1):
        pts = frames_pts[idx]
        current = {}
        for key in keys:
            frame_alpha = float(adaptive_alphas[key][idx])
            current[key] = prev[key] * (1.0 - frame_alpha) + np.array(pts[key], dtype=np.float64) * frame_alpha
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


def _smooth_optional_vectors(
    vectors: List[Optional[np.ndarray]],
    alpha: float = 0.45,
) -> List[Optional[np.ndarray]]:
    if not vectors:
        return []
    alpha = max(0.0, min(1.0, float(alpha)))
    if alpha <= 1e-6:
        return [None if value is None else np.array(value, dtype=np.float64) for value in vectors]

    forward: List[Optional[np.ndarray]] = [None] * len(vectors)
    prev: Optional[np.ndarray] = None
    for idx, value in enumerate(vectors):
        if value is None:
            forward[idx] = None if prev is None else np.array(prev, dtype=np.float64)
            continue
        current = np.array(value, dtype=np.float64)
        if prev is not None:
            current = prev * (1.0 - alpha) + current * alpha
        forward[idx] = current
        prev = current

    backward: List[Optional[np.ndarray]] = [None] * len(vectors)
    prev = None
    for idx in range(len(vectors) - 1, -1, -1):
        value = vectors[idx]
        if value is None:
            backward[idx] = None if prev is None else np.array(prev, dtype=np.float64)
            continue
        current = np.array(value, dtype=np.float64)
        if prev is not None:
            current = prev * (1.0 - alpha) + current * alpha
        backward[idx] = current
        prev = current

    smoothed: List[Optional[np.ndarray]] = [None] * len(vectors)
    for idx, value in enumerate(vectors):
        if value is None:
            continue
        if forward[idx] is not None and backward[idx] is not None:
            smoothed[idx] = (forward[idx] + backward[idx]) * 0.5
        elif forward[idx] is not None:
            smoothed[idx] = np.array(forward[idx], dtype=np.float64)
        elif backward[idx] is not None:
            smoothed[idx] = np.array(backward[idx], dtype=np.float64)
    return smoothed


def _empty_cleanup_stats(smooth_alpha: float = 0.0, length_constraints: float = 0.0) -> Dict[str, float]:
    return {
        "side_swaps": 0.0,
        "smooth_alpha": smooth_alpha,
        "adaptive_smoothing": 1.0 if smooth_alpha > 0.0 else 0.0,
        "length_constraints": length_constraints,
        "contact_windows": 0.0,
        "contact_frames": 0.0,
        "pelvis_contact_frames": 0.0,
        "root_stabilized_frames": 0.0,
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
        stats["root_stabilized_frames"] = pelvis_stats.get("root_stabilized_frames", 0.0)
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
            prev_heel = np.array(prev[heel_key], dtype=np.float64)
            ankle_speed = float(np.linalg.norm((ankle - prev_ankle)[[0, 2]]))
            toes_speed = float(np.linalg.norm((toes - prev_toes)[[0, 2]]))
            heel_speed = float(np.linalg.norm((heel - prev_heel)[[0, 2]]))
            foot_speeds.append((ankle_speed + toes_speed + heel_speed) / 3.0)

    height_floor = float(np.percentile(support_heights, 25))
    height_tol = max(float(np.std(support_heights)) * 0.6, 0.65)
    speed_tol = max(float(np.percentile(foot_speeds, 35)), 0.22)

    mask = np.zeros(len(frames_pts), dtype=bool)
    for idx, (height, speed) in enumerate(zip(support_heights, foot_speeds)):
        if height <= height_floor + height_tol and speed <= speed_tol:
            mask[idx] = True

    if len(mask) >= 3:
        refined = mask.copy()
        for idx in range(1, len(mask) - 1):
            if not mask[idx] and mask[idx - 1] and mask[idx + 1]:
                refined[idx] = True
        for idx in range(1, len(mask) - 1):
            if refined[idx] and not refined[idx - 1] and not refined[idx + 1]:
                refined[idx] = False
        mask = refined
    return mask


def _contact_window_blend_weights(length: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.float64)
    if length <= 2:
        return np.ones(length, dtype=np.float64)
    weights = np.ones(length, dtype=np.float64)
    edge_span = max(1, min(length // 3, 3))
    for idx in range(edge_span):
        alpha = (idx + 1) / float(edge_span + 1)
        weight = 0.45 + alpha * 0.55
        weights[idx] = min(weights[idx], weight)
        weights[length - 1 - idx] = min(weights[length - 1 - idx], weight)
    return weights


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

            support_centroids = []
            support_speeds = []
            for offset, frame in enumerate(window):
                ankle_xz = np.array(frame[ankle_key][[0, 2]], dtype=np.float64)
                toes_xz = np.array(frame[toes_key][[0, 2]], dtype=np.float64)
                heel_xz = np.array(frame[heel_key][[0, 2]], dtype=np.float64)
                support_centroids.append((ankle_xz + toes_xz + heel_xz) / 3.0)
                if offset == 0:
                    support_speeds.append(0.0)
                else:
                    support_speeds.append(float(np.linalg.norm(support_centroids[-1] - support_centroids[-2])))
            anchor_idx = int(np.argmin(np.array(support_speeds, dtype=np.float64)))
            anchor_frame = window[anchor_idx]
            anchor_centroid = np.array(support_centroids[anchor_idx], dtype=np.float64)
            support_target = float(
                np.median(
                    [
                        min(float(frame[ankle_key][1]), float(frame[toes_key][1]), float(frame[heel_key][1]))
                        for frame in window
                    ]
                )
            )
            weights = _contact_window_blend_weights(end - start)

            for idx in range(start, end):
                frame = adjusted[idx]
                frame_weight = float(weights[idx - start])
                current_centroid = (
                    np.array(frame[ankle_key][[0, 2]], dtype=np.float64)
                    + np.array(frame[toes_key][[0, 2]], dtype=np.float64)
                    + np.array(frame[heel_key][[0, 2]], dtype=np.float64)
                ) / 3.0
                delta_xz = (anchor_centroid - current_centroid) * frame_weight
                support_height = min(
                    float(frame[ankle_key][1]),
                    float(frame[toes_key][1]),
                    float(frame[heel_key][1]),
                )
                delta_y = (support_target - support_height) * frame_weight
                for key in (ankle_key, toes_key, heel_key):
                    point = np.array(frame[key], dtype=np.float64)
                    point[0] += float(delta_xz[0])
                    point[1] += float(delta_y)
                    point[2] += float(delta_xz[1])
                    frame[key] = point

                # Keep the support foot shape close to a stable anchor during contact windows.
                for key in (toes_key, heel_key):
                    anchor_delta = np.array(anchor_frame[key] - anchor_frame[ankle_key], dtype=np.float64)
                    current_delta = np.array(frame[key] - frame[ankle_key], dtype=np.float64)
                    frame[key] = np.array(
                        frame[key] + (anchor_delta - current_delta) * (0.35 * frame_weight),
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
        return [_copy_pose_frame(pts) for pts in frames_pts], {"pelvis_contact_frames": 0.0, "root_stabilized_frames": 0.0}

    adjusted = [_copy_pose_frame(pts) for pts in frames_pts]
    desired_mid_hip: List[List[np.ndarray]] = [[] for _ in adjusted]
    desired_mid_hip_weight: List[List[float]] = [[] for _ in adjusted]
    pelvis_contact_frames = 0

    for side in ("left", "right"):
        ankle_key, toes_key, heel_key = _foot_keys(side)
        mask = _detect_foot_contact_mask(adjusted, side)
        for start, end in _mask_runs(mask, min_len=2):
            window = adjusted[start:end]
            if not window:
                continue
            pelvis_contact_frames += end - start
            support_centroids = np.stack(
                [
                    (
                        np.array(frame[ankle_key][[0, 2]], dtype=np.float64)
                        + np.array(frame[toes_key][[0, 2]], dtype=np.float64)
                        + np.array(frame[heel_key][[0, 2]], dtype=np.float64)
                    )
                    / 3.0
                    for frame in window
                ],
                axis=0,
            )
            support_floor = np.array(
                [
                    min(float(frame[ankle_key][1]), float(frame[toes_key][1]), float(frame[heel_key][1]))
                    for frame in window
                ],
                dtype=np.float64,
            )
            target_offset_xz = np.median(
                np.stack(
                    [
                        np.array(frame["mid_hip"][[0, 2]], dtype=np.float64)
                        - support_centroids[offset]
                        for offset, frame in enumerate(window)
                    ],
                    axis=0,
                ),
                axis=0,
            )
            target_offset_y = float(
                np.median(
                    np.array([float(frame["mid_hip"][1]) for frame in window], dtype=np.float64) - support_floor
                )
            )
            weights = _contact_window_blend_weights(end - start)
            for idx in range(start, end):
                offset = idx - start
                desired_mid_hip[idx].append(
                    np.array(
                        [
                            support_centroids[offset][0] + target_offset_xz[0],
                            support_floor[offset] + target_offset_y,
                            support_centroids[offset][1] + target_offset_xz[1],
                        ],
                        dtype=np.float64,
                    )
                )
                desired_mid_hip_weight[idx].append(float(weights[offset]))

    if pelvis_contact_frames <= 0:
        return adjusted, {"pelvis_contact_frames": 0.0, "root_stabilized_frames": 0.0}

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

    raw_targets: List[Optional[np.ndarray]] = [None] * len(adjusted)
    target_blends = np.zeros(len(adjusted), dtype=np.float64)
    for idx, candidates in enumerate(desired_mid_hip):
        if not candidates:
            continue
        weights = np.array(desired_mid_hip_weight[idx], dtype=np.float64)
        weight_total = float(np.sum(weights))
        if weight_total <= 1e-8:
            continue
        stacked = np.stack(candidates, axis=0)
        raw_targets[idx] = np.average(stacked, axis=0, weights=weights)
        target_blends[idx] = float(np.mean(weights))

    smoothed_targets = _smooth_optional_vectors(raw_targets, alpha=0.4)
    root_stabilized_frames = 0

    for idx, target_mid_hip in enumerate(smoothed_targets):
        if target_mid_hip is None:
            continue
        current_mid_hip = np.array(adjusted[idx]["mid_hip"], dtype=np.float64)
        frame_blend = 0.45 + min(max(float(target_blends[idx]), 0.0), 1.0) * 0.35
        delta_xz = (target_mid_hip[[0, 2]] - current_mid_hip[[0, 2]]) * frame_blend
        delta_y = float(target_mid_hip[1] - current_mid_hip[1]) * min(frame_blend * 0.6, 0.45)
        if float(np.linalg.norm(delta_xz)) < 1e-8 and abs(delta_y) < 1e-8:
            continue
        root_stabilized_frames += 1
        for key in movable_keys:
            point = np.array(adjusted[idx][key], dtype=np.float64)
            point[0] += float(delta_xz[0])
            point[1] += float(delta_y)
            point[2] += float(delta_xz[1])
            adjusted[idx][key] = point

    return adjusted, {
        "pelvis_contact_frames": float(pelvis_contact_frames),
        "root_stabilized_frames": float(root_stabilized_frames),
    }


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
