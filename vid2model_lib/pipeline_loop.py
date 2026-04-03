from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


LOOP_FEATURE_KEYS: Tuple[str, ...] = (
    "spine",
    "chest",
    "upper_chest",
    "neck",
    "head",
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
    "left_toes",
    "right_toes",
)


def _normalize_loop_mode(mode: str) -> str:
    return str(mode or "off").strip().lower()


def _copy_pose_frame(pts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {key: np.array(value, dtype=np.float64) for key, value in pts.items()}


def _loop_feature_matrices(frames_pts: List[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    feature_matrix = np.stack([_loop_feature_vector(frame) for frame in frames_pts], axis=0)
    velocity_matrix = np.zeros_like(feature_matrix)
    if len(frames_pts) >= 2:
        velocity_matrix[1:] = feature_matrix[1:] - feature_matrix[:-1]
    return feature_matrix, velocity_matrix


def _loopability_base_stats(frames_pts: List[Dict[str, np.ndarray]]) -> Dict[str, float | str]:
    return {
        "label": "oneshot",
        "score": 0.0,
        "start": 0.0,
        "end": float(max(0, len(frames_pts) - 1)),
        "length": float(len(frames_pts)),
        "coverage": 0.0,
        "pose_dist": 0.0,
        "vel_dist": 0.0,
        "shoulder_dist": 0.0,
        "full_pose_dist": 0.0,
        "full_vel_dist": 0.0,
    }


def _extract_loop_base_stats(frames_pts: List[Dict[str, np.ndarray]]) -> Dict[str, float]:
    return {
        "applied": 0.0,
        "start": 0.0,
        "end": float(max(0, len(frames_pts) - 1)),
        "score": 0.0,
        "length": float(len(frames_pts)),
    }


def _blend_loop_base_stats() -> Dict[str, float]:
    return {
        "applied": 0.0,
        "blend_frames": 0.0,
        "score_before": 0.0,
        "score_after": 0.0,
    }


def _full_loop_distances(
    feature_matrix: np.ndarray,
    velocity_matrix: np.ndarray,
) -> Tuple[float, float]:
    full_pose_dist = float(np.linalg.norm(feature_matrix[0] - feature_matrix[-1]))
    full_vel_dist = float(np.linalg.norm(velocity_matrix[min(1, len(feature_matrix) - 1)] - velocity_matrix[-1]))
    return full_pose_dist, full_vel_dist


def _loop_feature_vector(pts: Dict[str, np.ndarray]) -> np.ndarray:
    mid_hip = np.array(pts.get("mid_hip", np.zeros(3, dtype=np.float64)), dtype=np.float64)
    chest = np.array(pts.get("chest", mid_hip + np.array([0.0, 1.0, 0.0])), dtype=np.float64)
    head = np.array(pts.get("head", chest), dtype=np.float64)
    shoulder_width = float(
        np.linalg.norm(
            np.array(pts.get("right_shoulder", chest), dtype=np.float64)
            - np.array(pts.get("left_shoulder", chest), dtype=np.float64)
        )
    )
    torso_height = float(np.linalg.norm(head - mid_hip))
    scale = max(shoulder_width, torso_height, 1e-6)
    features = []
    for key in LOOP_FEATURE_KEYS:
        point = np.array(pts.get(key, mid_hip), dtype=np.float64)
        features.append((point - mid_hip) / scale)
    return np.concatenate(features, axis=0)


def _find_best_loop_window(
    feature_matrix: np.ndarray,
    velocity_matrix: np.ndarray,
    fps: float,
) -> Dict[str, float] | None:
    frame_count = int(feature_matrix.shape[0])
    min_frames = max(8, int(round(max(float(fps), 1.0) * 0.6)))
    if frame_count < min_frames + 4:
        return None

    start_limit = max(1, int(frame_count * 0.25))
    end_start = max(min_frames, int(frame_count * 0.6))
    best: Tuple[float, int, int, float, float, float] | None = None

    for start in range(0, start_limit + 1):
        for end in range(end_start, frame_count):
            length = end - start + 1
            if length < min_frames:
                continue
            pose_dist = float(np.linalg.norm(feature_matrix[start] - feature_matrix[end]))
            vel_start_idx = min(start + 1, frame_count - 1)
            vel_end_idx = end
            vel_dist = float(np.linalg.norm(velocity_matrix[vel_start_idx] - velocity_matrix[vel_end_idx]))
            shoulder_dist = float(np.linalg.norm(feature_matrix[start][15:21] - feature_matrix[end][15:21]))
            score = pose_dist + vel_dist * 0.35 + shoulder_dist * 0.15
            score -= (length / max(frame_count, 1)) * 0.08
            if best is None or score < best[0]:
                best = (score, start, end, pose_dist, vel_dist, shoulder_dist)

    if best is None:
        return None
    score, start, end, pose_dist, vel_dist, shoulder_dist = best
    length = end - start + 1
    return {
        "score": float(score),
        "start": float(start),
        "end": float(end),
        "length": float(length),
        "coverage": float(length / max(frame_count, 1)),
        "pose_dist": float(pose_dist),
        "vel_dist": float(vel_dist),
        "shoulder_dist": float(shoulder_dist),
    }


def analyze_motion_loopability(
    frames_pts: List[Dict[str, np.ndarray]],
    fps: float,
) -> Dict[str, float | str]:
    base_stats = _loopability_base_stats(frames_pts)
    if len(frames_pts) < 12:
        return base_stats

    feature_matrix, velocity_matrix = _loop_feature_matrices(frames_pts)
    full_pose_dist, full_vel_dist = _full_loop_distances(feature_matrix, velocity_matrix)
    best = _find_best_loop_window(feature_matrix, velocity_matrix, fps)
    if best is None:
        return {
            **base_stats,
            "full_pose_dist": full_pose_dist,
            "full_vel_dist": full_vel_dist,
        }

    label = "cyclic"
    if (
        best["score"] > 0.95
        or best["coverage"] < 0.7
        or best["pose_dist"] > 0.85
        or best["vel_dist"] > 1.1
        or full_pose_dist > 1.15
        or full_vel_dist > 1.35
    ):
        label = "oneshot"

    return {
        "label": label,
        "full_pose_dist": full_pose_dist,
        "full_vel_dist": full_vel_dist,
        **best,
    }


def extract_motion_loop(
    frames_pts: List[Dict[str, np.ndarray]],
    fps: float,
    mode: str = "off",
) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, float]]:
    normalized_mode = _normalize_loop_mode(mode)
    base_stats = _extract_loop_base_stats(frames_pts)
    if normalized_mode == "off" or len(frames_pts) < 12:
        return frames_pts, base_stats

    feature_matrix, velocity_matrix = _loop_feature_matrices(frames_pts)
    best = _find_best_loop_window(feature_matrix, velocity_matrix, fps)
    if best is None:
        return frames_pts, base_stats

    best_score = float(best["score"])
    best_start = int(best["start"])
    best_end = int(best["end"])
    if normalized_mode == "auto":
        loopability = analyze_motion_loopability(frames_pts, fps)
        if str(loopability["label"]) != "cyclic":
            return frames_pts, {
                **base_stats,
                "score": float(loopability["score"]),
                "start": float(loopability["start"]),
                "end": float(loopability["end"]),
                "length": float(loopability["length"]),
            }
    if not np.isfinite(best_score) or best_score > 1.25:
        return frames_pts, {**base_stats, "score": float(best_score)}

    trimmed = [_copy_pose_frame(frame) for frame in frames_pts[best_start : best_end + 1]]
    return trimmed, {
        "applied": 1.0,
        "start": float(best_start),
        "end": float(best_end),
        "score": float(best_score),
        "length": float(len(trimmed)),
    }


def blend_motion_loop_edges(
    frames_pts: List[Dict[str, np.ndarray]],
    fps: float,
    mode: str = "off",
) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, float]]:
    normalized_mode = _normalize_loop_mode(mode)
    base_stats = _blend_loop_base_stats()
    if normalized_mode == "off" or len(frames_pts) < 10:
        return frames_pts, base_stats

    feature_start = _loop_feature_vector(frames_pts[0])
    feature_end = _loop_feature_vector(frames_pts[-1])
    score_before = float(np.linalg.norm(feature_start - feature_end))

    blend_frames = min(max(2, int(round(max(float(fps), 1.0) * 0.12))), max(2, len(frames_pts) // 4))
    if blend_frames * 2 > len(frames_pts):
        return frames_pts, {**base_stats, "score_before": score_before}

    adjusted = [_copy_pose_frame(frame) for frame in frames_pts]
    keys = list(adjusted[0].keys())
    for i in range(blend_frames):
        pair_idx = len(adjusted) - blend_frames + i
        edge_weight = 1.0 - (i / max(blend_frames - 1, 1))
        pair_strength = edge_weight * 0.85
        for key in keys:
            start_vec = np.array(adjusted[i][key], dtype=np.float64)
            end_vec = np.array(adjusted[pair_idx][key], dtype=np.float64)
            midpoint = (start_vec + end_vec) * 0.5
            adjusted[i][key] = start_vec * (1.0 - pair_strength) + midpoint * pair_strength
            adjusted[pair_idx][key] = end_vec * (1.0 - pair_strength) + midpoint * pair_strength

    score_after = float(np.linalg.norm(_loop_feature_vector(adjusted[0]) - _loop_feature_vector(adjusted[-1])))
    return adjusted, {
        "applied": 1.0 if score_after < score_before - 1e-6 else 0.0,
        "blend_frames": float(blend_frames),
        "score_before": score_before,
        "score_after": score_after,
    }
