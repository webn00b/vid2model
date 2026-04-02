from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .math3d import euler_zxy_from_matrix, normalize, rotation_align, rotation_align_with_secondary
from .pose_model import ensure_pose_model
from .pose_points import extract_pose_points
from .skeleton import CHILDREN, JOINTS, MAP_TO_POINTS


@dataclass(frozen=True)
class PoseCorrectionProfile:
    mode: str = "manual"
    model_path: str = ""
    shoulder_tracking: bool = True
    hip_camera: bool = False
    auto_grounding: bool = True
    use_arm_ik: bool = True
    use_leg_ik: bool = True
    body_bend_reduction_power: float = 0.5
    upper_rotation_offset_deg: float = 0.0
    arm_horizontal_offset_percent: float = 0.0
    arm_vertical_offset_percent: float = 0.0
    hip_depth_scale_percent: float = 100.0
    hip_y_position_offset_percent: float = 0.0
    hip_z_position_offset_percent: float = 0.0
    body_collider_mode: int = 2
    body_collider_head_size_percent: float = 100.0
    body_collider_chest_size_percent: float = 100.0
    body_collider_waist_size_percent: float = 100.0
    body_collider_hip_size_percent: float = 100.0
    body_collider_head_reaction_type: str = "z_push"


DEFAULT_POSE_CORRECTIONS = PoseCorrectionProfile()


def build_pose_correction_profile(data: Optional[Dict[str, Any]] = None) -> PoseCorrectionProfile:
    if not data:
        return DEFAULT_POSE_CORRECTIONS

    def as_bool(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return default

    def as_float(value: Any, default: float) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def as_int(value: Any, default: int) -> int:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def as_str(value: Any, default: str) -> str:
        if value is None:
            return default
        return str(value)

    return PoseCorrectionProfile(
        mode=str(data.get("mode", DEFAULT_POSE_CORRECTIONS.mode)).strip().lower() or DEFAULT_POSE_CORRECTIONS.mode,
        model_path=as_str(data.get("model_path"), DEFAULT_POSE_CORRECTIONS.model_path),
        shoulder_tracking=as_bool(data.get("shoulder_tracking"), DEFAULT_POSE_CORRECTIONS.shoulder_tracking),
        hip_camera=as_bool(data.get("hip_camera"), DEFAULT_POSE_CORRECTIONS.hip_camera),
        auto_grounding=as_bool(data.get("auto_grounding"), DEFAULT_POSE_CORRECTIONS.auto_grounding),
        use_arm_ik=as_bool(data.get("use_arm_ik"), DEFAULT_POSE_CORRECTIONS.use_arm_ik),
        use_leg_ik=as_bool(data.get("use_leg_ik"), DEFAULT_POSE_CORRECTIONS.use_leg_ik),
        body_bend_reduction_power=as_float(
            data.get("body_bend_reduction_power"), DEFAULT_POSE_CORRECTIONS.body_bend_reduction_power
        ),
        upper_rotation_offset_deg=as_float(
            data.get("upper_rotation_offset_deg", data.get("upper_rotation_offset")),
            DEFAULT_POSE_CORRECTIONS.upper_rotation_offset_deg,
        ),
        arm_horizontal_offset_percent=as_float(
            data.get("arm_horizontal_offset_percent"), DEFAULT_POSE_CORRECTIONS.arm_horizontal_offset_percent
        ),
        arm_vertical_offset_percent=as_float(
            data.get("arm_vertical_offset_percent"), DEFAULT_POSE_CORRECTIONS.arm_vertical_offset_percent
        ),
        hip_depth_scale_percent=as_float(
            data.get("hip_depth_scale_percent"), DEFAULT_POSE_CORRECTIONS.hip_depth_scale_percent
        ),
        hip_y_position_offset_percent=as_float(
            data.get("hip_y_position_offset_percent"), DEFAULT_POSE_CORRECTIONS.hip_y_position_offset_percent
        ),
        hip_z_position_offset_percent=as_float(
            data.get("hip_z_position_offset_percent"), DEFAULT_POSE_CORRECTIONS.hip_z_position_offset_percent
        ),
        body_collider_mode=as_int(data.get("body_collider_mode"), DEFAULT_POSE_CORRECTIONS.body_collider_mode),
        body_collider_head_size_percent=as_float(
            data.get("body_collider_head_size_percent"), DEFAULT_POSE_CORRECTIONS.body_collider_head_size_percent
        ),
        body_collider_chest_size_percent=as_float(
            data.get("body_collider_chest_size_percent"), DEFAULT_POSE_CORRECTIONS.body_collider_chest_size_percent
        ),
        body_collider_waist_size_percent=as_float(
            data.get("body_collider_waist_size_percent"), DEFAULT_POSE_CORRECTIONS.body_collider_waist_size_percent
        ),
        body_collider_hip_size_percent=as_float(
            data.get("body_collider_hip_size_percent"), DEFAULT_POSE_CORRECTIONS.body_collider_hip_size_percent
        ),
        body_collider_head_reaction_type=as_str(
            data.get("body_collider_head_reaction_type"),
            DEFAULT_POSE_CORRECTIONS.body_collider_head_reaction_type,
        ),
    )


AUTO_POSE_PRESETS: Dict[str, Dict[str, Any]] = {
    "default": {},
    "mirrored": {
        "shoulder_tracking": True,
        "hip_camera": False,
        "auto_grounding": True,
        "body_bend_reduction_power": 0.35,
        "body_collider_mode": 2,
        "body_collider_head_size_percent": 105,
        "body_collider_chest_size_percent": 100,
        "body_collider_waist_size_percent": 98,
        "body_collider_hip_size_percent": 98,
    },
    "low_camera": {
        "shoulder_tracking": False,
        "hip_camera": True,
        "auto_grounding": True,
        "body_bend_reduction_power": 0.60,
        "upper_rotation_offset_deg": 0.0,
        "arm_vertical_offset_percent": 8.0,
        "hip_depth_scale_percent": 115.0,
        "body_collider_mode": 1,
        "body_collider_head_size_percent": 110,
        "body_collider_chest_size_percent": 108,
        "body_collider_waist_size_percent": 102,
        "body_collider_hip_size_percent": 96,
    },
    "wide_arms": {
        "shoulder_tracking": True,
        "hip_camera": False,
        "auto_grounding": True,
        "use_arm_ik": True,
        "body_bend_reduction_power": 0.42,
        "arm_horizontal_offset_percent": 18.0,
        "arm_vertical_offset_percent": 4.0,
        "body_collider_mode": 2,
        "body_collider_head_size_percent": 100,
        "body_collider_chest_size_percent": 112,
        "body_collider_waist_size_percent": 100,
        "body_collider_hip_size_percent": 100,
    },
    "crouched": {
        "shoulder_tracking": True,
        "hip_camera": True,
        "auto_grounding": True,
        "use_leg_ik": True,
        "body_bend_reduction_power": 0.22,
        "hip_y_position_offset_percent": -6.0,
        "hip_depth_scale_percent": 108.0,
        "body_collider_mode": 1,
        "body_collider_head_size_percent": 100,
        "body_collider_chest_size_percent": 100,
        "body_collider_waist_size_percent": 108,
        "body_collider_hip_size_percent": 112,
    },
}


def _finite_or_zero(v: float) -> float:
    return float(v) if np.isfinite(v) else 0.0


def _sample_feature_summary(sample: Dict[str, np.ndarray]) -> Dict[str, float]:
    mid_hip = np.array(sample.get("mid_hip", np.zeros(3)), dtype=np.float64)
    chest = np.array(sample.get("chest", mid_hip), dtype=np.float64)
    head = np.array(sample.get("head", chest), dtype=np.float64)
    left_shoulder = np.array(sample.get("left_shoulder", chest), dtype=np.float64)
    right_shoulder = np.array(sample.get("right_shoulder", chest), dtype=np.float64)
    left_hip = np.array(sample.get("left_hip", mid_hip), dtype=np.float64)
    right_hip = np.array(sample.get("right_hip", mid_hip), dtype=np.float64)
    left_wrist = np.array(sample.get("left_wrist", chest), dtype=np.float64)
    right_wrist = np.array(sample.get("right_wrist", chest), dtype=np.float64)
    left_ankle = np.array(sample.get("left_ankle", mid_hip), dtype=np.float64)
    right_ankle = np.array(sample.get("right_ankle", mid_hip), dtype=np.float64)
    left_knee = np.array(sample.get("left_knee", mid_hip), dtype=np.float64)
    right_knee = np.array(sample.get("right_knee", mid_hip), dtype=np.float64)

    shoulder_width = max(float(np.linalg.norm(right_shoulder - left_shoulder)), 1e-6)
    hip_width = max(float(np.linalg.norm(right_hip - left_hip)), 1e-6)
    torso_height = max(float(np.linalg.norm(chest - mid_hip)), 1e-6)
    head_height = max(float(np.linalg.norm(head - chest)), 1e-6)
    arm_span = float(np.linalg.norm(right_wrist - left_wrist))
    leg_span = float(np.linalg.norm(right_ankle - left_ankle))
    torso_depth = abs(_finite_or_zero(chest[2] - mid_hip[2]))
    head_depth = abs(_finite_or_zero(head[2] - chest[2]))
    shoulder_bias = _finite_or_zero(left_shoulder[0] - right_shoulder[0])
    hip_bias = _finite_or_zero(left_hip[0] - right_hip[0])
    wrist_bias = _finite_or_zero(left_wrist[0] - right_wrist[0])
    ankle_bias = _finite_or_zero(left_ankle[0] - right_ankle[0])
    shoulder_depth_bias = _finite_or_zero(left_shoulder[2] - right_shoulder[2])
    hip_depth_bias = _finite_or_zero(left_hip[2] - right_hip[2])
    wrist_height = ((left_wrist[1] + right_wrist[1]) * 0.5) - chest[1]
    ankle_height = ((left_ankle[1] + right_ankle[1]) * 0.5) - mid_hip[1]
    knee_height = ((left_knee[1] + right_knee[1]) * 0.5) - mid_hip[1]
    hand_depth = abs(_finite_or_zero(left_wrist[2] - right_wrist[2]))
    arm_width_ratio = arm_span / shoulder_width
    leg_width_ratio = leg_span / hip_width
    torso_depth_ratio = torso_depth / shoulder_width
    head_depth_ratio = head_depth / shoulder_width
    hand_drop_ratio = (chest[1] - wrist_height) / torso_height
    crouch_ratio = (mid_hip[1] - ankle_height) / torso_height
    knee_drop_ratio = (mid_hip[1] - knee_height) / torso_height

    return {
        "shoulder_width": shoulder_width,
        "hip_width": hip_width,
        "torso_height": torso_height,
        "head_height": head_height,
        "arm_span": arm_span,
        "leg_span": leg_span,
        "torso_depth": torso_depth,
        "head_depth": head_depth,
        "hand_depth": hand_depth,
        "arm_width_ratio": arm_width_ratio,
        "leg_width_ratio": leg_width_ratio,
        "torso_depth_ratio": torso_depth_ratio,
        "head_depth_ratio": head_depth_ratio,
        "hand_drop_ratio": hand_drop_ratio,
        "crouch_ratio": crouch_ratio,
        "knee_drop_ratio": knee_drop_ratio,
        "shoulder_bias": shoulder_bias,
        "hip_bias": hip_bias,
        "wrist_bias": wrist_bias,
        "ankle_bias": ankle_bias,
        "shoulder_depth_bias": shoulder_depth_bias,
        "hip_depth_bias": hip_depth_bias,
        "arm_span_to_torso": arm_span / torso_height,
        "leg_span_to_torso": leg_span / torso_height,
        "head_to_torso": head_height / torso_height,
    }


def _auto_feature_vector(samples: List[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, Dict[str, float]]:
    summary = _sample_feature_summary(_median_pose_sample(samples))
    vector = np.array(
            [
                summary["shoulder_width"],
                summary["hip_width"],
                summary["torso_height"],
                summary["head_height"],
            summary["arm_span"],
            summary["leg_span"],
            summary["torso_depth"],
            summary["head_depth"],
            summary["hand_depth"],
            summary["arm_width_ratio"],
            summary["leg_width_ratio"],
            summary["torso_depth_ratio"],
            summary["head_depth_ratio"],
                summary["hand_drop_ratio"],
                summary["crouch_ratio"],
                summary["knee_drop_ratio"],
                summary["shoulder_bias"],
                summary["hip_bias"],
                summary["wrist_bias"],
                summary["ankle_bias"],
                summary["shoulder_depth_bias"],
                summary["hip_depth_bias"],
                summary["arm_span_to_torso"],
                summary["leg_span_to_torso"],
                summary["head_to_torso"],
            ],
        dtype=np.float64,
    )
    return vector, summary


def _load_auto_classifier(model_path: str) -> Optional[Dict[str, Any]]:
    path = Path(model_path).expanduser()
    if not path.exists():
        return None
    try:
        if path.suffix.lower() == ".npz":
            data = np.load(path, allow_pickle=True)
            payload = {k: data[k] for k in data.files}
            return payload
        if path.suffix.lower() == ".json":
            import json

            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[vid2model] auto model load failed: {exc}", file=sys.stderr)
        return None
    return None


def _predict_auto_label(features: np.ndarray, summary: Dict[str, float], model_path: str = "") -> Tuple[str, Dict[str, float]]:
    model = _load_auto_classifier(model_path) if model_path else None
    if model:
        try:
            x = np.asarray(features, dtype=np.float64).reshape(-1)
            feature_mean = model.get("feature_mean")
            feature_scale = model.get("feature_scale")
            if feature_mean is not None:
                x = x - np.asarray(feature_mean, dtype=np.float64).reshape(-1)
            if feature_scale is not None:
                scale = np.asarray(feature_scale, dtype=np.float64).reshape(-1)
                scale = np.where(np.abs(scale) < 1e-8, 1.0, scale)
                x = x / scale
            if {"W1", "b1", "W2", "b2"} <= model.keys():
                w1 = np.asarray(model["W1"], dtype=np.float64)
                b1 = np.asarray(model["b1"], dtype=np.float64)
                w2 = np.asarray(model["W2"], dtype=np.float64)
                b2 = np.asarray(model["b2"], dtype=np.float64)
                h = np.tanh(w1 @ x + b1)
                logits = w2 @ h + b2
            elif {"W", "b"} <= model.keys():
                logits = np.asarray(model["W"], dtype=np.float64) @ x + np.asarray(model["b"], dtype=np.float64)
            else:
                raise ValueError("unsupported auto model format")

            classes = [str(v) for v in np.asarray(model.get("classes", list(AUTO_POSE_PRESETS.keys())), dtype=object).tolist()]
            if len(classes) != len(logits):
                raise ValueError("class count mismatch")
            idx = int(np.argmax(logits))
            label = classes[idx]
            score = float(np.max(logits))
            return label if label in AUTO_POSE_PRESETS else "default", {"model_score": score}
        except Exception as exc:
            print(f"[vid2model] auto model inference failed, using heuristic: {exc}", file=sys.stderr)

    scores = {
        "default": 0.0,
        "mirrored": 0.0,
        "low_camera": 0.0,
        "wide_arms": 0.0,
        "crouched": 0.0,
    }
    if summary["hand_drop_ratio"] < 0.12:
        scores["wide_arms"] += 2.0
    if summary["arm_width_ratio"] > 1.25:
        scores["wide_arms"] += 3.0
    if summary["crouch_ratio"] > 0.55 or summary["knee_drop_ratio"] > 0.35:
        scores["crouched"] += 3.0
    if summary["torso_depth_ratio"] > 0.10 or summary["head_depth_ratio"] > 0.08:
        scores["low_camera"] += 2.5
    if summary["arm_span_to_torso"] > 1.8:
        scores["wide_arms"] += 1.0
    if summary["head_to_torso"] > 0.42:
        scores["low_camera"] += 0.5
    label = max(scores.items(), key=lambda item: item[1])[0]
    if scores[label] <= 0.5:
        label = "default"
    return label, {"heuristic_score": float(scores[label])}


def resolve_auto_pose_corrections(
    samples: List[Dict[str, np.ndarray]],
    base: PoseCorrectionProfile,
) -> Tuple[PoseCorrectionProfile, Dict[str, Any]]:
    features, summary = _auto_feature_vector(samples)
    label, meta = _predict_auto_label(features, summary, base.model_path)
    preset = AUTO_POSE_PRESETS.get(label, AUTO_POSE_PRESETS["default"])
    resolved = replace(base, mode="resolved")
    for key, value in preset.items():
        if hasattr(resolved, key):
            resolved = replace(resolved, **{key: value})
    meta.update(
        {
            "label": label,
            "features": {k: round(float(v), 6) for k, v in summary.items()},
        }
    )
    return resolved, meta


def _swap_lr_name(name: str) -> str:
    if name.startswith("left_"):
        return "right_" + name[5:]
    if name.startswith("right_"):
        return "left_" + name[6:]
    return name


def _mirror_pose_points(pts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    mirrored: Dict[str, np.ndarray] = {}
    for key, value in pts.items():
        mirrored[_swap_lr_name(key)] = np.array([-value[0], value[1], value[2]], dtype=np.float64)
    return mirrored


def _swap_pose_sides(pts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    swapped: Dict[str, np.ndarray] = {}
    for key, value in pts.items():
        swapped[_swap_lr_name(key)] = np.array(value, dtype=np.float64)
    return swapped


def _looks_mirrored(sample: Dict[str, np.ndarray]) -> bool:
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


def _looks_side_swapped(sample: Dict[str, np.ndarray]) -> bool:
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


def _pose_distance(
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


def _fix_temporal_side_swaps(
    frames_pts: List[Dict[str, np.ndarray]],
) -> Tuple[List[Dict[str, np.ndarray]], int]:
    if not frames_pts:
        return [], 0

    fixed: List[Dict[str, np.ndarray]] = []
    swaps = 0
    prev = None
    for pts in frames_pts:
        current = {key: np.array(value, dtype=np.float64) for key, value in pts.items()}
        swapped = _swap_pose_sides(current)
        choose_swap = _looks_side_swapped(current)
        if prev is not None:
            keep_cost = _pose_distance(prev, current)
            swap_cost = _pose_distance(prev, swapped)
            if swap_cost + 1e-6 < keep_cost * 0.9:
                choose_swap = True
        if choose_swap:
            current = swapped
            swaps += 1
        fixed.append(current)
        prev = current
    return fixed, swaps


def _smooth_pose_frames(
    frames_pts: List[Dict[str, np.ndarray]],
    alpha: float = 0.35,
) -> List[Dict[str, np.ndarray]]:
    if len(frames_pts) <= 2:
        return [{key: np.array(value, dtype=np.float64) for key, value in pts.items()} for pts in frames_pts]

    alpha = float(np.clip(alpha, 0.05, 0.95))
    keys = list(frames_pts[0].keys())
    forward: List[Dict[str, np.ndarray]] = []
    prev = {key: np.array(value, dtype=np.float64) for key, value in frames_pts[0].items()}
    forward.append(prev)
    for pts in frames_pts[1:]:
        current = {}
        for key in keys:
            current[key] = prev[key] * (1.0 - alpha) + np.array(pts[key], dtype=np.float64) * alpha
        forward.append(current)
        prev = current

    backward: List[Dict[str, np.ndarray]] = [None] * len(frames_pts)  # type: ignore[assignment]
    prev = {key: np.array(value, dtype=np.float64) for key, value in frames_pts[-1].items()}
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


def _detect_foot_contact_mask(
    frames_pts: List[Dict[str, np.ndarray]],
    side: str,
) -> np.ndarray:
    if len(frames_pts) < 2:
        return np.zeros(len(frames_pts), dtype=bool)

    ankle_key = f"{side}_ankle"
    toes_key = f"{side}_toes"
    heel_key = f"{side}_heel"

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
        ankle_key = f"{side}_ankle"
        toes_key = f"{side}_toes"
        heel_key = f"{side}_heel"
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
        ankle_key = f"{side}_ankle"
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
        hip_key = f"{side}_hip"
        knee_key = f"{side}_knee"
        ankle_key = f"{side}_ankle"
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
    if not frames_pts:
        return frames_pts, {
            "side_swaps": 0.0,
            "smooth_alpha": 0.0,
            "length_constraints": 0.0,
            "contact_windows": 0.0,
            "contact_frames": 0.0,
            "pelvis_contact_frames": 0.0,
            "leg_ik_frames": 0.0,
        }

    smoothed = _smooth_pose_frames(frames_pts, alpha=0.35)
    target_lengths = _build_target_segment_lengths(anchor_samples or smoothed[:20])
    constrained = [_apply_segment_length_constraints(frame, target_lengths) for frame in smoothed]
    if not use_contact_cleanup:
        return constrained, {
            "side_swaps": 0.0,
            "smooth_alpha": 0.35,
            "length_constraints": float(len(target_lengths)),
            "contact_windows": 0.0,
            "contact_frames": 0.0,
            "pelvis_contact_frames": 0.0,
            "leg_ik_frames": 0.0,
        }
    stabilized, contact_stats = _stabilize_foot_contacts(constrained)
    stabilized, pelvis_stats = _stabilize_pelvis_during_contacts(stabilized)
    stabilized = [_apply_segment_length_constraints(frame, target_lengths) for frame in stabilized]
    stabilized, leg_ik_stats = _apply_leg_ik_during_contacts(stabilized, target_lengths)
    return stabilized, {
        "side_swaps": 0.0,
        "smooth_alpha": 0.35,
        "length_constraints": float(len(target_lengths)),
        "contact_windows": contact_stats["contact_windows"],
        "contact_frames": contact_stats["contact_frames"],
        "pelvis_contact_frames": pelvis_stats["pelvis_contact_frames"],
        "leg_ik_frames": leg_ik_stats["leg_ik_frames"],
    }


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
) -> Optional[Dict[str, float]]:
    frame_count = int(feature_matrix.shape[0])
    min_frames = max(8, int(round(max(float(fps), 1.0) * 0.6)))
    if frame_count < min_frames + 4:
        return None

    start_limit = max(1, int(frame_count * 0.25))
    end_start = max(min_frames, int(frame_count * 0.6))
    best: Optional[Tuple[float, int, int, float, float, float]] = None

    for start in range(0, start_limit + 1):
        for end in range(end_start, frame_count):
            length = end - start + 1
            if length < min_frames:
                continue
            pose_dist = float(np.linalg.norm(feature_matrix[start] - feature_matrix[end]))
            vel_start_idx = min(start + 1, frame_count - 1)
            vel_end_idx = end
            vel_dist = float(np.linalg.norm(velocity_matrix[vel_start_idx] - velocity_matrix[vel_end_idx]))
            shoulder_dist = float(
                np.linalg.norm(
                    feature_matrix[start][15:21] - feature_matrix[end][15:21]
                )
            )
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
    base_stats: Dict[str, float | str] = {
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
    if len(frames_pts) < 12:
        return base_stats

    feature_matrix = np.stack([_loop_feature_vector(frame) for frame in frames_pts], axis=0)
    velocity_matrix = np.zeros_like(feature_matrix)
    velocity_matrix[1:] = feature_matrix[1:] - feature_matrix[:-1]
    full_pose_dist = float(np.linalg.norm(feature_matrix[0] - feature_matrix[-1]))
    full_vel_dist = float(np.linalg.norm(velocity_matrix[min(1, len(frames_pts) - 1)] - velocity_matrix[-1]))
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
    normalized_mode = str(mode or "off").strip().lower()
    base_stats = {
        "applied": 0.0,
        "start": 0.0,
        "end": float(max(0, len(frames_pts) - 1)),
        "score": 0.0,
        "length": float(len(frames_pts)),
    }
    if normalized_mode == "off" or len(frames_pts) < 12:
        return frames_pts, base_stats

    feature_matrix = np.stack([_loop_feature_vector(frame) for frame in frames_pts], axis=0)
    velocity_matrix = np.zeros_like(feature_matrix)
    if len(frames_pts) >= 2:
        velocity_matrix[1:] = feature_matrix[1:] - feature_matrix[:-1]

    best = _find_best_loop_window(feature_matrix, velocity_matrix, fps)
    if best is None:
        return frames_pts, base_stats

    best_score = float(best["score"])
    best_start = int(best["start"])
    best_end = int(best["end"])
    max_reasonable_score = 1.25
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
    if not np.isfinite(best_score) or best_score > max_reasonable_score:
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
    normalized_mode = str(mode or "off").strip().lower()
    base_stats = {
        "applied": 0.0,
        "blend_frames": 0.0,
        "score_before": 0.0,
        "score_after": 0.0,
    }
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
        length = float(np.linalg.norm(offset))
        if length < 1e-6:
            length = 5.0

        if joint.name in {"spine", "chest", "upperChest", "neck", "head"}:
            direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        elif joint.name == "leftShoulder":
            direction = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        elif joint.name == "rightShoulder":
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        elif joint.name in {"leftUpperArm", "leftLowerArm"}:
            direction = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        elif joint.name in {"rightUpperArm", "rightLowerArm"}:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        elif joint.name in {"leftLowerLeg", "leftFoot"}:
            direction = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        elif joint.name in {"rightLowerLeg", "rightFoot"}:
            direction = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        elif joint.name in {"leftToes", "rightToes"}:
            direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            direction = offset

        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-6:
            direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            direction_norm = 1.0
        rest[joint.name] = (direction / direction_norm) * length

    return rest


def apply_skeleton_profile_to_rest_offsets(
    rest_offsets: Dict[str, np.ndarray],
    skeleton_profile: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    if not skeleton_profile:
        return rest_offsets, {"applied": 0.0, "overridden": 0.0}

    offsets_raw = skeleton_profile.get("joint_offsets") if isinstance(skeleton_profile, dict) else None
    if not isinstance(offsets_raw, dict):
        return rest_offsets, {"applied": 0.0, "overridden": 0.0}

    out = {name: value.copy() for name, value in rest_offsets.items()}
    raw_vectors: Dict[str, np.ndarray] = {}
    scale_ratios: List[float] = []
    for joint in JOINTS:
        if joint.parent is None:
            continue
        raw = offsets_raw.get(joint.name)
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            continue
        try:
            vec = np.array([float(raw[0]), float(raw[1]), float(raw[2])], dtype=np.float64)
        except Exception:
            continue
        if not np.all(np.isfinite(vec)) or np.linalg.norm(vec) < 1e-6:
            continue
        raw_vectors[joint.name] = vec
        rest_vec = rest_offsets.get(joint.name)
        if rest_vec is None:
            continue
        rest_len = float(np.linalg.norm(rest_vec))
        raw_len = float(np.linalg.norm(vec))
        if rest_len > 1e-6 and raw_len > 1e-6:
            scale_ratios.append(rest_len / raw_len)

    scale_ratio = 1.0
    if scale_ratios:
        scale_ratio = float(np.median(np.array(scale_ratios, dtype=np.float64)))
        if not np.isfinite(scale_ratio) or scale_ratio < 1e-6:
            scale_ratio = 1.0

    overridden = 0
    for joint_name, vec in raw_vectors.items():
        out[joint_name] = vec * scale_ratio
        overridden += 1

    return out, {
        "applied": 1.0 if overridden > 0 else 0.0,
        "overridden": float(overridden),
        "scale_ratio": float(scale_ratio),
        "model_label": str(skeleton_profile.get("modelLabel", "")) if isinstance(skeleton_profile, dict) else "",
        "model_fingerprint": str(skeleton_profile.get("modelFingerprint", "")) if isinstance(skeleton_profile, dict) else "",
    }


def _median_pose_sample(samples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if not samples:
        raise RuntimeError("No valid pose frames detected in video.")

    sample_keys = list(samples[0].keys())
    median_sample: Dict[str, np.ndarray] = {}
    for key in sample_keys:
        stacked = np.stack([sample[key] for sample in samples], axis=0)
        median_sample[key] = np.median(stacked, axis=0)
    return median_sample


def _build_reference_basis(
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
    # Keep the canonical "up" axis consistent with the observed torso
    # direction. Without this safeguard, some clips can rebuild a valid
    # right-handed basis whose Y axis points downward, flipping the whole
    # skeleton upside down.
    if float(np.dot(y_axis, up_hint)) < 0.0:
        y_axis = -y_axis
        z_axis = -z_axis

    basis = np.column_stack((x_axis, y_axis, z_axis))
    return basis, origin


def _rotate_points_about_y(
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


def _apply_pose_corrections(
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
        _rotate_points_about_y(out, upper_keys, corrections.upper_rotation_offset_deg, mid_hip)

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


def _canonicalize_pose_points(
    pts: Dict[str, np.ndarray],
    basis: np.ndarray,
    origin: np.ndarray,
) -> Dict[str, np.ndarray]:
    inv_basis = basis.T
    out: Dict[str, np.ndarray] = {}
    for key, value in pts.items():
        out[key] = inv_basis @ (np.array(value, dtype=np.float64) - origin)
    return out


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

    write_joint(JOINTS[0].name, 0)
    return lines


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

    def body_forward_vector(pts_cur: Dict[str, np.ndarray]) -> np.ndarray:
        mid_hip = pts_cur.get("mid_hip", np.zeros(3, dtype=np.float64))
        chest = pts_cur.get("chest", mid_hip + np.array([0.0, 1.0, 0.0], dtype=np.float64))
        nose = pts_cur.get("nose", chest)
        shoulder_side = pts_cur.get("right_shoulder", chest) - pts_cur.get("left_shoulder", chest)
        hip_side = pts_cur.get("right_hip", mid_hip) - pts_cur.get("left_hip", mid_hip)
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

    def secondary_vectors_for_joint(name: str, pts_cur: Dict[str, np.ndarray], rest: Dict[str, np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if name == "hips":
            chest = pts_cur.get("chest")
            mid_hip = pts_cur.get("mid_hip")
            if chest is not None and mid_hip is not None:
                return chest - mid_hip, np.array([0.0, 1.0, 0.0], dtype=np.float64)
            return None, None

        if name in {"leftUpperLeg", "rightUpperLeg", "leftLowerLeg", "rightLowerLeg", "leftFoot", "rightFoot"}:
            body_forward = body_forward_vector(pts_cur)
            if np.linalg.norm(body_forward) > 1e-6:
                return body_forward, np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return None, None

    def solve_joint(name: str) -> Tuple[float, float, float]:
        if name == root_name:
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
            global_rot[name] = np.array(
                [
                    [cos_y, 0.0, sin_y],
                    [0.0, 1.0, 0.0],
                    [-sin_y, 0.0, cos_y],
                ],
                dtype=np.float64,
            )
            return (0.0, 0.0, yaw_deg)

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

        cur_secondary, rest_secondary = secondary_vectors_for_joint(name, pts, rest_offsets)
        if cur_secondary is not None and rest_secondary is not None:
            r_align = rotation_align_with_secondary(rest_vec, cur_vec, rest_secondary, cur_secondary)
        else:
            r_align = rotation_align(rest_vec, cur_vec)
        r_local = parent_global.T @ r_align

        global_rot[name] = parent_global @ r_local
        return euler_zxy_from_matrix(r_local)

    rz, rx, ry = solve_joint(root_name)
    channels.extend([float(root_pos[0]), float(root_pos[1]), float(root_pos[2]), rz, rx, ry])

    for joint in JOINTS[1:]:
        rz, rx, ry = solve_joint(joint.name)
        channels.extend([rz, rx, ry])

    return channels


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

    # Fill leading/trailing gaps with nearest known frame.
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


@lru_cache(maxsize=16)
def gamma_lut(gamma: float) -> np.ndarray:
    g = max(0.1, min(4.0, float(gamma)))
    return np.array([((i / 255.0) ** g) * 255.0 for i in range(256)], dtype=np.uint8)


def resize_frame_for_detection(frame: np.ndarray, max_frame_side: int, cv2) -> np.ndarray:
    if max_frame_side <= 0:
        return frame
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_frame_side:
        return frame

    scale = max_frame_side / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def preprocess_video_frame(
    frame: np.ndarray,
    cv2,
    opencv_enhance: str,
    max_frame_side: int,
) -> np.ndarray:
    processed = resize_frame_for_detection(frame, max_frame_side, cv2)
    if opencv_enhance == "off":
        return processed

    if opencv_enhance == "light":
        bilateral_d = 5
        bilateral_sigma = 20
        clahe_clip_limit = 1.6
        gamma = 0.95
    else:
        bilateral_d = 7
        bilateral_sigma = 40
        clahe_clip_limit = 2.4
        gamma = 0.90

    # Denoise while preserving edges before landmark detection.
    processed = cv2.bilateralFilter(
        processed, d=bilateral_d, sigmaColor=bilateral_sigma, sigmaSpace=bilateral_sigma
    )

    # Improve local contrast on luminance channel for challenging lighting.
    lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a_channel, b_channel))
    processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if abs(gamma - 1.0) > 1e-6:
        processed = cv2.LUT(processed, gamma_lut(round(gamma, 4)))

    return processed


def _first_landmark_list(payload):
    if payload is None:
        return None
    if isinstance(payload, (list, tuple)):
        return payload[0] if payload else None
    if hasattr(payload, "landmark"):
        return payload.landmark
    return None


def extract_pose_bbox_pixels(res, frame_w: int, frame_h: int) -> Optional[Tuple[float, float, float, float]]:
    if frame_w <= 0 or frame_h <= 0:
        return None
    landmarks = _first_landmark_list(getattr(res, "pose_landmarks", None))
    if landmarks is None:
        return None

    xs: List[float] = []
    ys: List[float] = []
    for lm in landmarks:
        x = getattr(lm, "x", None)
        y = getattr(lm, "y", None)
        if x is None or y is None:
            continue
        x = float(x)
        y = float(y)
        if not np.isfinite(x) or not np.isfinite(y):
            continue

        vis = getattr(lm, "visibility", None)
        if vis is not None:
            vis = float(vis)
            if np.isfinite(vis) and vis < 0.05:
                continue

        xs.append(min(max(x, 0.0), 1.0))
        ys.append(min(max(y, 0.0), 1.0))

    if len(xs) < 4 or len(ys) < 4:
        return None

    min_x = min(xs) * frame_w
    max_x = max(xs) * frame_w
    min_y = min(ys) * frame_h
    max_y = max(ys) * frame_h
    if max_x - min_x < 2.0 or max_y - min_y < 2.0:
        return None
    return (min_x, min_y, max_x, max_y)


def _should_fallback_to_legacy_pose(exc: Exception) -> bool:
    text = str(exc)
    return any(
        token in text
        for token in (
            "NSOpenGLPixelFormat",
            "kGpuService",
            "gl_context_nsgl",
            "Could not create an NSOpenGLPixelFormat",
        )
    )


def _create_pose_detector(
    model_complexity: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    cv2,
) -> Tuple[Callable[[np.ndarray, int], Any], Callable[[], None], str]:
    import mediapipe as mp

    try:
        from mediapipe.tasks import python as mp_tasks_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = ensure_pose_model(model_complexity)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks_python.BaseOptions(
                model_asset_path=str(model_path),
                delegate=mp_tasks_python.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        pose = mp_vision.PoseLandmarker.create_from_options(options)

        def detect_pose(frame_bgr: np.ndarray, ts_ms: int):
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            return pose.detect_for_video(mp_image, ts_ms)

        return detect_pose, pose.close, "tasks"
    except Exception as exc:
        if not _should_fallback_to_legacy_pose(exc):
            raise
        print(
            f"[vid2model] pose_backend tasks failed, falling back to mediapipe.solutions.pose: {exc}",
            file=sys.stderr,
        )

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=max(0, min(int(model_complexity), 2)),
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    def detect_pose(frame_bgr: np.ndarray, ts_ms: int):
        del ts_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return pose.process(rgb)

    return detect_pose, pose.close, "solutions"


def clamp_roi_box(
    roi: Tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
    min_side: float,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = roi
    x0 = max(0.0, min(float(frame_w), x0))
    y0 = max(0.0, min(float(frame_h), y0))
    x1 = max(0.0, min(float(frame_w), x1))
    y1 = max(0.0, min(float(frame_h), y1))
    if x1 <= x0:
        x1 = min(float(frame_w), x0 + min_side)
    if y1 <= y0:
        y1 = min(float(frame_h), y0 + min_side)

    w = x1 - x0
    h = y1 - y0
    if w < min_side:
        cx = (x0 + x1) * 0.5
        half = min_side * 0.5
        x0 = max(0.0, cx - half)
        x1 = min(float(frame_w), cx + half)
    if h < min_side:
        cy = (y0 + y1) * 0.5
        half = min_side * 0.5
        y0 = max(0.0, cy - half)
        y1 = min(float(frame_h), cy + half)

    x0i = int(np.floor(max(0.0, min(x0, float(frame_w - 1)))))
    y0i = int(np.floor(max(0.0, min(y0, float(frame_h - 1)))))
    x1i = int(np.ceil(max(float(x0i + 1), min(x1, float(frame_w)))))
    y1i = int(np.ceil(max(float(y0i + 1), min(y1, float(frame_h)))))
    return (x0i, y0i, x1i, y1i)


def update_tracking_roi(
    prev_roi: Optional[Tuple[int, int, int, int]],
    detected_bbox: Tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
) -> Tuple[int, int, int, int]:
    min_side = max(96.0, min(frame_w, frame_h) * 0.20)
    min_x, min_y, max_x, max_y = detected_bbox
    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5
    side = max(max_x - min_x, max_y - min_y)
    side = max(side * 1.9, min_side)
    target = (
        cx - side * 0.5,
        cy - side * 0.5,
        cx + side * 0.5,
        cy + side * 0.5,
    )

    if prev_roi is not None:
        alpha = 0.60
        target = (
            prev_roi[0] * alpha + target[0] * (1.0 - alpha),
            prev_roi[1] * alpha + target[1] * (1.0 - alpha),
            prev_roi[2] * alpha + target[2] * (1.0 - alpha),
            prev_roi[3] * alpha + target[3] * (1.0 - alpha),
        )

    return clamp_roi_box(target, frame_w, frame_h, min_side=min_side)


def collect_detected_pose_samples(
    input_path: Path,
    model_complexity: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    progress_every: int = 100,
    opencv_enhance: str = "off",
    max_frame_side: int = 0,
    roi_crop: str = "off",
) -> Tuple[float, List[Optional[Dict[str, np.ndarray]]], List[Dict[str, np.ndarray]], Dict[str, Any]]:
    import cv2

    opencv_enhance = str(opencv_enhance).strip().lower()
    if opencv_enhance not in {"off", "light", "strong"}:
        raise ValueError("opencv_enhance must be one of: off, light, strong")
    if max_frame_side < 0:
        raise ValueError("max_frame_side must be >= 0")
    roi_crop = str(roi_crop).strip().lower()
    if roi_crop not in {"off", "auto"}:
        raise ValueError("roi_crop must be one of: off, auto")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-6:
        fps = 30.0

    detect_pose, close_pose, pose_backend = _create_pose_detector(
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        cv2=cv2,
    )
    print(f"[vid2model] pose_backend={pose_backend}", file=sys.stderr)

    frames_pts_raw: List[Optional[Dict[str, np.ndarray]]] = []
    detected_samples: List[Dict[str, np.ndarray]] = []
    detected_count = 0
    roi_state: Optional[Tuple[int, int, int, int]] = None
    roi_used_count = 0
    roi_fallback_count = 0
    roi_reset_count = 0
    if opencv_enhance != "off" or max_frame_side > 0:
        print(
            f"[vid2model] opencv_preprocess enhance={opencv_enhance} max_frame_side={max_frame_side}",
            file=sys.stderr,
        )
    if roi_crop == "auto":
        print("[vid2model] roi_crop mode=auto", file=sys.stderr)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_for_pose = preprocess_video_frame(frame, cv2, opencv_enhance, max_frame_side)
        ts_ms = int((frame_idx * 1000.0) / fps)
        frame_h, frame_w = frame_for_pose.shape[:2]

        res = None
        pts = None
        bbox_for_roi: Optional[Tuple[float, float, float, float]] = None
        used_roi = False
        if roi_crop == "auto" and roi_state is not None:
            x0, y0, x1, y1 = roi_state
            roi_frame = frame_for_pose[y0:y1, x0:x1]
            if roi_frame.size > 0:
                used_roi = True
                roi_used_count += 1
                res = detect_pose(roi_frame, ts_ms)
                pts = extract_pose_points(res)
                roi_bbox = extract_pose_bbox_pixels(res, roi_frame.shape[1], roi_frame.shape[0])
                if roi_bbox is not None:
                    bbox_for_roi = (
                        roi_bbox[0] + x0,
                        roi_bbox[1] + y0,
                        roi_bbox[2] + x0,
                        roi_bbox[3] + y0,
                    )
            if pts is None:
                roi_fallback_count += 1
                used_roi = False
                res = detect_pose(frame_for_pose, ts_ms + 1)
                pts = extract_pose_points(res)
                bbox_for_roi = extract_pose_bbox_pixels(res, frame_w, frame_h)
        else:
            res = detect_pose(frame_for_pose, ts_ms)
            pts = extract_pose_points(res)
            bbox_for_roi = extract_pose_bbox_pixels(res, frame_w, frame_h)

        if roi_crop == "auto":
            if pts is not None and bbox_for_roi is not None:
                roi_state = update_tracking_roi(roi_state, bbox_for_roi, frame_w, frame_h)
            elif pts is None and roi_state is not None and not used_roi:
                roi_state = None
                roi_reset_count += 1

        frame_idx += 1
        frames_pts_raw.append(pts)
        if pts is not None:
            detected_count += 1
            if len(detected_samples) < 60:
                detected_samples.append(pts)

        if progress_every > 0 and frame_idx % progress_every == 0:
            print(
                f"[vid2model] processed={frame_idx} detected={detected_count} miss={frame_idx - detected_count}",
                file=sys.stderr,
            )

    cap.release()
    close_pose()

    if roi_crop == "auto":
        print(
            (
                f"[vid2model] roi_stats used={roi_used_count} "
                f"fallback_full={roi_fallback_count} resets={roi_reset_count}"
            ),
            file=sys.stderr,
        )

    stats = {
        "frames": frame_idx,
        "detected": detected_count,
        "roi_used": roi_used_count,
        "roi_fallback": roi_fallback_count,
        "roi_resets": roi_reset_count,
        "pose_backend": pose_backend,
    }
    return fps, frames_pts_raw, detected_samples, stats


def convert_video_to_bvh(
    input_path: Path,
    model_complexity: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    max_gap_interpolate: int = 8,
    progress_every: int = 100,
    opencv_enhance: str = "off",
    max_frame_side: int = 0,
    roi_crop: str = "off",
    pose_corrections: Optional[PoseCorrectionProfile] = None,
    skeleton_profile: Optional[Dict[str, Any]] = None,
    root_yaw_offset_deg: float = 0.0,
    lower_body_rotation_mode: str = "off",
    loop_mode: str = "off",
) -> Tuple[float, Dict[str, np.ndarray], List[List[float]], np.ndarray, List[Dict[str, np.ndarray]], Dict[str, Any]]:
    fps, frames_pts_raw, detected_samples, scan_stats = collect_detected_pose_samples(
        input_path=input_path,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        progress_every=progress_every,
        opencv_enhance=opencv_enhance,
        max_frame_side=max_frame_side,
        roi_crop=roi_crop,
    )
    detected_count = scan_stats["detected"]
    roi_used_count = scan_stats["roi_used"]
    roi_fallback_count = scan_stats["roi_fallback"]
    roi_reset_count = scan_stats["roi_resets"]
    pose_backend = str(scan_stats.get("pose_backend", "unknown"))

    frames_pts, interpolated_frames, carried_frames = fill_pose_gaps(frames_pts_raw, max_gap_interpolate)
    print(
        (
            f"[vid2model] done frames={len(frames_pts_raw)} detected={detected_count} "
            f"interpolated={interpolated_frames} carried={carried_frames}"
        ),
        file=sys.stderr,
    )

    anchor_samples = detected_samples if detected_samples else frames_pts[:20]
    reference_sample = _median_pose_sample(anchor_samples)
    corrections = pose_corrections or DEFAULT_POSE_CORRECTIONS
    auto_meta: Dict[str, Any] = {}
    if corrections.mode == "auto":
        corrections, auto_meta = resolve_auto_pose_corrections(anchor_samples, corrections)
        print(
            f"[vid2model] pose_corrections auto label={auto_meta.get('label', 'default')} "
            f"score={auto_meta.get('heuristic_score', auto_meta.get('model_score', 0.0))}",
            file=sys.stderr,
        )
    reference_basis, reference_origin = _build_reference_basis(reference_sample, corrections)
    frames_pts = [_canonicalize_pose_points(pts, reference_basis, reference_origin) for pts in frames_pts]
    canonical_anchor_samples = [
        _canonicalize_pose_points(sample, reference_basis, reference_origin) for sample in anchor_samples
    ]
    frames_pts = [_apply_pose_corrections(pts, corrections) for pts in frames_pts]
    canonical_anchor_samples = [_apply_pose_corrections(sample, corrections) for sample in canonical_anchor_samples]
    pre_cleanup_loopability: Dict[str, Any] = {}
    contact_cleanup_enabled = True
    if len(frames_pts) >= 12:
        preview_smoothed = _smooth_pose_frames(frames_pts, alpha=0.35)
        preview_target_lengths = _build_target_segment_lengths(canonical_anchor_samples or preview_smoothed[:20])
        preview_constrained = [
            _apply_segment_length_constraints(frame, preview_target_lengths) for frame in preview_smoothed
        ]
        pre_cleanup_loopability = analyze_motion_loopability(preview_constrained, fps)
        if str(pre_cleanup_loopability.get("label", "oneshot")) == "oneshot":
            contact_cleanup_enabled = False
        print(
            (
                f"[vid2model] cleanup_mode contact={'on' if contact_cleanup_enabled else 'off'} "
                f"motion={str(pre_cleanup_loopability.get('label', 'unknown'))} "
                f"score={float(pre_cleanup_loopability.get('score', 0.0)):.3f}"
            ),
            file=sys.stderr,
        )
    frames_pts, cleanup_stats = cleanup_pose_frames(
        frames_pts,
        canonical_anchor_samples,
        use_contact_cleanup=contact_cleanup_enabled,
    )
    canonical_anchor_samples, _ = cleanup_pose_frames(
        canonical_anchor_samples,
        canonical_anchor_samples,
        use_contact_cleanup=False,
    )
    print(
        (
            f"[vid2model] source_cleanup side_swaps={int(cleanup_stats['side_swaps'])} "
            f"smooth_alpha={cleanup_stats['smooth_alpha']:.2f} "
            f"segments={int(cleanup_stats['length_constraints'])} "
            f"foot_contacts={int(cleanup_stats['contact_windows'])}/{int(cleanup_stats['contact_frames'])} "
            f"pelvis_frames={int(cleanup_stats['pelvis_contact_frames'])} "
            f"leg_ik_frames={int(cleanup_stats['leg_ik_frames'])}"
        ),
        file=sys.stderr,
    )
    loopability: Dict[str, Any] = {}
    if str(loop_mode or "off").strip().lower() == "auto":
        loopability = analyze_motion_loopability(frames_pts, fps)
        print(
            (
                f"[vid2model] loop_detect label={str(loopability['label'])} "
                f"score={float(loopability['score']):.3f} "
                f"coverage={float(loopability['coverage']):.2f}"
            ),
            file=sys.stderr,
        )
    frames_pts, loop_stats = extract_motion_loop(frames_pts, fps, loop_mode)
    if loop_stats["applied"] > 0.5:
        canonical_anchor_samples = frames_pts[: min(len(frames_pts), 20)]
        print(
            (
                f"[vid2model] loop_extract start={int(loop_stats['start'])} "
                f"end={int(loop_stats['end'])} "
                f"frames={int(loop_stats['length'])} "
                f"score={loop_stats['score']:.3f}"
            ),
            file=sys.stderr,
        )
        frames_pts, blend_stats = blend_motion_loop_edges(frames_pts, fps, loop_mode)
        if blend_stats["applied"] > 0.5:
            canonical_anchor_samples = frames_pts[: min(len(frames_pts), 20)]
            print(
                (
                    f"[vid2model] loop_blend frames={int(blend_stats['blend_frames'])} "
                    f"score_before={blend_stats['score_before']:.3f} "
                    f"score_after={blend_stats['score_after']:.3f}"
                ),
                file=sys.stderr,
            )

    rest_offsets = build_rest_offsets(canonical_anchor_samples)
    rest_offsets, skeleton_profile_stats = apply_skeleton_profile_to_rest_offsets(rest_offsets, skeleton_profile)
    if skeleton_profile_stats["applied"] > 0.5:
        print(
            (
                f"[vid2model] skeleton_profile overridden={int(skeleton_profile_stats['overridden'])} "
                f"model={str(skeleton_profile_stats.get('model_label', '')) or 'unknown'}"
            ),
            file=sys.stderr,
        )
    root_point_name = MAP_TO_POINTS[JOINTS[0].name][0]
    ref_root = frames_pts[0][root_point_name].copy()

    motion_values = []
    for pts in frames_pts:
        motion_values.append(frame_channels(pts, rest_offsets, ref_root, corrections))
    motion_values, yaw_norm_stats = normalize_motion_root_yaw(motion_values)
    if yaw_norm_stats["applied"] > 0.5:
        print(
            (
                f"[vid2model] root_yaw normalized offset={yaw_norm_stats['offset_deg']:.0f} "
                f"near0={int(yaw_norm_stats['near0'])} near180={int(yaw_norm_stats['near180'])}"
            ),
            file=sys.stderr,
        )
    if np.isfinite(root_yaw_offset_deg) and abs(float(root_yaw_offset_deg)) > 1e-6:
        motion_values = apply_manual_root_yaw_offset(motion_values, float(root_yaw_offset_deg))
        print(f"[vid2model] root_yaw manual_offset={float(root_yaw_offset_deg):.0f}", file=sys.stderr)
    if str(lower_body_rotation_mode or "off").strip().lower() != "off":
        motion_values = apply_lower_body_rotation_mode(motion_values, lower_body_rotation_mode)
        print(f"[vid2model] lower_body_rotation mode={str(lower_body_rotation_mode).strip().lower()}", file=sys.stderr)
    motion_values, unwrap_stats = unwrap_motion_rotation_channels(motion_values)
    if unwrap_stats["applied"] > 0.5:
        print(
            (
                f"[vid2model] rotation_unwrap changed={int(unwrap_stats['changed_values'])} "
                f"max_step_before={unwrap_stats['max_step_before']:.1f} "
                f"max_step_after={unwrap_stats['max_step_after']:.1f}"
            ),
            file=sys.stderr,
        )

    diagnostics: Dict[str, Any] = {
        "input": {
            "fps": float(fps),
            "source_frame_count": int(len(frames_pts_raw)),
            "detected_frames": int(detected_count),
            "interpolated_frames": int(interpolated_frames),
            "carried_frames": int(carried_frames),
            "roi_used": int(roi_used_count),
            "roi_fallback": int(roi_fallback_count),
            "roi_resets": int(roi_reset_count),
            "pose_backend": pose_backend,
        },
        "cleanup": {key: float(value) for key, value in cleanup_stats.items()},
        "root_yaw": {
            "normalized_applied": float(yaw_norm_stats["applied"]),
            "normalized_offset_deg": float(yaw_norm_stats["offset_deg"]),
            "manual_offset_deg": float(root_yaw_offset_deg),
        },
        "skeleton_profile": {
            key: float(value) if isinstance(value, (int, float)) else value
            for key, value in skeleton_profile_stats.items()
        },
        "rotation_unwrap": {key: float(value) for key, value in unwrap_stats.items()},
        "loop": {
            "mode": str(loop_mode or "off"),
            "pre_cleanup_detected": pre_cleanup_loopability,
            "detected": loopability,
            "extracted": {key: float(value) for key, value in loop_stats.items()},
        },
        "output": {
            "frame_count": int(len(motion_values)),
        },
    }

    return fps, rest_offsets, motion_values, ref_root, frames_pts, diagnostics
