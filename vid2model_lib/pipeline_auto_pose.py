from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


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
    upper_body_rotation_scale: float = 1.0
    arm_rotation_scale: float = 1.0
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
AUTO_FEATURE_NAMES: tuple[str, ...] = (
    "shoulder_width",
    "hip_width",
    "torso_height",
    "head_height",
    "arm_span",
    "leg_span",
    "torso_depth",
    "head_depth",
    "hand_depth",
    "arm_width_ratio",
    "leg_width_ratio",
    "torso_depth_ratio",
    "head_depth_ratio",
    "hand_drop_ratio",
    "crouch_ratio",
    "knee_drop_ratio",
    "shoulder_bias",
    "hip_bias",
    "wrist_bias",
    "ankle_bias",
    "shoulder_depth_bias",
    "hip_depth_bias",
    "arm_span_to_torso",
    "leg_span_to_torso",
    "head_to_torso",
)


def _as_bool(value: Any, default: bool) -> bool:
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


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_str(value: Any, default: str) -> str:
    if value is None:
        return default
    return str(value)


def build_pose_correction_profile(data: Optional[Dict[str, Any]] = None) -> PoseCorrectionProfile:
    if not data:
        return DEFAULT_POSE_CORRECTIONS

    return PoseCorrectionProfile(
        mode=str(data.get("mode", DEFAULT_POSE_CORRECTIONS.mode)).strip().lower() or DEFAULT_POSE_CORRECTIONS.mode,
        model_path=_as_str(data.get("model_path"), DEFAULT_POSE_CORRECTIONS.model_path),
        shoulder_tracking=_as_bool(data.get("shoulder_tracking"), DEFAULT_POSE_CORRECTIONS.shoulder_tracking),
        hip_camera=_as_bool(data.get("hip_camera"), DEFAULT_POSE_CORRECTIONS.hip_camera),
        auto_grounding=_as_bool(data.get("auto_grounding"), DEFAULT_POSE_CORRECTIONS.auto_grounding),
        use_arm_ik=_as_bool(data.get("use_arm_ik"), DEFAULT_POSE_CORRECTIONS.use_arm_ik),
        use_leg_ik=_as_bool(data.get("use_leg_ik"), DEFAULT_POSE_CORRECTIONS.use_leg_ik),
        body_bend_reduction_power=_as_float(
            data.get("body_bend_reduction_power"), DEFAULT_POSE_CORRECTIONS.body_bend_reduction_power
        ),
        upper_rotation_offset_deg=_as_float(
            data.get("upper_rotation_offset_deg", data.get("upper_rotation_offset")),
            DEFAULT_POSE_CORRECTIONS.upper_rotation_offset_deg,
        ),
        upper_body_rotation_scale=_as_float(
            data.get("upper_body_rotation_scale"),
            DEFAULT_POSE_CORRECTIONS.upper_body_rotation_scale,
        ),
        arm_rotation_scale=_as_float(
            data.get("arm_rotation_scale"),
            DEFAULT_POSE_CORRECTIONS.arm_rotation_scale,
        ),
        arm_horizontal_offset_percent=_as_float(
            data.get("arm_horizontal_offset_percent"), DEFAULT_POSE_CORRECTIONS.arm_horizontal_offset_percent
        ),
        arm_vertical_offset_percent=_as_float(
            data.get("arm_vertical_offset_percent"), DEFAULT_POSE_CORRECTIONS.arm_vertical_offset_percent
        ),
        hip_depth_scale_percent=_as_float(
            data.get("hip_depth_scale_percent"), DEFAULT_POSE_CORRECTIONS.hip_depth_scale_percent
        ),
        hip_y_position_offset_percent=_as_float(
            data.get("hip_y_position_offset_percent"), DEFAULT_POSE_CORRECTIONS.hip_y_position_offset_percent
        ),
        hip_z_position_offset_percent=_as_float(
            data.get("hip_z_position_offset_percent"), DEFAULT_POSE_CORRECTIONS.hip_z_position_offset_percent
        ),
        body_collider_mode=_as_int(data.get("body_collider_mode"), DEFAULT_POSE_CORRECTIONS.body_collider_mode),
        body_collider_head_size_percent=_as_float(
            data.get("body_collider_head_size_percent"), DEFAULT_POSE_CORRECTIONS.body_collider_head_size_percent
        ),
        body_collider_chest_size_percent=_as_float(
            data.get("body_collider_chest_size_percent"), DEFAULT_POSE_CORRECTIONS.body_collider_chest_size_percent
        ),
        body_collider_waist_size_percent=_as_float(
            data.get("body_collider_waist_size_percent"), DEFAULT_POSE_CORRECTIONS.body_collider_waist_size_percent
        ),
        body_collider_hip_size_percent=_as_float(
            data.get("body_collider_hip_size_percent"), DEFAULT_POSE_CORRECTIONS.body_collider_hip_size_percent
        ),
        body_collider_head_reaction_type=_as_str(
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
    # Person facing camera directly: frontal shot, standing still, arms active.
    # Typical for sign language, gesture, or interview capture.
    "frontal_standing": {
        "shoulder_tracking": True,
        "hip_camera": False,
        "auto_grounding": False,
        "use_leg_ik": False,
        "body_bend_reduction_power": 0.5,
        "upper_body_rotation_scale": 0.35,
        "arm_rotation_scale": 1.0,
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


def _auto_feature_vector(
    samples: List[Dict[str, np.ndarray]],
    median_pose_sample: Callable[[List[Dict[str, np.ndarray]]], Dict[str, np.ndarray]],
) -> Tuple[np.ndarray, Dict[str, float]]:
    summary = _sample_feature_summary(median_pose_sample(samples))
    vector = np.array([summary[name] for name in AUTO_FEATURE_NAMES], dtype=np.float64)
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
        "frontal_standing": 0.0,
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
    # Frontal standing: person faces camera (low lateral bias), not crouched, no depth skew
    _shoulder_bias_abs = abs(summary.get("shoulder_bias", 1.0))
    _hip_bias_abs = abs(summary.get("hip_bias", 1.0))
    _depth_bias = abs(summary.get("torso_depth_ratio", 1.0))
    if _shoulder_bias_abs < 0.08 and _hip_bias_abs < 0.08 and _depth_bias < 0.08:
        scores["frontal_standing"] += 3.0
    if _shoulder_bias_abs < 0.12 and _hip_bias_abs < 0.12:
        scores["frontal_standing"] += 1.0
    label = max(scores.items(), key=lambda item: item[1])[0]
    if scores[label] <= 0.5:
        label = "default"
    return label, {"heuristic_score": float(scores[label])}


def _apply_pose_preset(base: PoseCorrectionProfile, preset: Dict[str, Any]) -> PoseCorrectionProfile:
    resolved = replace(base, mode="resolved")
    for key, value in preset.items():
        if hasattr(resolved, key):
            resolved = replace(resolved, **{key: value})
    return resolved


def resolve_auto_pose_corrections(
    samples: List[Dict[str, np.ndarray]],
    base: PoseCorrectionProfile,
    median_pose_sample: Callable[[List[Dict[str, np.ndarray]]], Dict[str, np.ndarray]],
) -> Tuple[PoseCorrectionProfile, Dict[str, Any]]:
    features, summary = _auto_feature_vector(samples, median_pose_sample)
    label, meta = _predict_auto_label(features, summary, base.model_path)
    preset = AUTO_POSE_PRESETS.get(label, AUTO_POSE_PRESETS["default"])
    resolved = _apply_pose_preset(base, preset)
    meta.update(
        {
            "label": label,
            "features": {k: round(float(v), 6) for k, v in summary.items()},
        }
    )
    return resolved, meta
