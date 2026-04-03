from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .skeleton import JOINTS, MAP_TO_POINTS


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


def _skeleton_profile_scale_group(joint_name: str) -> str:
    if joint_name in {"spine", "chest", "upperChest", "neck", "head", "leftShoulder", "rightShoulder"}:
        return "torso"
    if joint_name.startswith("leftUpperArm") or joint_name.startswith("leftLowerArm"):
        return "left_arm"
    if joint_name.startswith("rightUpperArm") or joint_name.startswith("rightLowerArm"):
        return "right_arm"
    if (
        joint_name.startswith("leftHand")
        or joint_name.startswith("leftThumb")
        or joint_name.startswith("leftIndex")
        or joint_name.startswith("leftMiddle")
        or joint_name.startswith("leftRing")
        or joint_name.startswith("leftLittle")
    ):
        return "left_hand"
    if (
        joint_name.startswith("rightHand")
        or joint_name.startswith("rightThumb")
        or joint_name.startswith("rightIndex")
        or joint_name.startswith("rightMiddle")
        or joint_name.startswith("rightRing")
        or joint_name.startswith("rightLittle")
    ):
        return "right_hand"
    if (
        joint_name.startswith("leftUpperLeg")
        or joint_name.startswith("leftLowerLeg")
        or joint_name.startswith("leftFoot")
        or joint_name.startswith("leftToes")
    ):
        return "left_leg"
    if (
        joint_name.startswith("rightUpperLeg")
        or joint_name.startswith("rightLowerLeg")
        or joint_name.startswith("rightFoot")
        or joint_name.startswith("rightToes")
    ):
        return "right_leg"
    return "global"


VRM_HUMANOID_BASELINE_CHAINS: Dict[str, Tuple[List[str], List[float]]] = {
    "torso": (
        ["spine", "chest", "upperChest", "neck", "head"],
        [0.18, 0.18, 0.22, 0.14, 0.28],
    ),
    "left_arm": (
        ["leftShoulder", "leftUpperArm", "leftLowerArm", "leftHand"],
        [0.10, 0.38, 0.34, 0.18],
    ),
    "right_arm": (
        ["rightShoulder", "rightUpperArm", "rightLowerArm", "rightHand"],
        [0.10, 0.38, 0.34, 0.18],
    ),
    "left_leg": (
        ["leftUpperLeg", "leftLowerLeg", "leftFoot", "leftToes"],
        [0.44, 0.40, 0.11, 0.05],
    ),
    "right_leg": (
        ["rightUpperLeg", "rightLowerLeg", "rightFoot", "rightToes"],
        [0.44, 0.40, 0.11, 0.05],
    ),
}


def _apply_vrm_humanoid_baseline_to_rest_offsets(
    rest_offsets: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    out = {name: value.copy() for name, value in rest_offsets.items()}
    applied_chains: List[str] = []
    for chain_name, (joint_names, weights) in VRM_HUMANOID_BASELINE_CHAINS.items():
        vectors = [out.get(joint_name) for joint_name in joint_names]
        if any(vec is None for vec in vectors):
            continue
        lengths = [float(np.linalg.norm(vec)) for vec in vectors if vec is not None]
        total_length = float(sum(lengths))
        weight_sum = float(sum(weights))
        if total_length <= 1e-6 or weight_sum <= 1e-6:
            continue
        for joint_name, weight in zip(joint_names, weights):
            current = out.get(joint_name)
            if current is None:
                continue
            current_len = float(np.linalg.norm(current))
            if current_len > 1e-6:
                direction = current / current_len
            else:
                direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            out[joint_name] = direction * (total_length * (float(weight) / weight_sum))
        applied_chains.append(chain_name)
    return out, {
        "humanoid_baseline_applied": 1.0 if applied_chains else 0.0,
        "humanoid_baseline_chains": applied_chains,
    }


def apply_skeleton_profile_to_rest_offsets(
    rest_offsets: Dict[str, np.ndarray],
    skeleton_profile: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    if not skeleton_profile:
        return rest_offsets, {"applied": 0.0, "overridden": 0.0}

    offsets_raw = skeleton_profile.get("joint_offsets") if isinstance(skeleton_profile, dict) else None
    blend_raw = skeleton_profile.get("joint_blend") if isinstance(skeleton_profile, dict) else None
    normalize_to_vrm = False
    if isinstance(skeleton_profile, dict):
        normalize_to_vrm = bool(skeleton_profile.get("normalize_to_vrm_humanoid", False))
    if not isinstance(offsets_raw, dict):
        if normalize_to_vrm:
            normalized, baseline_stats = _apply_vrm_humanoid_baseline_to_rest_offsets(rest_offsets)
            baseline_stats.update(
                {
                    "applied": 0.0,
                    "overridden": 0.0,
                    "scale_ratio": 1.0,
                    "group_scale_ratios": {},
                    "avg_blend": 0.0,
                    "model_label": str(skeleton_profile.get("modelLabel", "")) if isinstance(skeleton_profile, dict) else "",
                    "model_fingerprint": str(skeleton_profile.get("modelFingerprint", "")) if isinstance(skeleton_profile, dict) else "",
                }
            )
            return normalized, baseline_stats
        return rest_offsets, {"applied": 0.0, "overridden": 0.0}

    out = {name: value.copy() for name, value in rest_offsets.items()}
    raw_vectors: Dict[str, np.ndarray] = {}
    scale_ratios: List[float] = []
    group_scale_ratios: Dict[str, List[float]] = {}
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
            ratio = rest_len / raw_len
            scale_ratios.append(ratio)
            group_scale_ratios.setdefault(_skeleton_profile_scale_group(joint.name), []).append(ratio)

    scale_ratio = 1.0
    if scale_ratios:
        scale_ratio = float(np.median(np.array(scale_ratios, dtype=np.float64)))
        if not np.isfinite(scale_ratio) or scale_ratio < 1e-6:
            scale_ratio = 1.0

    resolved_group_scale_ratios: Dict[str, float] = {}
    for group_name, ratios in group_scale_ratios.items():
        if not ratios:
            continue
        group_ratio = float(np.median(np.array(ratios, dtype=np.float64)))
        if not np.isfinite(group_ratio) or group_ratio < 1e-6:
            group_ratio = scale_ratio
        resolved_group_scale_ratios[group_name] = group_ratio

    overridden = 0
    blend_sum = 0.0
    for joint_name, vec in raw_vectors.items():
        joint_scale_ratio = resolved_group_scale_ratios.get(_skeleton_profile_scale_group(joint_name), scale_ratio)
        scaled = vec * joint_scale_ratio
        blend = 1.0
        if isinstance(blend_raw, dict):
            raw_blend = blend_raw.get(joint_name)
            if isinstance(raw_blend, (int, float)) and np.isfinite(float(raw_blend)):
                blend = float(np.clip(float(raw_blend), 0.0, 1.0))
        rest_vec = rest_offsets.get(joint_name, scaled)
        out[joint_name] = rest_vec * (1.0 - blend) + scaled * blend
        overridden += 1
        blend_sum += blend

    stats = {
        "applied": 1.0 if overridden > 0 else 0.0,
        "overridden": float(overridden),
        "scale_ratio": float(scale_ratio),
        "group_scale_ratios": {
            key: float(value) for key, value in sorted(resolved_group_scale_ratios.items())
        },
        "avg_blend": float(blend_sum / overridden) if overridden > 0 else 0.0,
        "model_label": str(skeleton_profile.get("modelLabel", "")) if isinstance(skeleton_profile, dict) else "",
        "model_fingerprint": str(skeleton_profile.get("modelFingerprint", "")) if isinstance(skeleton_profile, dict) else "",
    }
    if normalize_to_vrm:
        out, baseline_stats = _apply_vrm_humanoid_baseline_to_rest_offsets(out)
        stats.update(baseline_stats)
    else:
        stats.update(
            {
                "humanoid_baseline_applied": 0.0,
                "humanoid_baseline_chains": [],
            }
        )
    return out, stats
