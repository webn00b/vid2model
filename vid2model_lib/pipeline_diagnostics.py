from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _frame_step_energy(frames_pts: List[Dict[str, np.ndarray]], key: str, axes: tuple[int, ...] = (0, 1, 2)) -> float:
    if len(frames_pts) < 2:
        return 0.0
    steps = []
    for idx in range(1, len(frames_pts)):
        current = np.array(frames_pts[idx].get(key, np.zeros(3, dtype=np.float64)), dtype=np.float64)
        previous = np.array(frames_pts[idx - 1].get(key, np.zeros(3, dtype=np.float64)), dtype=np.float64)
        steps.append(float(np.linalg.norm((current - previous)[list(axes)])))
    return float(np.mean(np.array(steps, dtype=np.float64))) if steps else 0.0


def _foot_contact_spread_metric(frames_pts: List[Dict[str, np.ndarray]], side: str) -> float:
    if not frames_pts:
        return 0.0
    ankle_key = f"{side}_ankle"
    toes_key = f"{side}_toes"
    heel_key = f"{side}_heel"
    centroids = []
    for frame in frames_pts:
        ankle = np.array(frame.get(ankle_key, np.zeros(3, dtype=np.float64)), dtype=np.float64)
        toes = np.array(frame.get(toes_key, ankle), dtype=np.float64)
        heel = np.array(frame.get(heel_key, ankle), dtype=np.float64)
        centroid = (ankle[[0, 2]] + toes[[0, 2]] + heel[[0, 2]]) / 3.0
        centroids.append(centroid)
    if not centroids:
        return 0.0
    stacked = np.stack(centroids, axis=0)
    spread = np.max(stacked, axis=0) - np.min(stacked, axis=0)
    return float(np.linalg.norm(spread))


def _ratio_delta(before: float, after: float, invert: bool = False) -> Dict[str, float]:
    before = float(before)
    after = float(after)
    if before <= 1e-8:
        improvement = 0.0 if abs(after - before) <= 1e-8 else (-1.0 if invert else 1.0)
    else:
        raw = (after - before) / before
        improvement = -raw if invert else raw
    return {
        "before": before,
        "after": after,
        "delta": after - before,
        "improvement_ratio": float(improvement),
    }


def build_cleanup_evaluation(
    frames_before_cleanup: List[Dict[str, np.ndarray]],
    frames_after_cleanup: List[Dict[str, np.ndarray]],
) -> Dict[str, Any]:
    if not frames_before_cleanup or not frames_after_cleanup:
        return {
            "root_position_jitter": _ratio_delta(0.0, 0.0, invert=True),
            "root_height_jitter": _ratio_delta(0.0, 0.0, invert=True),
            "left_foot_contact_spread": _ratio_delta(0.0, 0.0, invert=True),
            "right_foot_contact_spread": _ratio_delta(0.0, 0.0, invert=True),
            "left_wrist_motion_energy": _ratio_delta(0.0, 0.0, invert=False),
            "right_wrist_motion_energy": _ratio_delta(0.0, 0.0, invert=False),
        }

    return {
        "root_position_jitter": _ratio_delta(
            _frame_step_energy(frames_before_cleanup, "mid_hip", axes=(0, 2)),
            _frame_step_energy(frames_after_cleanup, "mid_hip", axes=(0, 2)),
            invert=True,
        ),
        "root_height_jitter": _ratio_delta(
            _frame_step_energy(frames_before_cleanup, "mid_hip", axes=(1,)),
            _frame_step_energy(frames_after_cleanup, "mid_hip", axes=(1,)),
            invert=True,
        ),
        "left_foot_contact_spread": _ratio_delta(
            _foot_contact_spread_metric(frames_before_cleanup, "left"),
            _foot_contact_spread_metric(frames_after_cleanup, "left"),
            invert=True,
        ),
        "right_foot_contact_spread": _ratio_delta(
            _foot_contact_spread_metric(frames_before_cleanup, "right"),
            _foot_contact_spread_metric(frames_after_cleanup, "right"),
            invert=True,
        ),
        "left_wrist_motion_energy": _ratio_delta(
            _frame_step_energy(frames_before_cleanup, "left_wrist", axes=(0, 1, 2)),
            _frame_step_energy(frames_after_cleanup, "left_wrist", axes=(0, 1, 2)),
            invert=False,
        ),
        "right_wrist_motion_energy": _ratio_delta(
            _frame_step_energy(frames_before_cleanup, "right_wrist", axes=(0, 1, 2)),
            _frame_step_energy(frames_after_cleanup, "right_wrist", axes=(0, 1, 2)),
            invert=False,
        ),
    }


def _finite_point(frame: Dict[str, np.ndarray], key: str) -> np.ndarray | None:
    point = frame.get(key)
    if point is None:
        return None
    arr = np.array(point, dtype=np.float64)
    if arr.shape[0] < 3 or not np.isfinite(arr).all():
        return None
    return arr


def _median_joint_distance(frames_pts: List[Dict[str, np.ndarray]], a_key: str, b_key: str) -> float:
    distances: List[float] = []
    for frame in frames_pts:
        a = _finite_point(frame, a_key)
        b = _finite_point(frame, b_key)
        if a is None or b is None:
            continue
        distances.append(float(np.linalg.norm(a - b)))
    if not distances:
        return 0.0
    return float(np.median(np.array(distances, dtype=np.float64)))


def _joint_distance_cv(frames_pts: List[Dict[str, np.ndarray]], a_key: str, b_key: str) -> float:
    distances: List[float] = []
    for frame in frames_pts:
        a = _finite_point(frame, a_key)
        b = _finite_point(frame, b_key)
        if a is None or b is None:
            continue
        distances.append(float(np.linalg.norm(a - b)))
    if len(distances) < 2:
        return 0.0
    arr = np.array(distances, dtype=np.float64)
    mean = float(np.mean(arr))
    if abs(mean) <= 1e-8:
        return 0.0
    return float(np.std(arr) / abs(mean))


def _lr_inversion_ratio(frames_pts: List[Dict[str, np.ndarray]], left_key: str, right_key: str, axis: int = 0) -> float:
    usable = 0
    inverted = 0
    for frame in frames_pts:
        left = _finite_point(frame, left_key)
        right = _finite_point(frame, right_key)
        if left is None or right is None:
            continue
        usable += 1
        if float(left[axis]) > float(right[axis]):
            inverted += 1
    if usable <= 0:
        return 0.0
    return float(inverted / usable)


def build_source_pose_stage_summary(frames_pts: List[Dict[str, np.ndarray]]) -> Dict[str, float]:
    left_foot_spread = _foot_contact_spread_metric(frames_pts, "left")
    right_foot_spread = _foot_contact_spread_metric(frames_pts, "right")
    left_wrist_energy = _frame_step_energy(frames_pts, "left_wrist", axes=(0, 1, 2))
    right_wrist_energy = _frame_step_energy(frames_pts, "right_wrist", axes=(0, 1, 2))

    return {
        "frame_count": float(len(frames_pts)),
        "root_position_jitter": _frame_step_energy(frames_pts, "mid_hip", axes=(0, 2)),
        "root_height_jitter": _frame_step_energy(frames_pts, "mid_hip", axes=(1,)),
        "foot_contact_spread_mean": float((left_foot_spread + right_foot_spread) * 0.5),
        "wrist_motion_energy_mean": float((left_wrist_energy + right_wrist_energy) * 0.5),
        "shoulder_width_median": _median_joint_distance(frames_pts, "left_shoulder", "right_shoulder"),
        "hip_width_median": _median_joint_distance(frames_pts, "left_hip", "right_hip"),
        "shoulder_width_cv": _joint_distance_cv(frames_pts, "left_shoulder", "right_shoulder"),
        "hip_width_cv": _joint_distance_cv(frames_pts, "left_hip", "right_hip"),
        "shoulder_lr_inversion_ratio": _lr_inversion_ratio(frames_pts, "left_shoulder", "right_shoulder"),
        "hip_lr_inversion_ratio": _lr_inversion_ratio(frames_pts, "left_hip", "right_hip"),
    }


def _source_pose_stage_compare(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    return {
        "root_position_jitter": _ratio_delta(before["root_position_jitter"], after["root_position_jitter"], invert=True),
        "root_height_jitter": _ratio_delta(before["root_height_jitter"], after["root_height_jitter"], invert=True),
        "foot_contact_spread_mean": _ratio_delta(
            before["foot_contact_spread_mean"],
            after["foot_contact_spread_mean"],
            invert=True,
        ),
        "shoulder_lr_inversion_ratio": _ratio_delta(
            before["shoulder_lr_inversion_ratio"],
            after["shoulder_lr_inversion_ratio"],
            invert=True,
        ),
        "hip_lr_inversion_ratio": _ratio_delta(
            before["hip_lr_inversion_ratio"],
            after["hip_lr_inversion_ratio"],
            invert=True,
        ),
    }


def build_source_pose_stage_diagnostics(stage_frames: Dict[str, List[Dict[str, np.ndarray]]]) -> Dict[str, Any]:
    stages = {name: build_source_pose_stage_summary(frames) for name, frames in stage_frames.items()}
    comparisons: Dict[str, Any] = {}
    pairs = [
        ("filled_source", "canonical"),
        ("canonical", "corrected_pre_cleanup"),
        ("corrected_pre_cleanup", "post_cleanup_pre_loop"),
        ("post_cleanup_pre_loop", "final_source"),
    ]
    for before_name, after_name in pairs:
        if before_name in stages and after_name in stages:
            comparisons[f"{before_name}_to_{after_name}"] = _source_pose_stage_compare(
                stages[before_name],
                stages[after_name],
            )

    filled = stages.get("filled_source", {})
    final_stage = stages.get("final_source", {})
    canonical_to_corrected = comparisons.get("canonical_to_corrected_pre_cleanup", {})
    corrected_to_cleanup = comparisons.get("corrected_pre_cleanup_to_post_cleanup_pre_loop", {})
    cleanup_to_final = comparisons.get("post_cleanup_pre_loop_to_final_source", {})

    def _improvement(stage_delta: Dict[str, Any], key: str) -> float:
        value = stage_delta.get(key)
        if not isinstance(value, dict):
            return 0.0
        return float(value.get("improvement_ratio", 0.0))

    early_stage_instability = (
        float(filled.get("shoulder_lr_inversion_ratio", 0.0)) > 0.15
        or float(filled.get("hip_lr_inversion_ratio", 0.0)) > 0.15
        or float(filled.get("shoulder_width_cv", 0.0)) > 0.35
    )
    corrections_degraded_root = _improvement(canonical_to_corrected, "root_position_jitter") < -0.25
    cleanup_degraded_root = _improvement(corrected_to_cleanup, "root_position_jitter") < -0.15
    final_stage_degraded_root = _improvement(cleanup_to_final, "root_position_jitter") < -0.15
    final_lr_risk = (
        float(final_stage.get("shoulder_lr_inversion_ratio", 0.0)) > 0.12
        or float(final_stage.get("hip_lr_inversion_ratio", 0.0)) > 0.12
    )
    source_pipeline_risk = (
        early_stage_instability
        or corrections_degraded_root
        or cleanup_degraded_root
        or final_stage_degraded_root
        or final_lr_risk
    )

    suspected_issue_stage = "viewer_likely"
    if early_stage_instability:
        suspected_issue_stage = "pre_corrections"
    elif corrections_degraded_root:
        suspected_issue_stage = "pose_corrections"
    elif cleanup_degraded_root:
        suspected_issue_stage = "cleanup"
    elif final_stage_degraded_root or final_lr_risk:
        suspected_issue_stage = "post_cleanup"

    return {
        "stages": stages,
        "comparisons": comparisons,
        "flags": {
            "early_stage_instability": early_stage_instability,
            "corrections_degraded_root": corrections_degraded_root,
            "cleanup_degraded_root": cleanup_degraded_root,
            "final_stage_degraded_root": final_stage_degraded_root,
            "final_lr_risk": final_lr_risk,
            "source_pipeline_risk": source_pipeline_risk,
            "suspected_issue_stage": suspected_issue_stage,
        },
    }


def _large_rotation_step_ratio(motion_values: List[List[float]], threshold_deg: float = 120.0) -> float:
    if len(motion_values) < 2:
        return 0.0
    changed = 0
    total = 0
    for idx in range(1, len(motion_values)):
        prev = np.array(motion_values[idx - 1][3:], dtype=np.float64)
        cur = np.array(motion_values[idx][3:], dtype=np.float64)
        if prev.size == 0 or cur.size == 0:
            continue
        diff = np.abs((cur - prev + 180.0) % 360.0 - 180.0)
        changed += int(np.count_nonzero(diff > float(threshold_deg)))
        total += int(diff.size)
    if total <= 0:
        return 0.0
    return float(changed / total)


def build_source_motion_stage_summary(motion_values: List[List[float]]) -> Dict[str, float]:
    if not motion_values:
        return {
            "frame_count": 0.0,
            "root_yaw_abs_median_deg": 0.0,
            "root_yaw_step_mean_deg": 0.0,
            "root_yaw_step_max_deg": 0.0,
            "large_rotation_step_ratio": 0.0,
        }

    root_yaw = np.array([float(frame[5]) if len(frame) > 5 else 0.0 for frame in motion_values], dtype=np.float64)
    root_yaw_steps = np.abs((np.diff(root_yaw) + 180.0) % 360.0 - 180.0)
    return {
        "frame_count": float(len(motion_values)),
        "root_yaw_abs_median_deg": float(np.median(np.abs(root_yaw))),
        "root_yaw_step_mean_deg": float(np.mean(root_yaw_steps)) if root_yaw_steps.size > 0 else 0.0,
        "root_yaw_step_max_deg": float(np.max(root_yaw_steps)) if root_yaw_steps.size > 0 else 0.0,
        "large_rotation_step_ratio": _large_rotation_step_ratio(motion_values),
    }


def _source_motion_stage_compare(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    return {
        "root_yaw_step_mean_deg": _ratio_delta(before["root_yaw_step_mean_deg"], after["root_yaw_step_mean_deg"], invert=True),
        "root_yaw_step_max_deg": _ratio_delta(before["root_yaw_step_max_deg"], after["root_yaw_step_max_deg"], invert=True),
        "large_rotation_step_ratio": _ratio_delta(
            before["large_rotation_step_ratio"],
            after["large_rotation_step_ratio"],
            invert=True,
        ),
    }


def build_source_motion_stage_diagnostics(
    pre_yaw_motion: List[List[float]],
    post_yaw_motion: List[List[float]],
    final_motion: List[List[float]],
) -> Dict[str, Any]:
    stages = {
        "pre_root_yaw": build_source_motion_stage_summary(pre_yaw_motion),
        "post_root_yaw": build_source_motion_stage_summary(post_yaw_motion),
        "final_motion": build_source_motion_stage_summary(final_motion),
    }
    comparisons = {
        "pre_root_yaw_to_post_root_yaw": _source_motion_stage_compare(stages["pre_root_yaw"], stages["post_root_yaw"]),
        "post_root_yaw_to_final_motion": _source_motion_stage_compare(stages["post_root_yaw"], stages["final_motion"]),
    }

    pre_to_post = comparisons["pre_root_yaw_to_post_root_yaw"]
    post_to_final = comparisons["post_root_yaw_to_final_motion"]
    yaw_normalization_spike = float(pre_to_post["root_yaw_step_mean_deg"]["improvement_ratio"]) < -0.2
    rotation_jump_increase_after_finalize = float(post_to_final["large_rotation_step_ratio"]["improvement_ratio"]) < -0.2
    final_rotation_jump_risk = float(stages["final_motion"]["large_rotation_step_ratio"]) > 0.015
    source_motion_risk = yaw_normalization_spike or rotation_jump_increase_after_finalize or final_rotation_jump_risk

    suspected_issue_stage = "pose_pipeline_likely"
    if yaw_normalization_spike:
        suspected_issue_stage = "root_yaw_normalization"
    elif rotation_jump_increase_after_finalize or final_rotation_jump_risk:
        suspected_issue_stage = "motion_finalize"

    return {
        "stages": stages,
        "comparisons": comparisons,
        "flags": {
            "yaw_normalization_spike": yaw_normalization_spike,
            "rotation_jump_increase_after_finalize": rotation_jump_increase_after_finalize,
            "final_rotation_jump_risk": final_rotation_jump_risk,
            "source_motion_risk": source_motion_risk,
            "suspected_issue_stage": suspected_issue_stage,
        },
    }


def build_quality_summary(
    *,
    source_frame_count: int,
    detected_count: int,
    interpolated_frames: int,
    carried_frames: int,
    cleanup_stats: Dict[str, float],
    yaw_norm_stats: Dict[str, float],
    unwrap_stats: Dict[str, float],
    loopability: Dict[str, Any],
    pre_cleanup_loopability: Dict[str, Any],
    skeleton_profile_stats: Dict[str, Any],
    pose_backend: str,
    contact_cleanup_enabled: bool,
) -> Dict[str, Any]:
    total_frames = max(int(source_frame_count), 1)
    detected_ratio = detected_count / total_frames
    interpolated_ratio = interpolated_frames / total_frames
    carried_ratio = carried_frames / total_frames
    side_swap_ratio = float(cleanup_stats.get("side_swaps", 0.0)) / total_frames
    contact_frames = float(cleanup_stats.get("contact_frames", 0.0))
    contact_ratio = min(contact_frames / total_frames, 1.0)
    unwrap_ratio = float(unwrap_stats.get("changed_values", 0.0)) / max(total_frames * 3.0, 1.0)
    loop_meta = loopability or pre_cleanup_loopability or {}
    loop_label = str(loop_meta.get("label", "oneshot"))

    score = 1.0
    score -= max(0.0, 0.85 - detected_ratio) * 1.1
    score -= interpolated_ratio * 0.25
    score -= carried_ratio * 0.55
    score -= min(side_swap_ratio, 0.2) * 0.5
    score -= min(unwrap_ratio, 0.35) * 0.2
    if pose_backend != "tasks":
        score -= 0.03
    if contact_cleanup_enabled and contact_ratio <= 0.0:
        score -= 0.08
    if float(yaw_norm_stats.get("normalized_applied", 0.0)) > 0.5:
        score -= 0.03
    if float(skeleton_profile_stats.get("applied", 0.0)) > 0.5:
        score += 0.02
    score = _clamp01(score)

    reasons: List[str] = []
    if detected_ratio < 0.75:
        reasons.append("low_detect_ratio")
    if interpolated_ratio > 0.12:
        reasons.append("high_interpolation_ratio")
    if carried_ratio > 0.08:
        reasons.append("high_carried_ratio")
    if side_swap_ratio > 0.03:
        reasons.append("side_swap_instability")
    if contact_cleanup_enabled and contact_ratio <= 0.0:
        reasons.append("no_stable_foot_contacts")
    if unwrap_ratio > 0.15:
        reasons.append("large_rotation_unwrap")
    if pose_backend != "tasks":
        reasons.append("legacy_pose_backend")

    if score >= 0.85:
        rating = "good"
    elif score >= 0.65:
        rating = "usable"
    elif score >= 0.45:
        rating = "risky"
    else:
        rating = "poor"

    tracking_ok = detected_ratio >= 0.8 and carried_ratio <= 0.05
    foot_contact_ok = (not contact_cleanup_enabled) or contact_ratio > 0.0
    loop_candidate = loop_label == "cyclic"
    retarget_risk = (not tracking_ok) or side_swap_ratio > 0.04 or unwrap_ratio > 0.2

    return {
        "score": float(score),
        "rating": rating,
        "tracking_ok": tracking_ok,
        "foot_contact_ok": foot_contact_ok,
        "loop_candidate": loop_candidate,
        "retarget_risk": retarget_risk,
        "detected_ratio": float(detected_ratio),
        "interpolated_ratio": float(interpolated_ratio),
        "carried_ratio": float(carried_ratio),
        "contact_ratio": float(contact_ratio),
        "loop_label": loop_label,
        "reasons": reasons,
    }
