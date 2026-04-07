from __future__ import annotations

"""Public pipeline facade.

This module keeps the stable orchestration API used by the CLI, tools, tests,
and package exports while delegating implementation details to focused
submodules.
"""

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .pipeline_auto_pose import (
    AUTO_FEATURE_NAMES,
    AUTO_POSE_PRESETS,
    DEFAULT_POSE_CORRECTIONS,
    PoseCorrectionProfile,
    _apply_pose_preset,
    _auto_feature_vector as _pipeline_auto_feature_vector,
    _finite_or_zero,
    _load_auto_classifier,
    _predict_auto_label,
    _sample_feature_summary,
    build_pose_correction_profile,
    resolve_auto_pose_corrections as _resolve_auto_pose_corrections,
)
from .pipeline_channels import frame_channels as _pipeline_frame_channels
from .pipeline_cleanup import (
    _apply_segment_length_constraints,
    _build_target_segment_lengths,
    _copy_pose_frame,
    _smooth_pose_frames,
    cleanup_pose_frames,
)
from .pipeline_gap_fill import (
    fill_pose_gaps as _pipeline_fill_pose_gaps,
    interpolate_pose_points as _pipeline_interpolate_pose_points,
)
from .pipeline_loop import (
    LOOP_FEATURE_KEYS,
    _loop_feature_vector,
    analyze_motion_loopability,
    blend_motion_loop_edges,
    extract_motion_loop,
)
from .pipeline_mirror import (
    fix_temporal_side_swaps as _pipeline_fix_temporal_side_swaps,
    looks_mirrored as _pipeline_looks_mirrored,
    looks_side_swapped as _pipeline_looks_side_swapped,
    mirror_pose_points as _pipeline_mirror_pose_points,
    pose_distance as _pipeline_pose_distance,
    swap_lr_name as _pipeline_swap_lr_name,
    swap_pose_sides as _pipeline_swap_pose_sides,
)
from .pipeline_motion_transforms import (
    _wrap_angle_deg,
    apply_lower_body_rotation_mode,
    apply_manual_root_yaw_offset,
    apply_upper_body_rotation_scale,
    normalize_motion_root_yaw,
    unwrap_motion_rotation_channels,
)
from .pipeline_diagnostics import (
    build_cleanup_evaluation,
    build_quality_summary,
    build_source_motion_stage_diagnostics,
    build_source_pose_stage_diagnostics,
)
from .pipeline_rest_offsets import (
    _apply_vrm_humanoid_baseline_to_rest_offsets,
    _skeleton_profile_scale_group,
    apply_skeleton_profile_to_rest_offsets,
    build_rest_offsets,
)
from .pipeline_retarget import (
    apply_pose_corrections as _pipeline_apply_pose_corrections,
    build_reference_basis as _pipeline_build_reference_basis,
    canonicalize_pose_points as _pipeline_canonicalize_pose_points,
    median_pose_sample as _pipeline_median_pose_sample,
    rotate_points_about_y as _pipeline_rotate_points_about_y,
)
from .pipeline_video_scan import (
    _should_fallback_to_legacy_pose,
    clamp_roi_box,
    collect_detected_pose_samples,
    extract_pose_bbox_pixels,
    gamma_lut,
    preprocess_video_frame,
    resize_frame_for_detection,
    update_tracking_roi,
)
from .skeleton import CHILDREN, JOINTS, MAP_TO_POINTS

__all__ = [
    "AUTO_FEATURE_NAMES",
    "AUTO_POSE_PRESETS",
    "DEFAULT_POSE_CORRECTIONS",
    "LOOP_FEATURE_KEYS",
    "PoseCorrectionProfile",
    "analyze_motion_loopability",
    "apply_lower_body_rotation_mode",
    "apply_manual_root_yaw_offset",
    "apply_skeleton_profile_to_rest_offsets",
    "apply_upper_body_rotation_scale",
    "blend_motion_loop_edges",
    "build_pose_correction_profile",
    "build_rest_offsets",
    "bvh_hierarchy_lines",
    "clamp_roi_box",
    "cleanup_pose_frames",
    "collect_detected_pose_samples",
    "convert_video_to_bvh",
    "extract_motion_loop",
    "extract_pose_bbox_pixels",
    "fill_pose_gaps",
    "frame_channels",
    "gamma_lut",
    "interpolate_pose_points",
    "normalize_motion_root_yaw",
    "preprocess_video_frame",
    "resolve_auto_pose_corrections",
    "resize_frame_for_detection",
    "unwrap_motion_rotation_channels",
    "update_tracking_roi",
]


def resolve_auto_pose_corrections(
    samples: List[Dict[str, np.ndarray]],
    base: PoseCorrectionProfile,
) -> Tuple[PoseCorrectionProfile, Dict[str, Any]]:
    return _resolve_auto_pose_corrections(samples, base, _median_pose_sample)


def _auto_feature_vector(samples: List[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, Dict[str, float]]:
    return _pipeline_auto_feature_vector(samples, _median_pose_sample)


# Compatibility layer: keep these private facade helpers stable while the
# implementation lives in focused submodules.
def _swap_lr_name(name: str) -> str:
    return _pipeline_swap_lr_name(name)


def _mirror_pose_points(pts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return _pipeline_mirror_pose_points(pts)


def _swap_pose_sides(pts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return _pipeline_swap_pose_sides(pts)


def _looks_mirrored(sample: Dict[str, np.ndarray]) -> bool:
    return _pipeline_looks_mirrored(sample)


def _looks_side_swapped(sample: Dict[str, np.ndarray]) -> bool:
    return _pipeline_looks_side_swapped(sample)


def _pose_distance(
    prev_pts: Dict[str, np.ndarray],
    cur_pts: Dict[str, np.ndarray],
) -> float:
    return _pipeline_pose_distance(prev_pts, cur_pts)


def _fix_temporal_side_swaps(
    frames_pts: List[Dict[str, np.ndarray]],
) -> Tuple[List[Dict[str, np.ndarray]], int]:
    return _pipeline_fix_temporal_side_swaps(frames_pts)


def _median_pose_sample(samples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return _pipeline_median_pose_sample(samples)


def _build_reference_basis(
    sample: Dict[str, np.ndarray],
    corrections: Optional[PoseCorrectionProfile] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    return _pipeline_build_reference_basis(sample, corrections)


def _rotate_points_about_y(
    pts: Dict[str, np.ndarray],
    keys: List[str],
    angle_deg: float,
    pivot: np.ndarray,
) -> None:
    _pipeline_rotate_points_about_y(pts, keys, angle_deg, pivot)


def _apply_pose_corrections(
    pts: Dict[str, np.ndarray],
    corrections: PoseCorrectionProfile,
) -> Dict[str, np.ndarray]:
    return _pipeline_apply_pose_corrections(pts, corrections)


def _canonicalize_pose_points(
    pts: Dict[str, np.ndarray],
    basis: np.ndarray,
    origin: np.ndarray,
) -> Dict[str, np.ndarray]:
    return _pipeline_canonicalize_pose_points(pts, basis, origin)


def _build_cleanup_evaluation(
    frames_before_cleanup: List[Dict[str, np.ndarray]],
    frames_after_cleanup: List[Dict[str, np.ndarray]],
) -> Dict[str, Any]:
    return build_cleanup_evaluation(frames_before_cleanup, frames_after_cleanup)


def _build_source_pose_stage_diagnostics(stage_frames: Dict[str, List[Dict[str, np.ndarray]]]) -> Dict[str, Any]:
    return build_source_pose_stage_diagnostics(stage_frames)


def _build_source_motion_stage_diagnostics(
    pre_yaw_motion: List[List[float]],
    post_yaw_motion: List[List[float]],
    final_motion: List[List[float]],
) -> Dict[str, Any]:
    return build_source_motion_stage_diagnostics(pre_yaw_motion, post_yaw_motion, final_motion)


def _build_quality_summary(
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
    return build_quality_summary(
        source_frame_count=source_frame_count,
        detected_count=detected_count,
        interpolated_frames=interpolated_frames,
        carried_frames=carried_frames,
        cleanup_stats=cleanup_stats,
        yaw_norm_stats=yaw_norm_stats,
        unwrap_stats=unwrap_stats,
        loopability=loopability,
        pre_cleanup_loopability=pre_cleanup_loopability,
        skeleton_profile_stats=skeleton_profile_stats,
        pose_backend=pose_backend,
        contact_cleanup_enabled=contact_cleanup_enabled,
    )


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
    return _pipeline_frame_channels(pts, rest_offsets, ref_root, corrections)


def interpolate_pose_points(
    prev_pts: Dict[str, np.ndarray],
    next_pts: Dict[str, np.ndarray],
    alpha: float,
) -> Dict[str, np.ndarray]:
    return _pipeline_interpolate_pose_points(prev_pts, next_pts, alpha)


def fill_pose_gaps(
    frames_pts: List[Optional[Dict[str, np.ndarray]]],
    max_gap_interpolate: int,
) -> Tuple[List[Dict[str, np.ndarray]], int, int]:
    return _pipeline_fill_pose_gaps(frames_pts, max_gap_interpolate)


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
    override_fps: Optional[float] = None,
    hand_tracking: str = "off",
    include_source_stage_diagnostics: bool = False,
    pose_cache_path: Optional[Path] = None,
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
        override_fps=override_fps,
        hand_tracking=hand_tracking,
        pose_cache_path=pose_cache_path,
    )
    detected_count = scan_stats["detected"]
    roi_used_count = scan_stats["roi_used"]
    roi_fallback_count = scan_stats["roi_fallback"]
    roi_reset_count = scan_stats["roi_resets"]
    pose_backend = str(scan_stats.get("pose_backend", "unknown"))

    frames_pts, interpolated_frames, carried_frames = fill_pose_gaps(frames_pts_raw, max_gap_interpolate)
    collect_source_stage_diagnostics = bool(include_source_stage_diagnostics)
    frames_filled_source: Optional[List[Dict[str, np.ndarray]]] = None
    if collect_source_stage_diagnostics:
        frames_filled_source = [_copy_pose_frame(pts) for pts in frames_pts]
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
    frames_canonical: Optional[List[Dict[str, np.ndarray]]] = None
    if collect_source_stage_diagnostics:
        frames_canonical = [_copy_pose_frame(pts) for pts in frames_pts]
    canonical_anchor_samples = [
        _canonicalize_pose_points(sample, reference_basis, reference_origin) for sample in anchor_samples
    ]
    frames_pts = [_apply_pose_corrections(pts, corrections) for pts in frames_pts]
    frames_corrected_pre_cleanup: Optional[List[Dict[str, np.ndarray]]] = None
    if collect_source_stage_diagnostics:
        frames_corrected_pre_cleanup = [_copy_pose_frame(pts) for pts in frames_pts]
    canonical_anchor_samples = [_apply_pose_corrections(sample, corrections) for sample in canonical_anchor_samples]
    frames_before_cleanup = [_copy_pose_frame(pts) for pts in frames_pts]
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
    frames_post_cleanup_pre_loop: Optional[List[Dict[str, np.ndarray]]] = None
    if collect_source_stage_diagnostics:
        frames_post_cleanup_pre_loop = [_copy_pose_frame(pts) for pts in frames_pts]
    canonical_anchor_samples, _ = cleanup_pose_frames(
        canonical_anchor_samples,
        canonical_anchor_samples,
        use_contact_cleanup=False,
    )
    cleanup_evaluation = build_cleanup_evaluation(frames_before_cleanup, frames_pts)
    print(
        (
            f"[vid2model] source_cleanup side_swaps={int(cleanup_stats['side_swaps'])} "
            f"smooth_alpha={cleanup_stats['smooth_alpha']:.2f} "
            f"segments={int(cleanup_stats['length_constraints'])} "
            f"foot_contacts={int(cleanup_stats['contact_windows'])}/{int(cleanup_stats['contact_frames'])} "
            f"pelvis_frames={int(cleanup_stats['pelvis_contact_frames'])} "
            f"root_frames={int(cleanup_stats.get('root_stabilized_frames', 0.0))} "
            f"leg_ik_frames={int(cleanup_stats['leg_ik_frames'])}"
        ),
        file=sys.stderr,
    )
    print(
        (
            f"[vid2model] cleanup_eval root_jitter={cleanup_evaluation['root_position_jitter']['before']:.3f}"
            f"->{cleanup_evaluation['root_position_jitter']['after']:.3f} "
            f"root_height={cleanup_evaluation['root_height_jitter']['before']:.3f}"
            f"->{cleanup_evaluation['root_height_jitter']['after']:.3f} "
            f"left_foot={cleanup_evaluation['left_foot_contact_spread']['before']:.3f}"
            f"->{cleanup_evaluation['left_foot_contact_spread']['after']:.3f}"
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
    motion_values_pre_root_yaw: Optional[List[List[float]]] = None
    if collect_source_stage_diagnostics:
        motion_values_pre_root_yaw = [list(frame) for frame in motion_values]
    motion_values, yaw_norm_stats = normalize_motion_root_yaw(motion_values)
    motion_values_post_root_yaw: Optional[List[List[float]]] = None
    if collect_source_stage_diagnostics:
        motion_values_post_root_yaw = [list(frame) for frame in motion_values]
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
    if (
        (np.isfinite(corrections.upper_body_rotation_scale) and abs(float(corrections.upper_body_rotation_scale) - 1.0) > 1e-6)
        or (np.isfinite(corrections.arm_rotation_scale) and abs(float(corrections.arm_rotation_scale) - 1.0) > 1e-6)
    ):
        motion_values = apply_upper_body_rotation_scale(
            motion_values,
            corrections.upper_body_rotation_scale,
            corrections.arm_rotation_scale,
        )
        print(
            (
                f"[vid2model] upper_body_rotation_scale={float(corrections.upper_body_rotation_scale):.3f} "
                f"arm_rotation_scale={float(corrections.arm_rotation_scale):.3f}"
            ),
            file=sys.stderr,
        )
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
        "evaluation": cleanup_evaluation,
        "root_yaw": {
            "normalized_applied": float(yaw_norm_stats["applied"]),
            "normalized_offset_deg": float(yaw_norm_stats["offset_deg"]),
            "manual_offset_deg": float(root_yaw_offset_deg),
            "upper_body_rotation_scale": float(corrections.upper_body_rotation_scale),
            "arm_rotation_scale": float(corrections.arm_rotation_scale),
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
    if (
        collect_source_stage_diagnostics
        and frames_filled_source is not None
        and frames_canonical is not None
        and frames_corrected_pre_cleanup is not None
        and frames_post_cleanup_pre_loop is not None
        and motion_values_pre_root_yaw is not None
        and motion_values_post_root_yaw is not None
    ):
        diagnostics["source_stages"] = {
            "pose": build_source_pose_stage_diagnostics(
                {
                    "filled_source": frames_filled_source,
                    "canonical": frames_canonical,
                    "corrected_pre_cleanup": frames_corrected_pre_cleanup,
                    "post_cleanup_pre_loop": frames_post_cleanup_pre_loop,
                    "final_source": frames_pts,
                }
            ),
            "motion": build_source_motion_stage_diagnostics(
                motion_values_pre_root_yaw,
                motion_values_post_root_yaw,
                motion_values,
            ),
        }
    diagnostics["quality"] = build_quality_summary(
        source_frame_count=len(frames_pts_raw),
        detected_count=detected_count,
        interpolated_frames=interpolated_frames,
        carried_frames=carried_frames,
        cleanup_stats=cleanup_stats,
        yaw_norm_stats=yaw_norm_stats,
        unwrap_stats=unwrap_stats,
        loopability=loopability,
        pre_cleanup_loopability=pre_cleanup_loopability,
        skeleton_profile_stats=skeleton_profile_stats,
        pose_backend=pose_backend,
        contact_cleanup_enabled=contact_cleanup_enabled,
    )

    return fps, rest_offsets, motion_values, ref_root, frames_pts, diagnostics
