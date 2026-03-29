from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .math3d import euler_zxy_from_matrix, rotation_align, rotation_align_with_secondary
from .pose_model import ensure_pose_model
from .pose_points import extract_pose_points
from .skeleton import CHILDREN, JOINTS, MAP_TO_POINTS


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
        if np.linalg.norm(offset) < 1e-6:
            offset = np.array([0.0, 5.0, 0.0], dtype=np.float64)
        rest[joint.name] = offset

    return rest


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
) -> List[float]:
    channels: List[float] = []
    root_name = JOINTS[0].name
    root_point_name = MAP_TO_POINTS[root_name][0]
    root_pos = pts[root_point_name] - ref_root
    global_rot: Dict[str, np.ndarray] = {root_name: np.eye(3)}

    def secondary_vectors_for_joint(name: str, pts_cur: Dict[str, np.ndarray], rest: Dict[str, np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if name == "hips":
            cur_side = pts_cur["right_hip"] - pts_cur["left_hip"]
            rest_side = rest["rightUpperLeg"] - rest["leftUpperLeg"]
            return cur_side, rest_side
        if name in ("spine", "chest", "upperChest"):
            cur_side = pts_cur["right_shoulder"] - pts_cur["left_shoulder"]
            rest_side = rest["rightUpperArm"] - rest["leftUpperArm"]
            return cur_side, rest_side
        return None, None

    def solve_joint(name: str) -> Tuple[float, float, float]:
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


def extract_pose_bbox_pixels(res, frame_w: int, frame_h: int) -> Optional[Tuple[float, float, float, float]]:
    if frame_w <= 0 or frame_h <= 0:
        return None
    if not getattr(res, "pose_landmarks", None):
        return None
    if not res.pose_landmarks:
        return None

    xs: List[float] = []
    ys: List[float] = []
    landmarks = res.pose_landmarks[0]
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
) -> Tuple[float, Dict[str, np.ndarray], List[List[float]], np.ndarray, List[Dict[str, np.ndarray]]]:
    # Lazy import of heavy deps keeps CLI/help and unit tests lightweight.
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks_python
    from mediapipe.tasks.python import vision as mp_vision

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

    def detect_pose(frame_bgr: np.ndarray, ts_ms: int):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return pose.detect_for_video(mp_image, ts_ms)

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
    pose.close()

    if roi_crop == "auto":
        print(
            (
                f"[vid2model] roi_stats used={roi_used_count} "
                f"fallback_full={roi_fallback_count} resets={roi_reset_count}"
            ),
            file=sys.stderr,
        )

    frames_pts, interpolated_frames, carried_frames = fill_pose_gaps(frames_pts_raw, max_gap_interpolate)
    print(
        (
            f"[vid2model] done frames={len(frames_pts_raw)} detected={detected_count} "
            f"interpolated={interpolated_frames} carried={carried_frames}"
        ),
        file=sys.stderr,
    )

    rest_offsets = build_rest_offsets(detected_samples if detected_samples else frames_pts[:20])
    root_point_name = MAP_TO_POINTS[JOINTS[0].name][0]
    ref_root = frames_pts[0][root_point_name].copy()

    motion_values = []
    for pts in frames_pts:
        motion_values.append(frame_channels(pts, rest_offsets, ref_root))

    return fps, rest_offsets, motion_values, ref_root, frames_pts
