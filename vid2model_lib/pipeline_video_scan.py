from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .pose_model import ensure_pose_model, ensure_hand_model
from .pose_points import extract_pose_points
from .hand_points import extract_hand_points


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


def _normalize_scan_options(opencv_enhance: str, max_frame_side: int, roi_crop: str) -> Tuple[str, int, str]:
    normalized_enhance = str(opencv_enhance).strip().lower()
    if normalized_enhance not in {"off", "light", "strong"}:
        raise ValueError("opencv_enhance must be one of: off, light, strong")
    if max_frame_side < 0:
        raise ValueError("max_frame_side must be >= 0")
    normalized_roi_crop = str(roi_crop).strip().lower()
    if normalized_roi_crop not in {"off", "auto"}:
        raise ValueError("roi_crop must be one of: off, auto")
    return normalized_enhance, max_frame_side, normalized_roi_crop


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

    processed = cv2.bilateralFilter(
        processed, d=bilateral_d, sigmaColor=bilateral_sigma, sigmaSpace=bilateral_sigma
    )
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


def _detect_pose_and_bbox(
    detect_pose: Callable[[np.ndarray, int], Any],
    frame_bgr: np.ndarray,
    ts_ms: int,
    bbox_w: int,
    bbox_h: int,
    detect_hand: Optional[Callable[[np.ndarray, int], Any]] = None,
) -> Tuple[Any, Optional[Dict[str, np.ndarray]], Optional[Tuple[float, float, float, float]]]:
    res = detect_pose(frame_bgr, ts_ms)
    hand_res = None
    if detect_hand is not None:
        try:
            hand_res = detect_hand(frame_bgr, ts_ms)
        except Exception as exc:
            print(f"[vid2model] hand detection failed: {exc}", file=sys.stderr)

    pts = extract_pose_points(res, hand_results=hand_res)
    bbox = extract_pose_bbox_pixels(res, bbox_w, bbox_h)
    return res, pts, bbox


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
    hand_tracking: str = "off",
) -> Tuple[
    Callable[[np.ndarray, int], Any],
    Optional[Callable[[np.ndarray, int], Any]],
    Callable[[], None],
    str,
]:
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

        detect_hand = None
        hand_close = lambda: None

        if hand_tracking == "auto":
            try:
                hand_model_path = ensure_hand_model()
                hand_options = mp_vision.HandLandmarkerOptions(
                    base_options=mp_tasks_python.BaseOptions(
                        model_asset_path=str(hand_model_path),
                        delegate=mp_tasks_python.BaseOptions.Delegate.CPU,
                    ),
                    running_mode=mp_vision.RunningMode.VIDEO,
                    num_hands=2,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                hands = mp_vision.HandLandmarker.create_from_options(hand_options)

                def detect_hand(frame_bgr: np.ndarray, ts_ms: int):
                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    return hands.detect_for_video(mp_image, ts_ms)

                hand_close = hands.close
            except Exception as exc:
                print(f"[vid2model] hand tracking failed to initialize: {exc}", file=sys.stderr)
                detect_hand = None

        def detect_pose(frame_bgr: np.ndarray, ts_ms: int):
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            return pose.detect_for_video(mp_image, ts_ms)

        def close_all():
            pose.close()
            hand_close()

        return detect_pose, detect_hand, close_all, "tasks"
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

    detect_hand = None
    hand_close = lambda: None

    if hand_tracking == "auto":
        try:
            hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            def detect_hand(frame_bgr: np.ndarray, ts_ms: int):
                del ts_ms
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                return hands.process(rgb)

            hand_close = hands.close
        except Exception as exc:
            print(f"[vid2model] hand tracking failed to initialize: {exc}", file=sys.stderr)
            detect_hand = None

    def detect_pose(frame_bgr: np.ndarray, ts_ms: int):
        del ts_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return pose.process(rgb)

    def close_all():
        pose.close()
        hand_close()

    return detect_pose, detect_hand, close_all, "solutions"


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


def _log_scan_configuration(opencv_enhance: str, max_frame_side: int, roi_crop: str) -> None:
    if opencv_enhance != "off" or max_frame_side > 0:
        print(
            f"[vid2model] opencv_preprocess enhance={opencv_enhance} max_frame_side={max_frame_side}",
            file=sys.stderr,
        )
    if roi_crop == "auto":
        print("[vid2model] roi_crop mode=auto", file=sys.stderr)


def _detect_frame_pose_with_roi(
    frame_for_pose: np.ndarray,
    ts_ms: int,
    detect_pose: Callable[[np.ndarray, int], Any],
    roi_crop: str,
    roi_state: Optional[Tuple[int, int, int, int]],
    detect_hand: Optional[Callable[[np.ndarray, int], Any]] = None,
) -> Tuple[
    Optional[Dict[str, np.ndarray]],
    Optional[Tuple[float, float, float, float]],
    bool,
    bool,
]:
    frame_h, frame_w = frame_for_pose.shape[:2]
    pts: Optional[Dict[str, np.ndarray]] = None
    bbox_for_roi: Optional[Tuple[float, float, float, float]] = None
    used_roi = False
    fell_back_to_full_frame = False

    if roi_crop == "auto" and roi_state is not None:
        x0, y0, x1, y1 = roi_state
        roi_frame = frame_for_pose[y0:y1, x0:x1]
        if roi_frame.size > 0:
            used_roi = True
            _, pts, roi_bbox = _detect_pose_and_bbox(
                detect_pose,
                roi_frame,
                ts_ms,
                roi_frame.shape[1],
                roi_frame.shape[0],
                detect_hand=detect_hand,
            )
            if roi_bbox is not None:
                bbox_for_roi = (
                    roi_bbox[0] + x0,
                    roi_bbox[1] + y0,
                    roi_bbox[2] + x0,
                    roi_bbox[3] + y0,
                )
        if pts is None:
            fell_back_to_full_frame = True
            used_roi = False
            _, pts, bbox_for_roi = _detect_pose_and_bbox(
                detect_pose, frame_for_pose, ts_ms + 1, frame_w, frame_h, detect_hand=detect_hand
            )
    else:
        _, pts, bbox_for_roi = _detect_pose_and_bbox(
            detect_pose, frame_for_pose, ts_ms, frame_w, frame_h, detect_hand=detect_hand
        )

    return pts, bbox_for_roi, used_roi, fell_back_to_full_frame


def _update_roi_tracking_state(
    roi_crop: str,
    roi_state: Optional[Tuple[int, int, int, int]],
    pts: Optional[Dict[str, np.ndarray]],
    bbox_for_roi: Optional[Tuple[float, float, float, float]],
    frame_w: int,
    frame_h: int,
    used_roi: bool,
) -> Tuple[Optional[Tuple[int, int, int, int]], bool]:
    roi_reset = False
    if roi_crop != "auto":
        return roi_state, roi_reset
    if pts is not None and bbox_for_roi is not None:
        return update_tracking_roi(roi_state, bbox_for_roi, frame_w, frame_h), roi_reset
    if pts is None and roi_state is not None and not used_roi:
        return None, True
    return roi_state, roi_reset


def collect_detected_pose_samples(
    input_path: Path,
    model_complexity: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    progress_every: int = 100,
    opencv_enhance: str = "off",
    max_frame_side: int = 0,
    roi_crop: str = "off",
    override_fps: float = None,
    hand_tracking: str = "off",
) -> Tuple[float, List[Optional[Dict[str, np.ndarray]]], List[Dict[str, np.ndarray]], Dict[str, Any]]:
    import cv2

    opencv_enhance, max_frame_side, roi_crop = _normalize_scan_options(
        opencv_enhance,
        max_frame_side,
        roi_crop,
    )

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

    if override_fps is not None and override_fps > 0:
        fps = override_fps
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1e-6:
            fps = 30.0

    detect_pose, detect_hand, close_all, pose_backend = _create_pose_detector(
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        cv2=cv2,
        hand_tracking=hand_tracking,
    )
    print(f"[vid2model] pose_backend={pose_backend} hand_tracking={hand_tracking}", file=sys.stderr)

    frames_pts_raw: List[Optional[Dict[str, np.ndarray]]] = []
    detected_samples: List[Dict[str, np.ndarray]] = []
    detected_count = 0
    roi_state: Optional[Tuple[int, int, int, int]] = None
    roi_used_count = 0
    roi_fallback_count = 0
    roi_reset_count = 0
    _log_scan_configuration(opencv_enhance, max_frame_side, roi_crop)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_for_pose = preprocess_video_frame(frame, cv2, opencv_enhance, max_frame_side)
        ts_ms = int((frame_idx * 1000.0) / fps)
        frame_h, frame_w = frame_for_pose.shape[:2]

        pts, bbox_for_roi, used_roi, fell_back_to_full_frame = _detect_frame_pose_with_roi(
            frame_for_pose,
            ts_ms,
            detect_pose,
            roi_crop,
            roi_state,
            detect_hand=detect_hand,
        )
        if used_roi:
            roi_used_count += 1
        if fell_back_to_full_frame:
            roi_fallback_count += 1

        roi_state, roi_reset = _update_roi_tracking_state(
            roi_crop,
            roi_state,
            pts,
            bbox_for_roi,
            frame_w,
            frame_h,
            used_roi,
        )
        if roi_reset:
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
    close_all()

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
