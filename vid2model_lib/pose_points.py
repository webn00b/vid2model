from __future__ import annotations

from typing import Dict, Optional

import numpy as np


LM = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}


def _average_points(points: list[np.ndarray]) -> np.ndarray:
    if not points:
        return np.zeros(3, dtype=np.float64)
    return np.mean(np.stack(points, axis=0), axis=0)


def _first_landmark_list(payload):
    if payload is None:
        return None
    if isinstance(payload, (list, tuple)):
        return payload[0] if payload else None
    if hasattr(payload, "landmark"):
        return payload.landmark
    return None


def _safe_normalized(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vec_norm = np.linalg.norm(vec)
    if vec_norm > 1e-8:
        return vec / vec_norm
    fallback_norm = np.linalg.norm(fallback)
    if fallback_norm > 1e-8:
        return fallback / fallback_norm
    return np.array([1.0, 0.0, 0.0], dtype=np.float64)


def _blend_directions(*weighted_dirs: tuple[np.ndarray, float], fallback: np.ndarray) -> np.ndarray:
    accum = np.zeros(3, dtype=np.float64)
    total_weight = 0.0
    for direction, weight in weighted_dirs:
        if weight <= 0.0:
            continue
        accum += np.array(direction, dtype=np.float64) * weight
        total_weight += weight
    if total_weight <= 1e-8:
        return _safe_normalized(fallback, fallback)
    return _safe_normalized(accum / total_weight, fallback)


def _build_hand_points(side: str, pts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    wrist = pts[f"{side}_wrist"]
    elbow = pts[f"{side}_elbow"]
    shoulder = pts[f"{side}_shoulder"]
    index_tip = pts[f"{side}_index_tip"]
    pinky_tip = pts[f"{side}_pinky_tip"]
    thumb_tip = pts[f"{side}_thumb_tip"]

    forearm_dir = _safe_normalized(wrist - elbow, wrist - shoulder)
    palm_center = (index_tip + pinky_tip) * 0.5
    hand_dir = _safe_normalized(palm_center - wrist, forearm_dir)
    hand_span = np.linalg.norm(index_tip - pinky_tip)
    forearm_len = np.linalg.norm(wrist - elbow)
    if hand_span < 1e-6:
        hand_span = max(forearm_len * 0.22, 1.0)
    palm_len = np.linalg.norm(palm_center - wrist)
    max_finger_len = max(forearm_len * 0.36, hand_span * 0.9, 1.0)
    finger_len = min(max(palm_len * 0.7, hand_span * 0.32, 1.0), max_finger_len)
    hand = wrist + hand_dir * (0.12 * finger_len)

    middle_tip = palm_center + hand_dir * (0.2 * finger_len)
    ring_tip = pinky_tip * 0.65 + index_tip * 0.35 + hand_dir * (0.08 * finger_len)

    def finger_chain(tip: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vec = tip - hand
        if np.linalg.norm(vec) < 1e-6:
            vec = hand_dir * finger_len
        proximal = hand + vec * 0.28
        intermediate = hand + vec * 0.58
        distal = hand + vec * 0.82
        return proximal, intermediate, distal

    def thumb_chain(tip: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vec = tip - hand
        if np.linalg.norm(vec) < 1e-6:
            vec = hand_dir * (finger_len * 0.75)
        metacarpal = hand + vec * 0.18
        proximal = hand + vec * 0.42
        distal = hand + vec * 0.68
        return metacarpal, proximal, distal

    index_prox, index_inter, index_dist = finger_chain(index_tip)
    middle_prox, middle_inter, middle_dist = finger_chain(middle_tip)
    ring_prox, ring_inter, ring_dist = finger_chain(ring_tip)
    little_prox, little_inter, little_dist = finger_chain(pinky_tip)
    thumb_meta, thumb_prox, thumb_dist = thumb_chain(thumb_tip)

    return {
        f"{side}_hand": hand,
        f"{side}_index_proximal": index_prox,
        f"{side}_index_intermediate": index_inter,
        f"{side}_index_distal": index_dist,
        f"{side}_middle_proximal": middle_prox,
        f"{side}_middle_intermediate": middle_inter,
        f"{side}_middle_distal": middle_dist,
        f"{side}_ring_proximal": ring_prox,
        f"{side}_ring_intermediate": ring_inter,
        f"{side}_ring_distal": ring_dist,
        f"{side}_little_proximal": little_prox,
        f"{side}_little_intermediate": little_inter,
        f"{side}_little_distal": little_dist,
        f"{side}_thumb_metacarpal": thumb_meta,
        f"{side}_thumb_proximal": thumb_prox,
        f"{side}_thumb_distal": thumb_dist,
    }


def extract_pose_points(res) -> Optional[Dict[str, np.ndarray]]:
    # MediaPipe Tasks returns a list of poses; mediapipe.solutions.pose returns
    # a NormalizedLandmarkList / LandmarkList object directly.
    world_landmarks = _first_landmark_list(getattr(res, "pose_world_landmarks", None))
    if world_landmarks is not None:

        def point(idx: int) -> np.ndarray:
            return np.array([world_landmarks[idx].x, -world_landmarks[idx].y, -world_landmarks[idx].z], dtype=np.float64)

    else:
        landmarks = _first_landmark_list(getattr(res, "pose_landmarks", None))
        if landmarks is None:
            return None

        def point(idx: int) -> np.ndarray:
            return np.array([landmarks[idx].x - 0.5, -(landmarks[idx].y - 0.5), -landmarks[idx].z], dtype=np.float64)

    pts = {
        "left_shoulder": point(LM["left_shoulder"]),
        "right_shoulder": point(LM["right_shoulder"]),
        "left_eye": point(LM["left_eye"]),
        "right_eye": point(LM["right_eye"]),
        "left_ear": point(LM["left_ear"]),
        "right_ear": point(LM["right_ear"]),
        "left_elbow": point(LM["left_elbow"]),
        "right_elbow": point(LM["right_elbow"]),
        "left_wrist": point(LM["left_wrist"]),
        "right_wrist": point(LM["right_wrist"]),
        "left_pinky_tip": point(LM["left_pinky"]),
        "right_pinky_tip": point(LM["right_pinky"]),
        "left_index_tip": point(LM["left_index"]),
        "right_index_tip": point(LM["right_index"]),
        "left_thumb_tip": point(LM["left_thumb"]),
        "right_thumb_tip": point(LM["right_thumb"]),
        "left_hip": point(LM["left_hip"]),
        "right_hip": point(LM["right_hip"]),
        "left_knee": point(LM["left_knee"]),
        "right_knee": point(LM["right_knee"]),
        "left_ankle": point(LM["left_ankle"]),
        "right_ankle": point(LM["right_ankle"]),
        "left_heel": point(LM["left_heel"]),
        "right_heel": point(LM["right_heel"]),
        "left_toes": point(LM["left_foot_index"]),
        "right_toes": point(LM["right_foot_index"]),
        "nose": point(LM["nose"]),
    }

    pts.update(_build_hand_points("left", pts))
    pts.update(_build_hand_points("right", pts))

    pts["mid_hip"] = (pts["left_hip"] + pts["right_hip"]) * 0.5
    pts["spine"] = (pts["mid_hip"] + (pts["left_shoulder"] + pts["right_shoulder"]) * 0.5) * 0.5
    pts["chest"] = (pts["left_shoulder"] + pts["right_shoulder"]) * 0.5
    pts["left_shoulder_clavicle"] = pts["chest"] + (pts["left_shoulder"] - pts["chest"]) * 0.35
    pts["right_shoulder_clavicle"] = pts["chest"] + (pts["right_shoulder"] - pts["chest"]) * 0.35

    shoulder_width = np.linalg.norm(pts["left_shoulder"] - pts["right_shoulder"])
    face_points = [
        pts["nose"],
        pts["left_eye"],
        pts["right_eye"],
        pts["left_ear"],
        pts["right_ear"],
    ]
    eye_center = _average_points([pts["left_eye"], pts["right_eye"]])
    ear_center = _average_points([pts["left_ear"], pts["right_ear"]])
    face_center = _average_points(face_points)
    torso_up = _safe_normalized(pts["chest"] - pts["mid_hip"], np.array([0.0, 1.0, 0.0], dtype=np.float64))
    shoulder_axis = _safe_normalized(
        pts["right_shoulder"] - pts["left_shoulder"],
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )
    face_hint = _safe_normalized(face_center - pts["chest"], torso_up)
    torso_forward = _safe_normalized(np.cross(shoulder_axis, torso_up), face_hint)
    if float(np.dot(torso_forward, face_hint)) < 0.0:
        torso_forward = -torso_forward

    upper_chest_dir = _blend_directions(
        (torso_up, 0.7),
        (torso_forward, 0.3),
        fallback=torso_up,
    )
    neck_anchor = _average_points([eye_center, ear_center])
    neck_dir = _blend_directions(
        (_safe_normalized(neck_anchor - pts["chest"], upper_chest_dir), 0.55),
        (upper_chest_dir, 0.45),
        fallback=upper_chest_dir,
    )
    head_anchor = _average_points([pts["nose"], eye_center, ear_center])
    head_dir = _blend_directions(
        (_safe_normalized(head_anchor - neck_anchor, neck_dir), 0.18),
        (neck_dir, 0.82),
        fallback=neck_dir,
    )

    upper_chest_len = shoulder_width * 0.04
    neck_len = shoulder_width * 0.06
    head_len = shoulder_width * 0.10
    pts["upper_chest"] = pts["chest"] + upper_chest_dir * upper_chest_len
    pts["neck"] = pts["upper_chest"] + neck_dir * neck_len
    pts["head"] = pts["neck"] + head_dir * head_len

    if shoulder_width > 1e-6:
        scale = 35.0 / shoulder_width
        for key in pts:
            pts[key] = pts[key] * scale

    return pts
