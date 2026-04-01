from __future__ import annotations

from typing import Dict, Optional

import numpy as np


LM = {
    "nose": 0,
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


def _safe_normalized(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vec_norm = np.linalg.norm(vec)
    if vec_norm > 1e-8:
        return vec / vec_norm
    fallback_norm = np.linalg.norm(fallback)
    if fallback_norm > 1e-8:
        return fallback / fallback_norm
    return np.array([1.0, 0.0, 0.0], dtype=np.float64)


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
    # MediaPipe Tasks returns a list of poses; we use the first detected one.
    if getattr(res, "pose_world_landmarks", None):
        if not res.pose_world_landmarks:
            return None
        world_landmarks = res.pose_world_landmarks[0]

        def point(idx: int) -> np.ndarray:
            return np.array([world_landmarks[idx].x, -world_landmarks[idx].y, -world_landmarks[idx].z], dtype=np.float64)

    elif getattr(res, "pose_landmarks", None):
        if not res.pose_landmarks:
            return None
        landmarks = res.pose_landmarks[0]

        def point(idx: int) -> np.ndarray:
            return np.array([landmarks[idx].x - 0.5, -(landmarks[idx].y - 0.5), -landmarks[idx].z], dtype=np.float64)

    else:
        return None

    pts = {
        "left_shoulder": point(LM["left_shoulder"]),
        "right_shoulder": point(LM["right_shoulder"]),
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
    pts["neck"] = (pts["chest"] + pts["nose"]) * 0.5
    pts["upper_chest"] = (pts["chest"] + pts["neck"]) * 0.5
    pts["left_shoulder_clavicle"] = pts["chest"] + (pts["left_shoulder"] - pts["chest"]) * 0.35
    pts["right_shoulder_clavicle"] = pts["chest"] + (pts["right_shoulder"] - pts["chest"]) * 0.35
    pts["head"] = pts["nose"]

    shoulder_width = np.linalg.norm(pts["left_shoulder"] - pts["right_shoulder"])
    if shoulder_width > 1e-6:
        scale = 35.0 / shoulder_width
        for key in pts:
            pts[key] = pts[key] * scale

    return pts
