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
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
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
        "left_hip": point(LM["left_hip"]),
        "right_hip": point(LM["right_hip"]),
        "left_knee": point(LM["left_knee"]),
        "right_knee": point(LM["right_knee"]),
        "left_ankle": point(LM["left_ankle"]),
        "right_ankle": point(LM["right_ankle"]),
        "nose": point(LM["nose"]),
    }

    pts["mid_hip"] = (pts["left_hip"] + pts["right_hip"]) * 0.5
    pts["spine"] = (pts["mid_hip"] + (pts["left_shoulder"] + pts["right_shoulder"]) * 0.5) * 0.5
    pts["chest"] = (pts["left_shoulder"] + pts["right_shoulder"]) * 0.5
    pts["neck"] = (pts["chest"] + pts["nose"]) * 0.5
    pts["head"] = pts["nose"]

    shoulder_width = np.linalg.norm(pts["left_shoulder"] - pts["right_shoulder"])
    if shoulder_width > 1e-6:
        scale = 35.0 / shoulder_width
        for key in pts:
            pts[key] = pts[key] * scale

    return pts
