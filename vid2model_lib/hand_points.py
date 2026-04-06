from __future__ import annotations

from typing import Dict, Optional

import numpy as np


# MediaPipe Hand landmark indices
HAND_LM = {
    "wrist": 0,
    "thumb_cmc": 1,
    "thumb_mcp": 2,
    "thumb_ip": 3,
    "thumb_tip": 4,
    "index_mcp": 5,
    "index_pip": 6,
    "index_dip": 7,
    "index_tip": 8,
    "middle_mcp": 9,
    "middle_pip": 10,
    "middle_dip": 11,
    "middle_tip": 12,
    "ring_mcp": 13,
    "ring_pip": 14,
    "ring_dip": 15,
    "ring_tip": 16,
    "pinky_mcp": 17,
    "pinky_pip": 18,
    "pinky_dip": 19,
    "pinky_tip": 20,
}


def _first_hand_result(hand_results, side: str) -> Optional[tuple]:
    """Extract left or right hand from MediaPipe HandLandmarker results.

    Returns tuple of (landmarks, handedness) or None if not found.
    """
    if hand_results is None or not hasattr(hand_results, "handedness"):
        return None

    if not hand_results.hand_landmarks:
        return None

    handedness_list = hand_results.handedness

    for i, hand_landmarks in enumerate(hand_results.hand_landmarks):
        if i < len(handedness_list):
            handedness = handedness_list[i]
            # Handedness is a classification result with a label ("Left" or "Right")
            label = handedness.category_name if hasattr(handedness, "category_name") else str(handedness)

            if side.lower() == "left" and "Left" in label:
                return (hand_landmarks, label)
            elif side.lower() == "right" and "Right" in label:
                return (hand_landmarks, label)

    return None


def _landmark_to_array(lm) -> np.ndarray:
    """Convert MediaPipe landmark to numpy array [x, y, z]."""
    return np.array([lm.x, -lm.y, -lm.z], dtype=np.float64)


def extract_hand_points(
    hand_results,
    side: str,
    wrist_world: np.ndarray,
    forearm_length: float,
) -> Optional[Dict[str, np.ndarray]]:
    """Convert MediaPipe Hand landmarks to skeleton point names.

    Args:
        hand_results: MediaPipe HandLandmarker results object
        side: "left" or "right"
        wrist_world: World-space wrist position from Pose landmarks (np.ndarray [3])
        forearm_length: Length of forearm (distance from elbow to wrist) for scaling

    Returns:
        Dictionary mapping skeleton point names to 3D positions, or None if hand not detected
    """
    hand_data = _first_hand_result(hand_results, side)
    if hand_data is None:
        return None

    hand_landmarks, _ = hand_data

    # Extract hand landmarks in hand-relative coordinate frame
    # MediaPipe Hand coordinates are normalized relative to hand bounding box
    hand_lms = {}
    for name, idx in HAND_LM.items():
        if idx < len(hand_landmarks):
            hand_lms[name] = _landmark_to_array(hand_landmarks[idx])

    if "wrist" not in hand_lms:
        return None

    # Hand landmarks are in hand-relative frame (roughly normalized)
    # We need to transform them to world space

    # Hand WRIST landmark position in hand-relative frame
    wrist_hand_rel = hand_lms["wrist"]

    # Compute scale: distance from wrist to middle_tip in hand-relative frame
    # compared to forearm length gives us scaling factor
    if "middle_tip" in hand_lms:
        wrist_to_middle = np.linalg.norm(hand_lms["middle_tip"] - wrist_hand_rel)
        # Expected hand length is roughly 0.5 of forearm length (hand + fingers)
        # MediaPipe hand landmarks span about 0.8 normalized units
        expected_hand_scale = max(forearm_length * 0.5, 0.01)
        if wrist_to_middle > 1e-6:
            scale = expected_hand_scale / wrist_to_middle
        else:
            scale = expected_hand_scale
    else:
        scale = max(forearm_length * 0.5, 0.01)

    # Compute hand orientation vector (from wrist towards middle finger)
    if "middle_tip" in hand_lms:
        hand_dir = hand_lms["middle_tip"] - wrist_hand_rel
        hand_dir_len = np.linalg.norm(hand_dir)
        if hand_dir_len > 1e-6:
            hand_dir = hand_dir / hand_dir_len
        else:
            hand_dir = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        hand_dir = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # Create orthonormal basis for hand coordinate frame
    # hand_dir is forward, create right and up vectors
    # Use a cross product to get perpendicular vector
    if abs(hand_dir[0]) < 0.9:
        right = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float64), hand_dir)
    else:
        right = np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float64), hand_dir)

    right_len = np.linalg.norm(right)
    if right_len > 1e-6:
        right = right / right_len
    else:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    up = np.cross(hand_dir, right)
    up_len = np.linalg.norm(up)
    if up_len > 1e-6:
        up = up / up_len
    else:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # Transform each hand landmark from hand-relative to world space
    pts = {}

    for name, idx in HAND_LM.items():
        if idx < len(hand_landmarks):
            # Hand-relative position
            hand_rel_pos = hand_lms[name]

            # Transform to world space:
            # 1. Scale to real-world units
            # 2. Rotate by hand basis vectors
            # 3. Translate to wrist world position
            scaled_pos = hand_rel_pos * scale
            world_pos = (
                right * scaled_pos[0] +
                up * scaled_pos[1] +
                hand_dir * scaled_pos[2] +
                wrist_world
            )

            pts[f"{side}_{name}"] = world_pos

    # Map MediaPipe Hand landmark names to skeleton bone names
    result = {
        f"{side}_hand": pts[f"{side}_wrist"],
        # Thumb
        f"{side}_thumb_metacarpal": pts[f"{side}_thumb_cmc"],
        f"{side}_thumb_proximal": pts[f"{side}_thumb_mcp"],
        f"{side}_thumb_distal": pts[f"{side}_thumb_ip"],
        # Index
        f"{side}_index_proximal": pts[f"{side}_index_mcp"],
        f"{side}_index_intermediate": pts[f"{side}_index_pip"],
        f"{side}_index_distal": pts[f"{side}_index_dip"],
        # Middle
        f"{side}_middle_proximal": pts[f"{side}_middle_mcp"],
        f"{side}_middle_intermediate": pts[f"{side}_middle_pip"],
        f"{side}_middle_distal": pts[f"{side}_middle_dip"],
        # Ring
        f"{side}_ring_proximal": pts[f"{side}_ring_mcp"],
        f"{side}_ring_intermediate": pts[f"{side}_ring_pip"],
        f"{side}_ring_distal": pts[f"{side}_ring_dip"],
        # Pinky
        f"{side}_little_proximal": pts[f"{side}_pinky_mcp"],
        f"{side}_little_intermediate": pts[f"{side}_pinky_pip"],
        f"{side}_little_distal": pts[f"{side}_pinky_dip"],
    }

    return result
