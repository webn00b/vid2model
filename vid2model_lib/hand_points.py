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


def _world_lm_to_array(lm) -> np.ndarray:
    """Convert MediaPipe world landmark (meters, wrist-centered) to numpy [x, y, z]."""
    return np.array([lm.x, -lm.y, -lm.z], dtype=np.float64)


def _find_hand_index(hand_results, side: str) -> Optional[int]:
    """Find index of left/right hand in HandLandmarker results."""
    if hand_results is None:
        return None

    # Tasks API: hand_results.handedness is list of Category objects
    handedness_list = getattr(hand_results, "handedness", None)
    if not handedness_list:
        return None

    for i, handedness in enumerate(handedness_list):
        # Tasks API: handedness is a list with one Category per hand
        if isinstance(handedness, list):
            label = handedness[0].category_name if handedness else ""
        elif hasattr(handedness, "category_name"):
            label = handedness.category_name
        else:
            label = str(handedness)

        if side.lower() == "left" and "Left" in label:
            return i
        elif side.lower() == "right" and "Right" in label:
            return i

    return None


def extract_hand_points(
    hand_results,
    side: str,
    wrist_world: np.ndarray,
    forearm_length: float,
) -> Optional[Dict[str, np.ndarray]]:
    """Convert MediaPipe Hand world landmarks to skeleton point names.

    Uses hand_world_landmarks (real meters, wrist at origin) — no scaling needed,
    just translate to wrist_world position from Pose.

    Args:
        hand_results: MediaPipe HandLandmarker results object
        side: "left" or "right"
        wrist_world: World-space wrist position from Pose landmarks (np.ndarray [3])
        forearm_length: Not used (kept for API compatibility)

    Returns:
        Dictionary mapping skeleton point names to 3D positions, or None if hand not detected
    """
    hand_idx = _find_hand_index(hand_results, side)
    if hand_idx is None:
        return None

    # Prefer world landmarks (real meters, wrist at origin)
    world_landmarks_list = getattr(hand_results, "hand_world_landmarks", None)

    # Solutions API fallback
    if world_landmarks_list is None:
        world_landmarks_list = getattr(hand_results, "multi_hand_world_landmarks", None)

    if not world_landmarks_list or hand_idx >= len(world_landmarks_list):
        return None

    world_lms = world_landmarks_list[hand_idx]
    if len(world_lms) < 21:
        return None

    # hand_world_landmarks: wrist is at origin, coordinates are in meters
    # We just translate by wrist_world to place in pose space
    pts = {}
    for name, idx in HAND_LM.items():
        if idx < len(world_lms):
            # world landmark is relative to wrist (wrist = 0,0,0)
            # so add wrist_world to get absolute position
            pts[f"{side}_{name}"] = _world_lm_to_array(world_lms[idx]) + wrist_world

    if f"{side}_wrist" not in pts:
        return None

    return {
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
