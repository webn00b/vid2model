from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class JointDef:
    name: str
    parent: Optional[str]
    channels: int  # 6 for root, 3 for rotation, 0 for end site


JOINTS: List[JointDef] = [
    JointDef("Hips", None, 6),
    JointDef("Spine", "Hips", 3),
    JointDef("Chest", "Spine", 3),
    JointDef("Neck", "Chest", 3),
    JointDef("Head", "Neck", 3),
    JointDef("LeftShoulder", "Chest", 3),
    JointDef("LeftElbow", "LeftShoulder", 3),
    JointDef("LeftWrist", "LeftElbow", 3),
    JointDef("RightShoulder", "Chest", 3),
    JointDef("RightElbow", "RightShoulder", 3),
    JointDef("RightWrist", "RightElbow", 3),
    JointDef("LeftHip", "Hips", 3),
    JointDef("LeftKnee", "LeftHip", 3),
    JointDef("LeftAnkle", "LeftKnee", 3),
    JointDef("RightHip", "Hips", 3),
    JointDef("RightKnee", "RightHip", 3),
    JointDef("RightAnkle", "RightKnee", 3),
]


CHILDREN: Dict[str, List[str]] = {}
for joint in JOINTS:
    CHILDREN.setdefault(joint.name, [])
    if joint.parent is not None:
        CHILDREN.setdefault(joint.parent, []).append(joint.name)


MAP_TO_POINTS = {
    "Hips": ("mid_hip",),
    "Spine": ("spine",),
    "Chest": ("chest",),
    "Neck": ("neck",),
    "Head": ("head",),
    "LeftShoulder": ("left_shoulder",),
    "LeftElbow": ("left_elbow",),
    "LeftWrist": ("left_wrist",),
    "RightShoulder": ("right_shoulder",),
    "RightElbow": ("right_elbow",),
    "RightWrist": ("right_wrist",),
    "LeftHip": ("left_hip",),
    "LeftKnee": ("left_knee",),
    "LeftAnkle": ("left_ankle",),
    "RightHip": ("right_hip",),
    "RightKnee": ("right_knee",),
    "RightAnkle": ("right_ankle",),
}
