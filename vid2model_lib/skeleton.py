from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class JointDef:
    name: str
    parent: Optional[str]
    channels: int  # 6 for root, 3 for rotation, 0 for end site


JOINTS: List[JointDef] = [
    JointDef("hips", None, 6),
    JointDef("spine", "hips", 3),
    JointDef("chest", "spine", 3),
    JointDef("upperChest", "chest", 3),
    JointDef("neck", "upperChest", 3),
    JointDef("head", "neck", 3),
    JointDef("leftShoulder", "chest", 3),
    JointDef("leftUpperArm", "leftShoulder", 3),
    JointDef("leftLowerArm", "leftUpperArm", 3),
    JointDef("leftHand", "leftLowerArm", 3),
    JointDef("leftThumbMetacarpal", "leftHand", 3),
    JointDef("leftThumbProximal", "leftThumbMetacarpal", 3),
    JointDef("leftThumbDistal", "leftThumbProximal", 3),
    JointDef("leftIndexProximal", "leftHand", 3),
    JointDef("leftIndexIntermediate", "leftIndexProximal", 3),
    JointDef("leftIndexDistal", "leftIndexIntermediate", 3),
    JointDef("leftMiddleProximal", "leftHand", 3),
    JointDef("leftMiddleIntermediate", "leftMiddleProximal", 3),
    JointDef("leftMiddleDistal", "leftMiddleIntermediate", 3),
    JointDef("leftRingProximal", "leftHand", 3),
    JointDef("leftRingIntermediate", "leftRingProximal", 3),
    JointDef("leftRingDistal", "leftRingIntermediate", 3),
    JointDef("leftLittleProximal", "leftHand", 3),
    JointDef("leftLittleIntermediate", "leftLittleProximal", 3),
    JointDef("leftLittleDistal", "leftLittleIntermediate", 3),
    JointDef("rightShoulder", "chest", 3),
    JointDef("rightUpperArm", "rightShoulder", 3),
    JointDef("rightLowerArm", "rightUpperArm", 3),
    JointDef("rightHand", "rightLowerArm", 3),
    JointDef("rightThumbMetacarpal", "rightHand", 3),
    JointDef("rightThumbProximal", "rightThumbMetacarpal", 3),
    JointDef("rightThumbDistal", "rightThumbProximal", 3),
    JointDef("rightIndexProximal", "rightHand", 3),
    JointDef("rightIndexIntermediate", "rightIndexProximal", 3),
    JointDef("rightIndexDistal", "rightIndexIntermediate", 3),
    JointDef("rightMiddleProximal", "rightHand", 3),
    JointDef("rightMiddleIntermediate", "rightMiddleProximal", 3),
    JointDef("rightMiddleDistal", "rightMiddleIntermediate", 3),
    JointDef("rightRingProximal", "rightHand", 3),
    JointDef("rightRingIntermediate", "rightRingProximal", 3),
    JointDef("rightRingDistal", "rightRingIntermediate", 3),
    JointDef("rightLittleProximal", "rightHand", 3),
    JointDef("rightLittleIntermediate", "rightLittleProximal", 3),
    JointDef("rightLittleDistal", "rightLittleIntermediate", 3),
    JointDef("leftUpperLeg", "hips", 3),
    JointDef("leftLowerLeg", "leftUpperLeg", 3),
    JointDef("leftFoot", "leftLowerLeg", 3),
    JointDef("leftToes", "leftFoot", 3),
    JointDef("rightUpperLeg", "hips", 3),
    JointDef("rightLowerLeg", "rightUpperLeg", 3),
    JointDef("rightFoot", "rightLowerLeg", 3),
    JointDef("rightToes", "rightFoot", 3),
]


CHILDREN: Dict[str, List[str]] = {}
for joint in JOINTS:
    CHILDREN.setdefault(joint.name, [])
    if joint.parent is not None:
        CHILDREN.setdefault(joint.parent, []).append(joint.name)


MAP_TO_POINTS = {
    "hips": ("mid_hip",),
    "spine": ("spine",),
    "chest": ("chest",),
    "upperChest": ("upper_chest",),
    "neck": ("neck",),
    "head": ("head",),
    "leftShoulder": ("left_shoulder_clavicle",),
    "leftUpperArm": ("left_shoulder",),
    "leftLowerArm": ("left_elbow",),
    "leftHand": ("left_wrist",),
    "leftThumbMetacarpal": ("left_thumb_metacarpal",),
    "leftThumbProximal": ("left_thumb_proximal",),
    "leftThumbDistal": ("left_thumb_distal",),
    "leftIndexProximal": ("left_index_proximal",),
    "leftIndexIntermediate": ("left_index_intermediate",),
    "leftIndexDistal": ("left_index_distal",),
    "leftMiddleProximal": ("left_middle_proximal",),
    "leftMiddleIntermediate": ("left_middle_intermediate",),
    "leftMiddleDistal": ("left_middle_distal",),
    "leftRingProximal": ("left_ring_proximal",),
    "leftRingIntermediate": ("left_ring_intermediate",),
    "leftRingDistal": ("left_ring_distal",),
    "leftLittleProximal": ("left_little_proximal",),
    "leftLittleIntermediate": ("left_little_intermediate",),
    "leftLittleDistal": ("left_little_distal",),
    "rightShoulder": ("right_shoulder_clavicle",),
    "rightUpperArm": ("right_shoulder",),
    "rightLowerArm": ("right_elbow",),
    "rightHand": ("right_wrist",),
    "rightThumbMetacarpal": ("right_thumb_metacarpal",),
    "rightThumbProximal": ("right_thumb_proximal",),
    "rightThumbDistal": ("right_thumb_distal",),
    "rightIndexProximal": ("right_index_proximal",),
    "rightIndexIntermediate": ("right_index_intermediate",),
    "rightIndexDistal": ("right_index_distal",),
    "rightMiddleProximal": ("right_middle_proximal",),
    "rightMiddleIntermediate": ("right_middle_intermediate",),
    "rightMiddleDistal": ("right_middle_distal",),
    "rightRingProximal": ("right_ring_proximal",),
    "rightRingIntermediate": ("right_ring_intermediate",),
    "rightRingDistal": ("right_ring_distal",),
    "rightLittleProximal": ("right_little_proximal",),
    "rightLittleIntermediate": ("right_little_intermediate",),
    "rightLittleDistal": ("right_little_distal",),
    "leftUpperLeg": ("left_hip",),
    "leftLowerLeg": ("left_knee",),
    "leftFoot": ("left_ankle",),
    "leftToes": ("left_toes",),
    "rightUpperLeg": ("right_hip",),
    "rightLowerLeg": ("right_knee",),
    "rightFoot": ("right_ankle",),
    "rightToes": ("right_toes",),
}
