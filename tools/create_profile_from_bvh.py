#!/usr/bin/env python3
"""
Create a rig profile for a BVH file by analyzing its skeleton structure.
Usage: python3 tools/create_profile_from_bvh.py output/poklon.bvh
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

def parse_bvh_skeleton(bvh_path: str) -> Dict[str, Dict]:
    """Parse BVH file and extract bone hierarchy with offsets using regex."""
    bones = {}
    parent_stack = []
    in_endsite = False
    endsite_indent = -1

    with open(bvh_path, 'r') as f:
        content = f.read()

    # Find HIERARCHY section
    hierarchy_start = content.find("HIERARCHY")
    motion_start = content.find("MOTION")
    if hierarchy_start == -1 or motion_start == -1:
        return bones

    hierarchy = content[hierarchy_start:motion_start]
    lines = hierarchy.split('\n')

    in_hierarchy = False
    pending_bone = None
    pending_indent = -1

    for line in lines:
        stripped = line.strip()

        if stripped == "HIERARCHY":
            in_hierarchy = True
            continue

        if not in_hierarchy or not stripped:
            continue

        indent = len(line) - len(line.lstrip())

        # Handle End Site blocks
        if stripped == "End Site":
            in_endsite = True
            endsite_indent = indent
            continue

        if in_endsite:
            # Exit End Site when we see a closing brace at same or lower indent
            if stripped == "}":
                if indent <= endsite_indent:
                    in_endsite = False
            # Skip all other lines inside End Site
            continue

        # Pop parent stack based on indent
        while parent_stack and parent_stack[-1][1] >= indent:
            parent_stack.pop()

        # Handle ROOT
        if stripped.startswith("ROOT "):
            name = stripped.split()[1]
            bones[name] = {
                "name": name,
                "parent": None,
                "offset": [0.0, 0.0, 0.0],
                "children": []
            }
            pending_bone = name
            pending_indent = indent
            continue

        # Handle JOINT
        if stripped.startswith("JOINT "):
            name = stripped.split()[1]
            parent = parent_stack[-1][0] if parent_stack else None
            bones[name] = {
                "name": name,
                "parent": parent,
                "offset": [0.0, 0.0, 0.0],
                "children": []
            }
            if parent and parent in bones:
                bones[parent]["children"].append(name)
            pending_bone = name
            pending_indent = indent
            continue

        # Handle opening brace
        if stripped == "{":
            if pending_bone:
                parent_stack.append((pending_bone, pending_indent))
                pending_bone = None
            continue

        # Handle OFFSET
        if stripped.startswith("OFFSET"):
            parts = stripped.split()
            if len(parts) >= 4:
                try:
                    offset = [float(parts[1]), float(parts[2]), float(parts[3])]
                    # Find the current bone (last one on parent stack)
                    if parent_stack:
                        bone_name = parent_stack[-1][0]
                        if bone_name in bones:
                            bones[bone_name]["offset"] = offset
                except (ValueError, IndexError):
                    pass

    return bones

def calculate_scale_from_proportions(poklon_bones: Dict, reference_bones: Dict) -> float:
    """Calculate scale factor based on skeleton proportions."""
    # Calculate total skeleton height by summing spine offsets
    poklon_spine_height = 0
    ref_spine_height = 0

    spine_chain = ["hips", "spine", "chest", "upperChest", "neck", "head"]
    for bone_name in spine_chain:
        if bone_name in poklon_bones:
            poklon_spine_height += poklon_bones[bone_name]["offset"][1]
        if bone_name in reference_bones:
            ref_spine_height += reference_bones[bone_name]["offset"][1]

    if poklon_spine_height > 0 and ref_spine_height > 0:
        scale = ref_spine_height / poklon_spine_height
        return max(0.5, min(2.0, scale))  # Clamp between 0.5 and 2.0

    return 1.0

def create_bone_mapping(bvh_bones: Dict) -> Dict[str, str]:
    """Create mapping from canonical VRM names to source BVH names."""
    canonical_names = [
        "hips", "spine", "chest", "upperChest", "neck", "head",
        "leftShoulder", "rightShoulder",
        "leftUpperArm", "rightUpperArm", "leftLowerArm", "rightLowerArm",
        "leftHand", "rightHand",
        "leftUpperLeg", "rightUpperLeg", "leftLowerLeg", "rightLowerLeg",
        "leftFoot", "rightFoot",
        "leftToes", "rightToes",
        "leftThumbMetacarpal", "leftThumbProximal", "leftThumbDistal",
        "leftIndexProximal", "leftIndexIntermediate", "leftIndexDistal",
        "leftMiddleProximal", "leftMiddleIntermediate", "leftMiddleDistal",
        "leftRingProximal", "leftRingIntermediate", "leftRingDistal",
        "leftLittleProximal", "leftLittleIntermediate", "leftLittleDistal",
        "rightThumbMetacarpal", "rightThumbProximal", "rightThumbDistal",
        "rightIndexProximal", "rightIndexIntermediate", "rightIndexDistal",
        "rightMiddleProximal", "rightMiddleIntermediate", "rightMiddleDistal",
        "rightRingProximal", "rightRingIntermediate", "rightRingDistal",
        "rightLittleProximal", "rightLittleIntermediate", "rightLittleDistal",
    ]

    bvh_bone_names = set(bvh_bones.keys())
    mapping = {}

    for canonical_name in canonical_names:
        # Try exact match first
        if canonical_name in bvh_bone_names:
            mapping[canonical_name] = canonical_name
        else:
            # Try case-insensitive match
            lower_name = canonical_name.lower()
            for bvh_name in bvh_bone_names:
                if bvh_name.lower() == lower_name:
                    mapping[canonical_name] = bvh_name
                    break

    return mapping

def create_profile_json(bvh_path: str, reference_bvh_path: Optional[str] = None) -> Dict:
    """Create a rig profile JSON from BVH skeleton analysis."""
    bvh_name = Path(bvh_path).stem

    poklon_bones = parse_bvh_skeleton(bvh_path)
    ref_bones = parse_bvh_skeleton(reference_bvh_path) if reference_bvh_path else {}

    bone_mapping = create_bone_mapping(poklon_bones)
    pos_scale = calculate_scale_from_proportions(poklon_bones, ref_bones) if ref_bones else 1.0

    # Inner profile object
    profile_obj = {
        "id": f"auto-{bvh_name}",
        "modelLabel": f"{bvh_name}.bvh",
        "stage": "body",
        "namesTargetToSource": bone_mapping,
        "mode": "builtin-manual-map",
        "posScale": pos_scale,
        "yawOffset": 0.0,
        "preferredMode": "skeletonutils-skinnedmesh",
        "forceLiveDelta": False,
        "notes": f"Auto-generated profile from {bvh_name}.bvh skeleton analysis"
    }

    # Outer export format (as expected by register_rig_profile.py)
    exported_profile = {
        "format": "vid2model.rig-profile.v1",
        "modelFingerprint": f"auto:{bvh_name}",
        "stage": "body",
        "profile": profile_obj
    }

    return exported_profile

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 tools/create_profile_from_bvh.py <bvh_file> [reference_bvh]")
        sys.exit(1)

    bvh_file = sys.argv[1]
    reference_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(bvh_file).exists():
        print(f"Error: BVH file not found: {bvh_file}", file=sys.stderr)
        sys.exit(1)

    if reference_file and not Path(reference_file).exists():
        print(f"Error: Reference BVH file not found: {reference_file}", file=sys.stderr)
        sys.exit(1)

    profile = create_profile_json(bvh_file, reference_file)

    # Print JSON
    print(json.dumps(profile, indent=2))

    # Also save to file
    bvh_stem = Path(bvh_file).stem
    output_path = Path(bvh_file).parent / f"{bvh_stem}.rig-profile.json"
    with open(output_path, 'w') as f:
        json.dump(profile, f, indent=2)

    print(f"\n✓ Profile saved to: {output_path}", file=sys.stderr)
    pos_scale = profile.get('profile', {}).get('posScale', 1.0)
    print(f"✓ posScale: {pos_scale:.2f}", file=sys.stderr)
