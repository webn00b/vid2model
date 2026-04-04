"""Retarget BVH animation onto a VRM model via Blender + VRM Add-on.

Usage:
    blender -b --python retarget_bvh_to_vrm_blender.py -- <input.bvh> <model.vrm> <output.vrm>

Requires: VRM Add-on for Blender (https://vrm-addon-for-blender.info/)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
import mathutils


# ---------------------------------------------------------------------------
# BVH joint names → VRM humanoid bone names.
# vid2model BVH already uses VRM naming, so most map 1:1.
# ---------------------------------------------------------------------------
BVH_TO_VRM_HUMANOID: dict[str, str] = {
    "hips": "hips",
    "spine": "spine",
    "chest": "chest",
    "upperChest": "upperChest",
    "neck": "neck",
    "head": "head",
    "leftShoulder": "leftShoulder",
    "leftUpperArm": "leftUpperArm",
    "leftLowerArm": "leftLowerArm",
    "leftHand": "leftHand",
    "rightShoulder": "rightShoulder",
    "rightUpperArm": "rightUpperArm",
    "rightLowerArm": "rightLowerArm",
    "rightHand": "rightHand",
    "leftUpperLeg": "leftUpperLeg",
    "leftLowerLeg": "leftLowerLeg",
    "leftFoot": "leftFoot",
    "leftToes": "leftToes",
    "rightUpperLeg": "rightUpperLeg",
    "rightLowerLeg": "rightLowerLeg",
    "rightFoot": "rightFoot",
    "rightToes": "rightToes",
    # Fingers
    "leftThumbMetacarpal": "leftThumbMetacarpal",
    "leftThumbProximal": "leftThumbProximal",
    "leftThumbDistal": "leftThumbDistal",
    "leftIndexProximal": "leftIndexProximal",
    "leftIndexIntermediate": "leftIndexIntermediate",
    "leftIndexDistal": "leftIndexDistal",
    "leftMiddleProximal": "leftMiddleProximal",
    "leftMiddleIntermediate": "leftMiddleIntermediate",
    "leftMiddleDistal": "leftMiddleDistal",
    "leftRingProximal": "leftRingProximal",
    "leftRingIntermediate": "leftRingIntermediate",
    "leftRingDistal": "leftRingDistal",
    "leftLittleProximal": "leftLittleProximal",
    "leftLittleIntermediate": "leftLittleIntermediate",
    "leftLittleDistal": "leftLittleDistal",
    "rightThumbMetacarpal": "rightThumbMetacarpal",
    "rightThumbProximal": "rightThumbProximal",
    "rightThumbDistal": "rightThumbDistal",
    "rightIndexProximal": "rightIndexProximal",
    "rightIndexIntermediate": "rightIndexIntermediate",
    "rightIndexDistal": "rightIndexDistal",
    "rightMiddleProximal": "rightMiddleProximal",
    "rightMiddleIntermediate": "rightMiddleIntermediate",
    "rightMiddleDistal": "rightMiddleDistal",
    "rightRingProximal": "rightRingProximal",
    "rightRingIntermediate": "rightRingIntermediate",
    "rightRingDistal": "rightRingDistal",
    "rightLittleProximal": "rightLittleProximal",
    "rightLittleIntermediate": "rightLittleIntermediate",
    "rightLittleDistal": "rightLittleDistal",
}


def parse_args() -> tuple[Path, Path, Path]:
    if "--" not in sys.argv:
        raise SystemExit(
            "Expected '-- <input.bvh> <model.vrm> <output.vrm>'"
        )
    args = sys.argv[sys.argv.index("--") + 1 :]
    if len(args) != 3:
        raise SystemExit(
            "Usage: blender -b --python retarget_bvh_to_vrm_blender.py "
            "-- <input.bvh> <model.vrm> <output.vrm>"
        )
    in_bvh = Path(args[0]).expanduser().resolve()
    in_vrm = Path(args[1]).expanduser().resolve()
    out_vrm = Path(args[2]).expanduser().resolve()
    if not in_bvh.exists():
        raise SystemExit(f"BVH not found: {in_bvh}")
    if not in_vrm.exists():
        raise SystemExit(f"VRM not found: {in_vrm}")
    out_vrm.parent.mkdir(parents=True, exist_ok=True)
    return in_bvh, in_vrm, out_vrm


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)


def find_armature(hint: str) -> bpy.types.Object:
    """Return the first armature object whose name contains *hint*."""
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE" and hint.lower() in obj.name.lower():
            return obj
    # Fallback: any armature
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            return obj
    raise RuntimeError(f"No armature found (hint={hint!r})")


def _camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case: leftUpperArm -> left_upper_arm."""
    import re
    s = re.sub(r"([A-Z])", r"_\1", name).lower().lstrip("_")
    return s


def get_vrm_humanoid_map(vrm_armature: bpy.types.Object) -> dict[str, str]:
    """Build VRM_humanoid_name → actual_bone_name map from VRM extension data.

    Returns a dict keyed by VRM humanoid names as used in BVH_TO_VRM_HUMANOID
    (camelCase, e.g. "leftUpperArm").
    """
    # The VRM addon stores data as a Blender property group accessible
    # via attribute access (not dict .get()).
    raw: dict[str, str] = {}  # snake_case attr -> bone name

    try:
        ext = vrm_armature.data.vrm_addon_extension
    except AttributeError:
        ext = None

    # VRM 1.0 path
    if ext:
        vrm1 = getattr(ext, "vrm1", None)
        if vrm1:
            humanoid = getattr(vrm1, "humanoid", None)
            if humanoid:
                human_bones = getattr(humanoid, "human_bones", None)
                if human_bones:
                    for attr_name in dir(human_bones):
                        if attr_name.startswith("_") or attr_name in (
                            "bl_rna", "rna_type",
                        ):
                            continue
                        bone_prop = getattr(human_bones, attr_name, None)
                        if (
                            bone_prop
                            and hasattr(bone_prop, "node")
                            and hasattr(bone_prop.node, "bone_name")
                        ):
                            bone_name = bone_prop.node.bone_name
                            if bone_name:
                                raw[attr_name] = bone_name

    # VRM 0.x path
    if not raw and ext:
        vrm0 = getattr(ext, "vrm0", None)
        if vrm0:
            humanoid = getattr(vrm0, "humanoid", None)
            if humanoid:
                human_bones = getattr(humanoid, "human_bones", [])
                for hb in human_bones:
                    bone_node = getattr(hb, "node", None)
                    bone_name = (
                        getattr(bone_node, "bone_name", "")
                        if bone_node
                        else ""
                    )
                    human_name = getattr(hb, "bone", "")
                    if bone_name and human_name:
                        raw[human_name] = bone_name

    # Build final mapping keyed by camelCase VRM humanoid names.
    # The addon uses snake_case attrs (left_upper_arm), but
    # BVH_TO_VRM_HUMANOID uses camelCase (leftUpperArm).
    mapping: dict[str, str] = {}
    # Build reverse lookup: snake_case -> camelCase
    snake_to_camel: dict[str, str] = {}
    for camel_name in BVH_TO_VRM_HUMANOID.values():
        snake_to_camel[_camel_to_snake(camel_name)] = camel_name

    for attr_name, bone_name in raw.items():
        camel = snake_to_camel.get(attr_name, attr_name)
        mapping[camel] = bone_name

    if mapping:
        return mapping

    # Last resort: assume bone names match VRM humanoid names directly
    bone_names = {b.name for b in vrm_armature.data.bones}
    for vrm_name in BVH_TO_VRM_HUMANOID.values():
        if vrm_name in bone_names:
            mapping[vrm_name] = vrm_name

    return mapping


def measure_chain_length(
    armature_data, bone_names: list[str], rest_dict: dict[str, mathutils.Matrix],
) -> float:
    """Sum bone lengths along a chain of bone names."""
    total = 0.0
    for name in bone_names:
        if name not in rest_dict:
            continue
        bone = armature_data.bones.get(name)
        if bone:
            total += bone.length
    return total


def compute_skeleton_height(
    armature_data,
    rest_dict: dict[str, mathutils.Matrix],
    humanoid_map: dict[str, str],
    is_source: bool = True,
) -> float:
    """Estimate skeleton height from hips to head + hips to foot.

    For source (BVH) bones use names directly; for target (VRM) use
    humanoid_map to resolve actual bone names.
    """
    def resolve(humanoid_name: str) -> str:
        if is_source:
            return humanoid_name
        return humanoid_map.get(humanoid_name, "")

    # Spine chain: hips -> spine -> chest -> upperChest -> neck -> head
    spine_chain = [
        resolve(n)
        for n in ("spine", "chest", "upperChest", "neck", "head")
        if resolve(n)
    ]
    # Leg chain: leftUpperLeg -> leftLowerLeg -> leftFoot
    leg_chain = [
        resolve(n)
        for n in ("leftUpperLeg", "leftLowerLeg", "leftFoot")
        if resolve(n)
    ]

    spine_len = measure_chain_length(armature_data, spine_chain, rest_dict)
    leg_len = measure_chain_length(armature_data, leg_chain, rest_dict)

    height = spine_len + leg_len
    return height if height > 0.001 else 1.0


def compute_limb_scales(
    bvh_data, vrm_data,
    bvh_rest: dict[str, mathutils.Matrix],
    vrm_rest: dict[str, mathutils.Matrix],
    humanoid_map: dict[str, str],
) -> dict[str, float]:
    """Compute per-bone scale ratios for limb length compensation."""
    scales: dict[str, float] = {}

    # Pairs: (bvh_bone, vrm_humanoid_name)
    # We compute scale for each bone as vrm_length / bvh_length
    for bvh_name, vrm_humanoid_name in BVH_TO_VRM_HUMANOID.items():
        vrm_bone_name = humanoid_map.get(vrm_humanoid_name)
        if not vrm_bone_name:
            continue
        bvh_bone = bvh_data.bones.get(bvh_name)
        vrm_bone = vrm_data.bones.get(vrm_bone_name)
        if not bvh_bone or not vrm_bone:
            continue
        if bvh_bone.length > 0.0001 and vrm_bone.length > 0.0001:
            scales[bvh_name] = vrm_bone.length / bvh_bone.length

    return scales


def retarget(
    bvh_armature: bpy.types.Object,
    vrm_armature: bpy.types.Object,
    humanoid_map: dict[str, str],
) -> int:
    """Copy BVH animation to VRM armature bone-by-bone with rest-pose
    compensation and automatic proportion scaling.

    Returns number of bones successfully retargeted.
    """
    scene = bpy.context.scene

    # Determine frame range from BVH action
    bvh_action = bvh_armature.animation_data and bvh_armature.animation_data.action
    if not bvh_action:
        raise RuntimeError("BVH armature has no animation data")

    frame_start = int(bvh_action.frame_range[0])
    frame_end = int(bvh_action.frame_range[1])
    scene.frame_start = frame_start
    scene.frame_end = frame_end

    # Ensure VRM armature has animation data
    if not vrm_armature.animation_data:
        vrm_armature.animation_data_create()
    vrm_action = bpy.data.actions.new(name="Retargeted")
    vrm_armature.animation_data.action = vrm_action

    # Build rest-pose matrices for both armatures
    bvh_rest: dict[str, mathutils.Matrix] = {}
    vrm_rest: dict[str, mathutils.Matrix] = {}

    for bone in bvh_armature.data.bones:
        bvh_rest[bone.name] = bone.matrix_local.copy()

    for bone in vrm_armature.data.bones:
        vrm_rest[bone.name] = bone.matrix_local.copy()

    # --- Proportion scaling ---
    bvh_height = compute_skeleton_height(
        bvh_armature.data, bvh_rest, humanoid_map, is_source=True,
    )
    vrm_height = compute_skeleton_height(
        vrm_armature.data, vrm_rest, humanoid_map, is_source=False,
    )
    height_scale = vrm_height / bvh_height if bvh_height > 0.001 else 1.0
    print(f"Skeleton heights: BVH={bvh_height:.4f}  VRM={vrm_height:.4f}  scale={height_scale:.4f}")

    limb_scales = compute_limb_scales(
        bvh_armature.data, vrm_armature.data,
        bvh_rest, vrm_rest, humanoid_map,
    )

    # Build parent-relative rest matrices
    def parent_relative_rest(armature_data, bone_name: str, rest_dict: dict) -> mathutils.Matrix:
        bone = armature_data.bones[bone_name]
        if bone.parent:
            return rest_dict[bone.parent.name].inverted() @ rest_dict[bone_name]
        return rest_dict[bone_name].copy()

    # Map BVH bone names to VRM bone names
    bone_pairs: list[tuple[str, str]] = []  # (bvh_bone, vrm_bone)
    for bvh_name, vrm_humanoid_name in BVH_TO_VRM_HUMANOID.items():
        vrm_bone_name = humanoid_map.get(vrm_humanoid_name)
        if not vrm_bone_name:
            continue
        if bvh_name not in bvh_rest:
            continue
        if vrm_bone_name not in vrm_rest:
            continue
        bone_pairs.append((bvh_name, vrm_bone_name))

    if not bone_pairs:
        raise RuntimeError(
            "No bone pairs found for retarget. "
            "Check that VRM model has humanoid bone mapping."
        )

    print(f"Retargeting {len(bone_pairs)} bones, frames {frame_start}-{frame_end}")

    # Determine which BVH bone is root (has position channels)
    root_bvh_bone = "hips"
    root_vrm_bone = humanoid_map.get("hips")

    # Pre-compute rest-pose compensation quaternions per bone pair
    rest_compensations: dict[str, mathutils.Quaternion] = {}
    for bvh_name, vrm_bone_name in bone_pairs:
        bvh_local_rest = parent_relative_rest(
            bvh_armature.data, bvh_name, bvh_rest
        )
        vrm_local_rest = parent_relative_rest(
            vrm_armature.data, vrm_bone_name, vrm_rest
        )
        bvh_rest_rot = bvh_local_rest.to_quaternion()
        vrm_rest_rot = vrm_local_rest.to_quaternion()
        rest_compensations[bvh_name] = (
            vrm_rest_rot.inverted() @ bvh_rest_rot
        )

    # Switch to pose mode on BVH armature to read posed transforms
    bpy.context.view_layer.objects.active = bvh_armature
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.context.view_layer.objects.active = vrm_armature
    bpy.ops.object.mode_set(mode="POSE")

    # Iterate frames and copy transforms
    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)

        for bvh_name, vrm_bone_name in bone_pairs:
            bvh_pose_bone = bvh_armature.pose.bones.get(bvh_name)
            vrm_pose_bone = vrm_armature.pose.bones.get(vrm_bone_name)
            if not bvh_pose_bone or not vrm_pose_bone:
                continue

            # Get BVH bone's local rotation (pose-space, relative to rest)
            if bvh_pose_bone.rotation_mode == "QUATERNION":
                bvh_rot = bvh_pose_bone.rotation_quaternion.copy()
            else:
                bvh_rot = bvh_pose_bone.matrix_basis.to_quaternion()

            # Apply rest-pose compensation
            comp = rest_compensations[bvh_name]
            vrm_rot = comp @ bvh_rot @ comp.inverted()

            # Set rotation
            vrm_pose_bone.rotation_mode = "QUATERNION"
            vrm_pose_bone.rotation_quaternion = vrm_rot
            vrm_pose_bone.keyframe_insert(
                data_path="rotation_quaternion", frame=frame
            )

            # Copy root position with height-based scaling
            if bvh_name == root_bvh_bone and vrm_bone_name == root_vrm_bone:
                bvh_loc = bvh_pose_bone.location.copy()
                vrm_pose_bone.location = bvh_loc * height_scale
                vrm_pose_bone.keyframe_insert(
                    data_path="location", frame=frame
                )

    bpy.ops.object.mode_set(mode="OBJECT")
    print(f"Retarget complete: {len(bone_pairs)} bones, {frame_end - frame_start + 1} frames")
    return len(bone_pairs)


def ensure_vrm_addon() -> None:
    """Enable VRM addon through Blender preferences (required for headless)."""
    try:
        bpy.ops.preferences.addon_enable(module="bl_ext.blender_org.vrm")
    except Exception:
        # Fallback for older Blender / different addon naming
        try:
            import addon_utils
            addon_utils.enable("VRM_Addon_for_Blender")
        except Exception:
            pass
    # Verify
    if not hasattr(bpy.ops.import_scene, "vrm"):
        raise RuntimeError(
            "VRM Add-on for Blender not available. "
            "Install from https://vrm-addon-for-blender.info/"
        )


def main() -> int:
    in_bvh, in_vrm, out_vrm = parse_args()
    clear_scene()
    ensure_vrm_addon()

    # Import VRM
    print(f"Importing VRM: {in_vrm}")
    bpy.ops.import_scene.vrm(filepath=str(in_vrm))
    vrm_armature = find_armature("armature")
    vrm_armature.name = "VRM_Armature"

    # Get humanoid bone mapping from VRM data
    humanoid_map = get_vrm_humanoid_map(vrm_armature)
    print(f"VRM humanoid bones found: {len(humanoid_map)}")
    if humanoid_map:
        for hname, bname in sorted(humanoid_map.items()):
            print(f"  {hname} -> {bname}")

    # Import BVH
    print(f"Importing BVH: {in_bvh}")
    bpy.ops.import_anim.bvh(
        filepath=str(in_bvh),
        axis_forward="-Z",
        axis_up="Y",
        rotate_mode="NATIVE",
    )
    bvh_armature = None
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE" and obj is not vrm_armature:
            bvh_armature = obj
            break
    if not bvh_armature:
        raise RuntimeError("Failed to find BVH armature after import")
    bvh_armature.name = "BVH_Armature"

    # Retarget
    n_bones = retarget(bvh_armature, vrm_armature, humanoid_map)

    # Remove BVH armature before export
    bpy.ops.object.select_all(action="DESELECT")
    bvh_armature.select_set(True)
    bpy.ops.object.delete()

    # Select VRM armature and all its children for export
    bpy.ops.object.select_all(action="SELECT")

    # Export VRM
    print(f"Exporting VRM: {out_vrm}")
    bpy.ops.export_scene.vrm(filepath=str(out_vrm))

    print(f"Done! Retargeted {n_bones} bones -> {out_vrm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
