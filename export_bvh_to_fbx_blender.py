import sys
from pathlib import Path

import bpy


def parse_args() -> tuple[Path, Path]:
    if "--" not in sys.argv:
        raise SystemExit("Expected '-- <input.bvh> <output.fbx>'")
    args = sys.argv[sys.argv.index("--") + 1 :]
    if len(args) != 2:
        raise SystemExit("Usage: blender -b --python export_bvh_to_fbx_blender.py -- <input.bvh> <output.fbx>")
    in_bvh = Path(args[0]).expanduser().resolve()
    out_fbx = Path(args[1]).expanduser().resolve()
    if not in_bvh.exists():
        raise SystemExit(f"Input BVH not found: {in_bvh}")
    out_fbx.parent.mkdir(parents=True, exist_ok=True)
    return in_bvh, out_fbx


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)



def main() -> int:
    in_bvh, out_fbx = parse_args()
    clear_scene()

    bpy.ops.import_anim.bvh(
        filepath=str(in_bvh),
        axis_forward="-Z",
        axis_up="Y",
        rotate_mode="NATIVE",
    )

    # Export only the animated armature.
    bpy.ops.export_scene.fbx(
        filepath=str(out_fbx),
        check_existing=False,
        use_selection=False,
        object_types={"ARMATURE"},
        add_leaf_bones=False,
        bake_anim=True,
        path_mode="AUTO",
    )

    print(f"Saved FBX: {out_fbx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
