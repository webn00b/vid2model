from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .pipeline import bvh_hierarchy_lines
from .skeleton import JOINTS, MAP_TO_POINTS


def channel_headers() -> List[str]:
    root_name = JOINTS[0].name
    headers = [
        f"{root_name}_Xposition",
        f"{root_name}_Yposition",
        f"{root_name}_Zposition",
        f"{root_name}_Zrotation",
        f"{root_name}_Xrotation",
        f"{root_name}_Yrotation",
    ]
    for joint in JOINTS[1:]:
        headers.extend(
            [
                f"{joint.name}_Zrotation",
                f"{joint.name}_Xrotation",
                f"{joint.name}_Yrotation",
            ]
        )
    return headers


def write_bvh(
    output_path: Path,
    fps: float,
    rest_offsets: Dict[str, np.ndarray],
    motion_values: List[List[float]],
) -> None:
    hierarchy = bvh_hierarchy_lines(rest_offsets)
    motion_lines = [" ".join(f"{v:.6f}" for v in row) for row in motion_values]
    bvh_text = "\n".join(
        hierarchy
        + ["MOTION", f"Frames: {len(motion_lines)}", f"Frame Time: {1.0 / fps:.8f}"]
        + motion_lines
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(bvh_text, encoding="utf-8")


def write_json(
    output_path: Path,
    input_path: Path,
    fps: float,
    rest_offsets: Dict[str, np.ndarray],
    motion_values: List[List[float]],
    ref_root: np.ndarray,
) -> None:
    joints = [
        {
            "name": joint.name,
            "parent": joint.parent,
            "channels": joint.channels,
            "rest_offset": [float(v) for v in rest_offsets.get(joint.name, np.zeros(3))],
        }
        for joint in JOINTS
    ]

    channel_layout = [
        {
            "joint": JOINTS[0].name,
            "channels": ["Xposition", "Yposition", "Zposition", "Zrotation", "Xrotation", "Yrotation"],
        }
    ]
    channel_layout.extend(
        [{"joint": joint.name, "channels": ["Zrotation", "Xrotation", "Yrotation"]} for joint in JOINTS[1:]]
    )

    payload = {
        "metadata": {
            "input_video": str(input_path),
            "fps": float(fps),
            "frame_time": float(1.0 / fps),
            "frame_count": len(motion_values),
            "coordinate_system": "Y-up",
        },
        "skeleton": {"joints": joints, "root_reference": [float(v) for v in ref_root]},
        "motion": {"channel_layout": channel_layout, "frames": motion_values},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(output_path: Path, motion_values: List[List[float]]) -> None:
    headers = channel_headers()
    lines = [",".join(headers)]
    for row in motion_values:
        lines.append(",".join(f"{v:.6f}" for v in row))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_npz(
    output_path: Path,
    input_path: Path,
    fps: float,
    rest_offsets: Dict[str, np.ndarray],
    motion_values: List[List[float]],
    ref_root: np.ndarray,
) -> None:
    joint_names = np.array([joint.name for joint in JOINTS], dtype="<U32")
    joint_parents = np.array([joint.parent if joint.parent is not None else "" for joint in JOINTS], dtype="<U32")
    joint_channels = np.array([joint.channels for joint in JOINTS], dtype=np.int32)
    rest = np.array([rest_offsets.get(joint.name, np.zeros(3)) for joint in JOINTS], dtype=np.float32)
    motion = np.array(motion_values, dtype=np.float32)
    headers = np.array(channel_headers(), dtype="<U64")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output_path),
        input_video=str(input_path),
        fps=np.float32(fps),
        frame_time=np.float32(1.0 / fps),
        frame_count=np.int32(len(motion_values)),
        joint_names=joint_names,
        joint_parents=joint_parents,
        joint_channels=joint_channels,
        rest_offsets=rest,
        root_reference=np.array(ref_root, dtype=np.float32),
        channel_headers=headers,
        motion=motion,
    )


def write_trc(
    output_path: Path,
    input_path: Path,
    fps: float,
    frames_pts: List[Dict[str, np.ndarray]],
    ref_root: np.ndarray,
) -> None:
    marker_joints = [joint.name for joint in JOINTS]
    marker_point_keys = [MAP_TO_POINTS[name][0] for name in marker_joints]

    lines: List[str] = []
    lines.append(f"PathFileType\t4\t(X/Y/Z)\t{input_path.name}")
    lines.append("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames")
    lines.append(f"{fps:.6f}\t{fps:.6f}\t{len(frames_pts)}\t{len(marker_joints)}\tmm\t{fps:.6f}\t1\t{len(frames_pts)}")

    header_markers = ["Frame#", "Time"] + marker_joints
    lines.append("\t".join(header_markers))

    xyz_header = ["", ""]
    for idx in range(1, len(marker_joints) + 1):
        xyz_header.extend([f"X{idx}", f"Y{idx}", f"Z{idx}"])
    lines.append("\t".join(xyz_header))
    lines.append("")

    for frame_idx, pts in enumerate(frames_pts, start=1):
        t = (frame_idx - 1) / fps
        row = [str(frame_idx), f"{t:.6f}"]
        for key in marker_point_keys:
            p_mm = (pts[key] - ref_root) * 10.0
            row.extend([f"{p_mm[0]:.6f}", f"{p_mm[1]:.6f}", f"{p_mm[2]:.6f}"])
        lines.append("\t".join(row))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
