from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

from .pipeline import convert_video_to_bvh
from .writers import write_bvh, write_csv, write_json, write_npz, write_trc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert video to BVH/JSON/CSV/NPZ/TRC using MediaPipe Pose.")
    parser.add_argument("--input", help="Path to input video file (mp4/webm/mov/etc)")
    parser.add_argument("--output", help="Legacy output path for BVH (same as --output-bvh).")
    parser.add_argument("--output-bvh", help="Path to output BVH file.")
    parser.add_argument("--output-json", help="Path to output JSON file.")
    parser.add_argument("--output-csv", help="Path to output CSV file (frame channels).")
    parser.add_argument("--output-npz", help="Path to output NPZ file (compressed arrays).")
    parser.add_argument("--output-trc", help="Path to output TRC file (marker trajectories).")
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--max-gap-interpolate", type=int, default=8, help="Interpolate missing detections for gaps up to N frames.")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N frames (0 disables).")
    parser.add_argument("--check-tools", action="store_true", help="Validate local toolchain and exit.")
    return parser.parse_args()


def check_tools() -> int:
    py_ok = (3, 10) <= sys.version_info[:2] <= (3, 12)
    cv2_ok = importlib.util.find_spec("cv2") is not None
    mediapipe_ok = importlib.util.find_spec("mediapipe") is not None
    blender_bin = shutil.which("blender")
    if blender_bin is None and Path("/Applications/Blender.app/Contents/MacOS/Blender").exists():
        blender_bin = "/Applications/Blender.app/Contents/MacOS/Blender"

    print(f"python: {sys.version.split()[0]} ({'ok' if py_ok else 'warn: recommended 3.10-3.12'})")
    print(f"opencv(cv2): {'ok' if cv2_ok else 'missing'}")
    print(f"mediapipe: {'ok' if mediapipe_ok else 'missing'}")
    print(f"blender: {blender_bin if blender_bin else 'not found (optional for FBX)'}")

    return 0 if (cv2_ok and mediapipe_ok) else 1


def main() -> int:
    args = parse_args()
    if args.check_tools:
        return check_tools()

    if not args.input:
        raise ValueError("Specify --input (or use --check-tools).")

    input_path = Path(args.input).expanduser().resolve()
    output_bvh = Path(args.output_bvh).expanduser().resolve() if args.output_bvh else None
    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else None
    output_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else None
    output_npz = Path(args.output_npz).expanduser().resolve() if args.output_npz else None
    output_trc = Path(args.output_trc).expanduser().resolve() if args.output_trc else None

    if args.output:
        if output_bvh is not None:
            raise ValueError("Use either --output or --output-bvh, not both.")
        output_bvh = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if (
        output_bvh is None
        and output_json is None
        and output_csv is None
        and output_npz is None
        and output_trc is None
    ):
        raise ValueError(
            "Specify at least one target: --output-bvh, --output-json, --output-csv, --output-npz, --output-trc (or --output)."
        )

    fps, rest_offsets, motion_values, ref_root, frames_pts = convert_video_to_bvh(
        input_path=input_path,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        max_gap_interpolate=args.max_gap_interpolate,
        progress_every=args.progress_every,
    )

    if output_bvh is not None:
        write_bvh(output_bvh, fps, rest_offsets, motion_values)
        print(f"Saved BVH: {output_bvh}")
    if output_json is not None:
        write_json(output_json, input_path, fps, rest_offsets, motion_values, ref_root)
        print(f"Saved JSON: {output_json}")
    if output_csv is not None:
        write_csv(output_csv, motion_values)
        print(f"Saved CSV: {output_csv}")
    if output_npz is not None:
        write_npz(output_npz, input_path, fps, rest_offsets, motion_values, ref_root)
        print(f"Saved NPZ: {output_npz}")
    if output_trc is not None:
        write_trc(output_trc, input_path, fps, frames_pts, ref_root)
        print(f"Saved TRC: {output_trc}")

    return 0
